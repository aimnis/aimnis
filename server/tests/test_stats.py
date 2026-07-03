"""Stats aggregation tests. Drive lookup_event + pool_entry directly (no network,
no embedding model) so the flywheel math is verified deterministically."""

from __future__ import annotations

from aimnis import stats


async def _seed_entry(pool, *, query_text, query_hash, hit_count=0):
    row = await pool.fetchrow(
        "INSERT INTO pool_entry (query_text, query_norm, query_hash, sources, "
        "model, status, hit_count) "
        "VALUES ($1,$2,$3,'[]'::jsonb,'searxng-live','active',$4) RETURNING id",
        query_text,
        query_text,
        query_hash,
        hit_count,
    )
    return str(row["id"])


async def test_click_analytics(clean):
    e1 = await _seed_entry(clean, query_text="enable http2 nginx", query_hash="a", hit_count=4)
    e2 = await _seed_entry(clean, query_text="pgvector index", query_hash="b", hit_count=1)
    # two served hits (the follow_through denominator)
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", entry_id=e1)
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", entry_id=e1)
    await clean.executemany(
        "INSERT INTO citation_click (entry_id, source_idx, host) VALUES ($1,$2,$3)",
        [(e1, 0, "nginx.org"), (e1, 1, "nginx.org"), (e2, 0, "github.com")],
    )

    ca = await stats.click_analytics(clean)
    assert ca.clicks_total == 3
    assert abs(ca.follow_through - 1.5) < 1e-9        # 3 clicks / 2 hits
    assert ca.top_hosts[0] == ("nginx.org", 2)        # most-followed host first
    assert ("github.com", 1) in ca.top_hosts
    assert ca.top_entries[0] == ("enable http2 nginx", 2, 4)  # (query, clicks, hit_count)


async def test_empty_pool_reports_zero(clean):
    s = await stats.gather(clean)
    assert s.corpus_total == 0
    assert s.lookups_total == 0
    assert s.hit_rate == 0.0
    assert s.top_queries == []
    assert s.avg_results_per_reply == 0.0


async def test_storage_stats(clean):
    empty = await stats.storage_stats(clean)
    assert empty.entries == 0
    assert empty.bytes_per_entry == 0.0  # no divide-by-zero on an empty pool
    assert empty.total_bytes >= 0

    await _seed_entry(clean, query_text="one entry", query_hash="a")
    one = await stats.storage_stats(clean)
    assert one.entries == 1
    assert one.total_bytes > 0
    assert one.bytes_per_entry == one.total_bytes  # total / 1


async def test_avg_results_per_reply(clean):
    # Averaged over answerable replies (hits + misses); errors (NULL count) excluded.
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", result_count=8)
    await stats.record_event(clean, query_hash="b", outcome="miss", result_count=4)
    await stats.record_event(clean, query_hash="c", outcome="error")  # NULL → ignored
    s = await stats.gather(clean)
    assert s.avg_results_per_reply == 6.0  # (8 + 4) / 2


async def test_hit_rate_and_breakdown(clean):
    eid = await _seed_entry(clean, query_text="q one", query_hash="h1", hit_count=3)
    await _seed_entry(clean, query_text="q two", query_hash="h2", hit_count=0)

    # Two misses (new entries), then three hits against the first entry.
    await stats.record_event(clean, query_hash="h1", outcome="miss", entry_id=eid)
    await stats.record_event(clean, query_hash="h2", outcome="miss")
    await stats.record_event(clean, query_hash="h1", outcome="hit_exact",
                             distance=0.0, entry_id=eid)
    await stats.record_event(clean, query_hash="h1", outcome="hit_semantic",
                             distance=0.07, entry_id=eid)
    await stats.record_event(clean, query_hash="h1", outcome="hit_exact",
                             distance=0.0, entry_id=eid)
    await stats.record_event(clean, query_hash="h9", outcome="error")

    s = await stats.gather(clean)
    assert s.corpus_total == 2
    assert s.corpus_servable == 2
    assert (s.hits, s.hits_exact, s.hits_semantic) == (3, 2, 1)
    assert s.misses == 2
    assert s.errors == 1
    # 3 hits / (3 hits + 2 misses) = 0.6; errors excluded from the denominator.
    assert abs(s.hit_rate - 0.6) < 1e-9
    assert s.top_queries[0] == ("q one", 3)


async def test_recent_window(clean):
    # 10 misses then 10 hits; a window of 5 should be all-hits (100%).
    for i in range(10):
        await stats.record_event(clean, query_hash=f"m{i}", outcome="miss")
    for i in range(10):
        await stats.record_event(clean, query_hash="h", outcome="hit_exact")

    s = await stats.gather(clean, recent_window=5)
    assert s.recent_window == 5
    assert s.recent_hit_rate == 1.0
    # All-time is 10/20 = 0.5, so the recent window shows the slope bending up.
    assert abs(s.hit_rate - 0.5) < 1e-9


async def test_rerank_calibration_buckets(clean):
    # Accepted semantic hits (scores ≥ threshold) and rejected near-misses
    # (logged on the miss). Buckets are 0.1-wide over [0,1].
    await stats.record_event(clean, query_hash="a", outcome="hit_semantic",
                             distance=0.08, rerank_score=0.94)   # bucket 9 [0.9,1.0)
    await stats.record_event(clean, query_hash="b", outcome="hit_semantic",
                             distance=0.10, rerank_score=0.62)   # bucket 6 [0.6,0.7)
    await stats.record_event(clean, query_hash="c", outcome="miss",
                             distance=0.12, rerank_score=0.41)   # bucket 4 [0.4,0.5)
    await stats.record_event(clean, query_hash="d", outcome="miss")  # true miss, no score
    await stats.record_event(clean, query_hash="e", outcome="hit_exact")  # no score

    calib = await stats.rerank_calibration(clean)
    assert calib.accepted_total == 2
    assert calib.rejected_total == 1
    assert calib.accepted[9] == 1 and calib.accepted[6] == 1
    assert calib.rejected[4] == 1
    # Exact hits and score-less misses are excluded from both histograms.
    assert sum(calib.accepted) == 2 and sum(calib.rejected) == 1


async def test_rerank_calibration_follow_through_overlay(clean):
    # An accepted semantic hit whose served entry later earns a click shows up in
    # accepted_clicked at its score bin; an accepted hit with no click does not.
    eid = await _seed_entry(clean, query_text="clicked entry", query_hash="a")
    await stats.record_event(clean, query_hash="a", outcome="hit_semantic",
                             rerank_score=0.94, entry_id=eid)      # bin 9, will be clicked
    await stats.record_event(clean, query_hash="b", outcome="hit_semantic",
                             rerank_score=0.62)                    # bin 6, no entry/click
    await clean.execute(
        "INSERT INTO citation_click (entry_id, source_idx, host) VALUES ($1,0,'x.io')", eid
    )

    calib = await stats.rerank_calibration(clean)
    assert calib.accepted[9] == 1 and calib.accepted_clicked[9] == 1  # engaged
    assert calib.accepted[6] == 1 and calib.accepted_clicked[6] == 0  # served, no follow-through


async def test_record_event_persists_rerank_score(clean):
    await stats.record_event(clean, query_hash="x", outcome="hit_semantic",
                             distance=0.05, rerank_score=0.87)
    val = await clean.fetchval(
        "SELECT rerank_score FROM lookup_event WHERE query_hash='x'"
    )
    assert abs(val - 0.87) < 1e-6


def test_format_shapes():
    s = stats.Stats(
        corpus_total=3, corpus_servable=3, lookups_total=5, hits=3,
        hits_exact=2, hits_semantic=1, misses=2, errors=0, hit_rate=0.6,
        recent_window=5, recent_hit_rate=0.8,
        top_queries=[("how to register pgvector", 3)],
        clicks_total=0,
        avg_results_per_reply=5.0,
    )
    out = stats.format_for_agent(s)
    assert "flywheel stats" in out
    assert "60%" in out          # all-time hit rate
    assert "80%" in out          # recent window
    assert "how to register pgvector" in out
