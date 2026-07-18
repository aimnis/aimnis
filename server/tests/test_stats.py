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


# ---- hit satisfaction (explicit rejects + implicit near-duplicate retries) ---- #

DIM = 384


def _vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


async def _backdate(pool, minutes: int) -> None:
    """Shift the most recently inserted lookup_event into the past."""
    await pool.execute(
        "UPDATE lookup_event SET ts = now() - make_interval(mins => $1) "
        "WHERE id = (SELECT max(id) FROM lookup_event)",
        minutes,
    )


async def test_satisfaction_quiet_hit_is_satisfied(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="a")
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", entry_id=e,
                             client_hash="c1", embedding=_vec(0))
    await _backdate(clean, 20)  # older than the window, no follow-up

    sat = await stats.hit_satisfaction(clean)
    assert (sat.hits_scored, sat.dissatisfied, sat.pending) == (1, 0, 0)
    assert sat.satisfaction_rate == 1.0


async def test_satisfaction_near_duplicate_retry_flags_hit(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="a")
    await stats.record_event(clean, query_hash="a", outcome="hit_semantic", entry_id=e,
                             client_hash="c1", embedding=_vec(0))
    await _backdate(clean, 20)
    # Same client re-asks a near-verbatim question (identical embedding) in-window.
    await stats.record_event(clean, query_hash="a2", outcome="miss",
                             client_hash="c1", embedding=_vec(0))
    await _backdate(clean, 15)

    sat = await stats.hit_satisfaction(clean)
    # Only the hit is scored (the retry itself is a miss); it counts as implicit
    # dissatisfaction, not an explicit reject.
    assert (sat.hits_scored, sat.dissatisfied, sat.explicit_rejects) == (1, 1, 0)
    assert sat.satisfaction_rate == 0.0


async def test_satisfaction_explicit_reject_flags_hit(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="a")
    await stats.record_event(clean, query_hash="a", outcome="hit_semantic", entry_id=e,
                             client_hash="c1", embedding=_vec(0))
    await _backdate(clean, 20)
    # Explicit reject retry: different wording (orthogonal embedding), but names the entry.
    await stats.record_event(clean, query_hash="b", outcome="miss",
                             client_hash="c1", embedding=_vec(5), rejected_entry=e)
    await _backdate(clean, 15)

    sat = await stats.hit_satisfaction(clean)
    assert sat.dissatisfied == 1
    assert sat.explicit_rejects == 1


async def test_satisfaction_follow_up_and_other_client_do_not_flag(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="a")
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", entry_id=e,
                             client_hash="c1", embedding=_vec(0))
    await _backdate(clean, 20)
    # Topic follow-up (orthogonal embedding, same client, in-window): not a retry.
    await stats.record_event(clean, query_hash="f", outcome="miss",
                             client_hash="c1", embedding=_vec(5))
    await _backdate(clean, 15)
    # Near-duplicate but from a DIFFERENT client: says nothing about c1's hit.
    await stats.record_event(clean, query_hash="a2", outcome="miss",
                             client_hash="c2", embedding=_vec(0))
    await _backdate(clean, 14)

    sat = await stats.hit_satisfaction(clean)
    assert (sat.hits_scored, sat.dissatisfied) == (1, 0)
    assert sat.satisfaction_rate == 1.0


async def test_satisfaction_fresh_hit_is_pending(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="a")
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", entry_id=e,
                             client_hash="c1", embedding=_vec(0))  # ts = now()

    sat = await stats.hit_satisfaction(clean)
    assert (sat.hits_scored, sat.pending) == (0, 1)


async def test_satisfaction_ignores_untracked_and_out_of_window(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="a")
    # No client identity (local/stdio) → not scored at all.
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", entry_id=e,
                             embedding=_vec(0))
    await _backdate(clean, 30)
    # Tracked hit; near-duplicate retry arrives AFTER the window closed.
    await stats.record_event(clean, query_hash="a", outcome="hit_exact", entry_id=e,
                             client_hash="c1", embedding=_vec(0))
    await _backdate(clean, 25)
    await stats.record_event(clean, query_hash="a2", outcome="miss",
                             client_hash="c1", embedding=_vec(0))
    await _backdate(clean, 11)

    sat = await stats.hit_satisfaction(clean)
    assert (sat.hits_scored, sat.dissatisfied) == (1, 0)


async def test_latency_stats_by_outcome(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="h", hit_count=1)
    await stats.record_event(clean, query_hash="h", outcome="hit_exact", entry_id=e,
                             latency_ms=400, user_agent="claude-code/2.0")
    await stats.record_event(clean, query_hash="h", outcome="hit_exact", entry_id=e,
                             latency_ms=600, user_agent="claude-code/2.0")
    await stats.record_event(clean, query_hash="m", outcome="miss",
                             latency_ms=16000, user_agent="opencode/1.0")
    lat = await stats.latency_stats(clean)
    assert lat.samples == 3
    bo = {o: (n, p50, p95) for (o, n, p50, p95) in lat.by_outcome}
    assert bo["hit_exact"][0] == 2
    assert bo["hit_exact"][1] == 500.0        # p50 of [400, 600]
    assert bo["miss"] == (1, 16000.0, 16000.0)


async def test_latency_stats_empty(clean):
    lat = await stats.latency_stats(clean)
    assert lat.samples == 0
    assert lat.overall_p50_ms == 0.0
    assert lat.by_outcome == []


async def test_search_user_agents_breakdown(clean):
    e = await _seed_entry(clean, query_text="q", query_hash="h", hit_count=1)
    await stats.record_event(clean, query_hash="h", outcome="hit_exact", entry_id=e,
                             user_agent="claude-code/2.0")
    await stats.record_event(clean, query_hash="h", outcome="hit_semantic", entry_id=e,
                             user_agent="claude-code/2.0")
    await stats.record_event(clean, query_hash="m", outcome="miss", user_agent="opencode/1.0")
    await stats.record_event(clean, query_hash="e", outcome="error", user_agent="opencode/1.0")
    await stats.record_event(clean, query_hash="n", outcome="miss")   # no UA → (none)
    apps = await stats.search_user_agents(clean)
    d = {ua: (n, h) for (ua, n, h) in apps}
    assert d["claude-code/2.0"] == (2, 2)     # two searches, both hits
    assert d["opencode/1.0"] == (1, 0)        # error row excluded; one miss, no hits
    assert d["(none)"] == (1, 0)              # stdio/local search, unattributed
