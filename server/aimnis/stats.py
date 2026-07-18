"""Flywheel statistics — the Gate 1 pass/kill instrument.

Gate 1 turns on one curve: cache HIT RATE rising as the corpus grows. This
module reads the append-only `lookup_event` log (plus `pool_entry` for corpus
size) and reports both the all-time snapshot and a recent-window rate so the
slope's direction is visible after only a handful of queries.

Reads only; never spends quota. `record_event` (the write side) lives here too
so the log's schema stays in one place.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import asyncpg

from .config import settings

_HIT_OUTCOMES = ("hit_exact", "hit_semantic")


@dataclass(frozen=True)
class Stats:
    corpus_total: int          # every pool_entry row
    corpus_servable: int       # active + unexpired + opted-in (what a lookup can serve)
    lookups_total: int         # answerable lookups: hits + misses (errors excluded)
    hits: int
    hits_exact: int
    hits_semantic: int
    misses: int
    errors: int
    hit_rate: float            # hits / lookups_total (0.0 when no lookups yet)
    recent_window: int         # size of the trailing window below
    recent_hit_rate: float     # hit rate over the last `recent_window` answerable lookups
    top_queries: list          # [(query_text, hit_count), ...] most-reused entries
    clicks_total: int          # cited-source follow-throughs (signed /r redirects)
    avg_results_per_reply: float  # mean cited sources returned per answerable reply


async def record_event(
    pool: asyncpg.Pool,
    *,
    query_hash: str,
    outcome: str,
    distance: float | None = None,
    entry_id: str | None = None,
    niche: str | None = None,
    rerank_score: float | None = None,
    result_count: int | None = None,
    client_hash: str | None = None,
    embedding: list[float] | None = None,
    rejected_entry: str | None = None,
    latency_ms: int | None = None,
    user_agent: str | None = None,
) -> None:
    """Append one lookup to the log. Best-effort: observability must never break
    a search, so a logging failure is swallowed rather than propagated.

    `rerank_score` is the cross-encoder score (0..1) of the best semantic
    candidate — set on a semantic hit (accepted) and on a miss that had a
    rejected candidate — so rerank_min_score can be tuned against real traffic.
    `result_count` is the number of cited sources returned in the reply (NULL on
    error/empty replies) — averaged for grounding-richness-per-reply.
    `client_hash` + `embedding` power hit-satisfaction detection (a near-duplicate
    re-ask by the same client within the window marks the prior hit dissatisfied);
    `rejected_entry` records an explicit reject_entry=<id> retry. `latency_ms` is
    the end-to-end resolve_search time and `user_agent` the calling application
    (NULL for stdio/local) — both tied to this search's outcome for per-outcome
    latency and per-app breakdowns."""
    try:
        await pool.execute(
            "INSERT INTO lookup_event "
            "(query_hash, outcome, distance, entry_id, niche, rerank_score, result_count, "
            " client_hash, embedding, rejected_entry, latency_ms, user_agent) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)",
            query_hash,
            outcome,
            distance,
            entry_id,
            niche,
            rerank_score,
            result_count,
            client_hash,
            embedding,
            rejected_entry,
            latency_ms,
            (user_agent or "")[:512] or None,
        )
    except Exception:  # noqa: BLE001 — never let stats logging fail the answer path
        pass


@dataclass(frozen=True)
class LatencyStats:
    """Per-outcome search latency (ms) from lookup_event.latency_ms. p50/p95 by
    outcome make the 'instant from cache, escalate on miss' promise measurable.
    Older lookups predate the column (NULL) and are excluded."""

    samples: int                 # lookups with a recorded latency
    overall_p50_ms: float
    overall_p95_ms: float
    by_outcome: list             # [(outcome, count, p50_ms, p95_ms), ...]


async def latency_stats(pool: asyncpg.Pool) -> LatencyStats:
    ov = await pool.fetchrow(
        "SELECT count(*) AS n, "
        "percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50, "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95 "
        "FROM lookup_event WHERE latency_ms IS NOT NULL"
    )
    by = await pool.fetch(
        "SELECT outcome, count(*) AS n, "
        "percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50, "
        "percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95 "
        "FROM lookup_event WHERE latency_ms IS NOT NULL "
        "GROUP BY outcome ORDER BY outcome"
    )
    return LatencyStats(
        samples=ov["n"] or 0,
        overall_p50_ms=float(ov["p50"] or 0.0),
        overall_p95_ms=float(ov["p95"] or 0.0),
        by_outcome=[
            (r["outcome"], r["n"], float(r["p50"] or 0.0), float(r["p95"] or 0.0)) for r in by
        ],
    )


async def search_user_agents(pool: asyncpg.Pool, *, top_n: int = 10) -> list:
    """Which client applications ran searches (from lookup_event.user_agent), with
    each app's hit count. Per-source detail → gated to the authenticated /v1/stats.
    stdio/local searches carry no UA and bucket as '(none)'."""
    rows = await pool.fetch(
        "SELECT coalesce(user_agent, '(none)') AS ua, count(*) AS searches, "
        "count(*) FILTER (WHERE outcome IN ('hit_exact','hit_semantic')) AS hits "
        "FROM lookup_event WHERE outcome <> 'error' "
        "GROUP BY coalesce(user_agent, '(none)') ORDER BY searches DESC, ua LIMIT $1",
        top_n,
    )
    return [(r["ua"], r["searches"], r["hits"]) for r in rows]


async def gather(pool: asyncpg.Pool, *, recent_window: int = 20, top_n: int = 5) -> Stats:
    corpus = await pool.fetchrow(
        "SELECT count(*) AS total, "
        "       count(*) FILTER (WHERE status='active' AND opt_in "
        "                        AND (expires_at IS NULL OR expires_at > now())) AS servable "
        "FROM pool_entry"
    )

    counts = await pool.fetchrow(
        "SELECT "
        "  count(*) FILTER (WHERE outcome='hit_exact')    AS hits_exact, "
        "  count(*) FILTER (WHERE outcome='hit_semantic') AS hits_semantic, "
        "  count(*) FILTER (WHERE outcome='miss')         AS misses, "
        "  count(*) FILTER (WHERE outcome='error')        AS errors, "
        "  avg(result_count) FILTER "
        "    (WHERE outcome IN ('hit_exact','hit_semantic','miss')) AS avg_results "
        "FROM lookup_event"
    )
    hits_exact = counts["hits_exact"]
    hits_semantic = counts["hits_semantic"]
    hits = hits_exact + hits_semantic
    misses = counts["misses"]
    errors = counts["errors"]
    answerable = hits + misses
    hit_rate = hits / answerable if answerable else 0.0

    # Trailing-window rate over answerable lookups (excludes errors from both parts).
    recent = await pool.fetch(
        "SELECT outcome FROM lookup_event "
        "WHERE outcome <> 'error' ORDER BY id DESC LIMIT $1",
        recent_window,
    )
    recent_hits = sum(1 for r in recent if r["outcome"] in _HIT_OUTCOMES)
    recent_hit_rate = recent_hits / len(recent) if recent else 0.0

    top = await pool.fetch(
        "SELECT query_text, hit_count FROM pool_entry "
        "WHERE hit_count > 0 ORDER BY hit_count DESC, updated_at DESC LIMIT $1",
        top_n,
    )

    clicks_total = await pool.fetchval("SELECT count(*) FROM citation_click") or 0

    return Stats(
        corpus_total=corpus["total"],
        corpus_servable=corpus["servable"],
        lookups_total=answerable,
        hits=hits,
        hits_exact=hits_exact,
        hits_semantic=hits_semantic,
        misses=misses,
        errors=errors,
        hit_rate=hit_rate,
        recent_window=len(recent),
        recent_hit_rate=recent_hit_rate,
        top_queries=[(r["query_text"], r["hit_count"]) for r in top],
        clicks_total=clicks_total,
        avg_results_per_reply=float(counts["avg_results"] or 0.0),
    )


@dataclass(frozen=True)
class FlywheelPoint:
    unique_queries: int      # cumulative distinct queries seen (= cumulative misses), the x-axis
    lookups: int             # cumulative answerable lookups (hits + misses)
    hit_rate: float          # cumulative hit rate at this point
    rolling_hit_rate: float  # hit rate over the trailing window at this point


async def flywheel_series(
    pool: asyncpg.Pool, *, window: int = 20, max_points: int = 60, limit: int = 5000
) -> list[FlywheelPoint]:
    """The Gate 1 curve: cache hit rate as cumulative unique queries grow. Walks
    the lookup log in order (errors excluded — neither hit nor answerable-miss),
    tracking cumulative + trailing-window hit rate, then downsamples for plotting.
    The thesis passes if the curve slopes UP."""
    rows = await pool.fetch(
        "SELECT outcome FROM lookup_event WHERE outcome <> 'error' ORDER BY id LIMIT $1",
        limit,
    )
    pts: list[FlywheelPoint] = []
    hits = uniques = 0
    win: deque[int] = deque(maxlen=window)
    for i, r in enumerate(rows, 1):
        is_hit = r["outcome"] in _HIT_OUTCOMES
        if is_hit:
            hits += 1
        else:
            uniques += 1  # a miss is a newly-seen unique query
        win.append(1 if is_hit else 0)
        pts.append(FlywheelPoint(
            unique_queries=uniques, lookups=i,
            hit_rate=hits / i, rolling_hit_rate=sum(win) / len(win),
        ))

    if len(pts) > max_points:
        step = len(pts) / max_points
        sampled = [pts[int(k * step)] for k in range(max_points)]
        if sampled[-1] is not pts[-1]:
            sampled.append(pts[-1])
        return sampled
    return pts


@dataclass(frozen=True)
class StorageStats:
    """Pool storage footprint. `total_bytes` is pool_entry incl. TOAST + indexes
    (the whole knowledge pool; the append-only logs are separate). `bytes_per_entry`
    extrapolates linearly for the projection — it reflects the CURRENT answer/source
    mix, so a corpus that later skews more distilled/wide will trend upward."""
    total_bytes: int
    entries: int
    bytes_per_entry: float


async def storage_stats(pool: asyncpg.Pool) -> StorageStats:
    total = await pool.fetchval("SELECT pg_total_relation_size('pool_entry')") or 0
    entries = await pool.fetchval("SELECT count(*) FROM pool_entry") or 0
    return StorageStats(
        total_bytes=total,
        entries=entries,
        bytes_per_entry=(total / entries) if entries else 0.0,
    )


@dataclass(frozen=True)
class ClickAnalytics:
    """Aggregate citation-click signal for the dashboard. Reads only.

    `follow_through` = clicks per served hit (clicks_total / hits) — a rough read on
    how often a served answer sent the agent on to a source (high ⇒ answers may be
    thin; near-zero with many hits ⇒ answers are self-sufficient). All aggregate;
    no per-user data exists to report."""
    clicks_total: int
    follow_through: float             # clicks_total / hits (0.0 when no hits)
    top_hosts: list                   # [(host, clicks), ...] most-followed destinations
    top_entries: list                 # [(query_text, clicks, hit_count), ...] most-followed answers


async def click_analytics(pool: asyncpg.Pool, *, top_n: int = 8) -> ClickAnalytics:
    total = await pool.fetchval("SELECT count(*) FROM citation_click") or 0
    hits = await pool.fetchval(
        "SELECT count(*) FROM lookup_event WHERE outcome IN ('hit_exact','hit_semantic')"
    ) or 0

    hosts = await pool.fetch(
        "SELECT host, count(*) AS n FROM citation_click "
        "WHERE host IS NOT NULL GROUP BY host ORDER BY n DESC, host LIMIT $1",
        top_n,
    )
    entries = await pool.fetch(
        "SELECT pe.query_text, count(*) AS clicks, pe.hit_count "
        "FROM citation_click cc JOIN pool_entry pe ON pe.id = cc.entry_id "
        "GROUP BY pe.id, pe.query_text, pe.hit_count "
        "ORDER BY clicks DESC, pe.hit_count DESC LIMIT $1",
        top_n,
    )
    return ClickAnalytics(
        clicks_total=total,
        follow_through=(total / hits) if hits else 0.0,
        top_hosts=[(r["host"], r["n"]) for r in hosts],
        top_entries=[(r["query_text"], r["clicks"], r["hit_count"]) for r in entries],
    )


@dataclass(frozen=True)
class SatisfactionStats:
    """Hit-satisfaction: did callers accept the cached answers we served?

    A served hit is DISSATISFIED when the same client, within
    `satisfaction_window_minutes`, either explicitly rejected the entry
    (reject_entry=<id> retry) or re-asked a near-duplicate question (embedding
    distance < `satisfaction_requery_max_distance` — far tighter than the serve
    threshold, so topic follow-ups don't count). Hits younger than the window are
    `pending` (their client could still retry) and excluded from the rate.
    Aggregate only — the per-client sequences never leave the database."""
    hits_scored: int          # hits old enough to judge, with client identity
    dissatisfied: int         # retried (implicit) or rejected (explicit) in-window
    explicit_rejects: int     # subset of dissatisfied with a reject_entry retry
    pending: int              # hits still inside the window — not yet scored
    satisfaction_rate: float  # 1 - dissatisfied/hits_scored (0.0 when unscored)


async def hit_satisfaction(pool: asyncpg.Pool) -> SatisfactionStats:
    window = settings.satisfaction_window_minutes
    row = await pool.fetchrow(
        """
        SELECT
          count(*) FILTER (WHERE NOT pending)                   AS scored,
          count(*) FILTER (WHERE NOT pending AND bad)           AS dissatisfied,
          count(*) FILTER (WHERE NOT pending AND explicit)      AS explicit_rejects,
          count(*) FILTER (WHERE pending)                       AS pending
        FROM (
          SELECT
            h.ts > now() - make_interval(mins => $1) AS pending,
            EXISTS (
              SELECT 1 FROM lookup_event r
              WHERE r.client_hash = h.client_hash AND r.id > h.id
                AND r.ts <= h.ts + make_interval(mins => $1)
                AND (r.rejected_entry = h.entry_id
                     OR (r.embedding IS NOT NULL AND h.embedding IS NOT NULL
                         AND (r.embedding <=> h.embedding) < $2))
            ) AS bad,
            EXISTS (
              SELECT 1 FROM lookup_event r
              WHERE r.client_hash = h.client_hash AND r.id > h.id
                AND r.ts <= h.ts + make_interval(mins => $1)
                AND r.rejected_entry = h.entry_id
            ) AS explicit
          FROM lookup_event h
          WHERE h.outcome IN ('hit_exact','hit_semantic')
            AND h.client_hash IS NOT NULL
        ) t
        """,
        window,
        settings.satisfaction_requery_max_distance,
    )
    scored = row["scored"] or 0
    dissatisfied = row["dissatisfied"] or 0
    return SatisfactionStats(
        hits_scored=scored,
        dissatisfied=dissatisfied,
        explicit_rejects=row["explicit_rejects"] or 0,
        pending=row["pending"] or 0,
        satisfaction_rate=(1 - dissatisfied / scored) if scored else 0.0,
    )


@dataclass(frozen=True)
class RerankCalibration:
    """Score distribution for tuning rerank_min_score. `buckets` are 10 counts over
    [0.0, 1.0) in 0.1-wide bins (index 0 = [0.0,0.1) … index 9 = [0.9,1.0])."""
    threshold: float          # the current rerank_min_score accept floor
    accepted: list[int]       # hit_semantic score histogram (all ≥ threshold)
    rejected: list[int]       # miss-with-candidate score histogram (all < threshold)
    accepted_clicked: list[int]  # subset of `accepted` whose served entry later earned ≥1 click
    accepted_total: int
    rejected_total: int


async def rerank_calibration(pool: asyncpg.Pool) -> RerankCalibration:
    """Histogram of rerank scores split by accepted (hit_semantic) vs rejected
    (miss that had a candidate). Reading the rejected bin just below the threshold
    tells you how many hits a lower floor would recover; a fat accepted low-bin
    warns the floor may be too loose. `accepted_clicked` overlays how many accepted
    hits in each bin earned a citation follow-through — a bin with accepts but ~no
    follow-through is a matching-quality warning (served but not engaged with; note
    a self-sufficient answer also needs no click, so read it as a soft signal).
    Reads only; spends no quota."""
    rows = await pool.fetch(
        "SELECT le.outcome, "
        "  least(greatest(width_bucket(le.rerank_score, 0.0, 1.0, 10), 1), 10) AS bucket, "
        "  (cc.entry_id IS NOT NULL) AS clicked, "
        "  count(*) AS n "
        "FROM lookup_event le "
        "LEFT JOIN (SELECT DISTINCT entry_id FROM citation_click) cc "
        "       ON cc.entry_id = le.entry_id "
        "WHERE le.rerank_score IS NOT NULL AND le.outcome IN ('hit_semantic','miss') "
        "GROUP BY le.outcome, bucket, clicked",
    )
    accepted = [0] * 10
    rejected = [0] * 10
    accepted_clicked = [0] * 10
    for r in rows:
        b = r["bucket"] - 1  # width_bucket is 1-based
        if r["outcome"] == "hit_semantic":
            accepted[b] += r["n"]
            if r["clicked"]:
                accepted_clicked[b] += r["n"]
        else:
            rejected[b] += r["n"]
    return RerankCalibration(
        threshold=settings.rerank_min_score,
        accepted=accepted,
        rejected=rejected,
        accepted_clicked=accepted_clicked,
        accepted_total=sum(accepted),
        rejected_total=sum(rejected),
    )


def format_for_agent(s: Stats) -> str:
    """Render stats as a compact text block for an agent / the dogfood console."""
    lines = [
        "[Aimnis · flywheel stats]",
        f"Corpus:      {s.corpus_total} entries ({s.corpus_servable} servable)",
        f"Lookups:     {s.lookups_total} answerable  "
        f"(+{s.errors} live-search errors)" if s.errors else
        f"Lookups:     {s.lookups_total} answerable",
        f"Hit rate:    {s.hit_rate:.0%} all-time  "
        f"({s.hits} hits: {s.hits_exact} exact + {s.hits_semantic} semantic, "
        f"{s.misses} misses)",
    ]
    if s.recent_window:
        lines.append(
            f"Recent:      {s.recent_hit_rate:.0%} over last {s.recent_window} lookups"
        )
    if s.top_queries:
        lines.append("Most reused:")
        for q, n in s.top_queries:
            snippet = q if len(q) <= 70 else q[:67] + "..."
            lines.append(f"  {n:>4}×  {snippet}")
    lines.append("(hit-rate curve vs corpus size is the Gate 1 pass test; "
                 "target >~30% by 5–10k unique queries)")
    return "\n".join(lines)


# Convenience: current OpenRouter quota headroom (unused in the dogfood path, but
# the stats tool surfaces it so background-spend limits are visible once wired).
async def quota_headroom(pool: asyncpg.Pool, *, label: str | None = None) -> dict | None:
    row = await pool.fetchrow(
        "SELECT * FROM quota_usage($1)", label or settings.quota_key_label
    )
    if row is None:
        return None
    return {
        "minute_used": row["minute_used"],
        "minute_limit": row["minute_limit"],
        "day_used": row["day_used"],
        "day_limit": row["day_limit"],
    }
