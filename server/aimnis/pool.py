"""Knowledge pool (== semantic cache) read/write.

Lookup is two-stage: an exact normalized-hash match first (fast, unambiguous),
then a vector nearest-neighbour within `cache_max_distance`. Only 'active',
unexpired, opted-in entries are servable. Embeddings are passed in by the caller
(computed via aimnis.embedding) so this module stays free of the ONNX dependency
and its logic is unit-testable with injected vectors.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import asyncpg

from .config import settings

_SERVABLE = "status = 'active' AND opt_in AND (expires_at IS NULL OR expires_at > now())"


@dataclass(frozen=True)
class Hit:
    id: str
    query_text: str
    answer_text: str | None
    sources: list
    model: str | None
    distance: float  # 0.0 for an exact hash match
    match: str  # 'exact' | 'semantic'
    content_at: datetime | None = None  # when the answer/sources were produced or last refreshed


async def insert(
    pool: asyncpg.Pool,
    *,
    query_text: str,
    query_norm: str,
    query_hash: str,
    embedding: list[float] | None,
    answer_text: str | None = None,
    sources: list | None = None,
    model: str | None = None,
    provenance: dict | None = None,
    status: str = "active",
    ttl_seconds: int | None = None,
    niche: str | None = None,
    quality_score: float | None = None,
    output_trainable: bool = False,
    attribution_required: bool = False,
    no_grounded_cache: bool = False,
) -> str:
    import json

    row = await pool.fetchrow(
        """
        INSERT INTO pool_entry (
            query_text, query_norm, query_hash, embedding,
            answer_text, sources, model, provenance, status,
            ttl_seconds, expires_at, niche, quality_score,
            output_trainable, attribution_required, no_grounded_cache
        )
        VALUES ($1,$2,$3,$4,$5,$6::jsonb,$7,$8::jsonb,$9,$10,
                CASE WHEN $10::int IS NULL THEN NULL ELSE now() + ($10 || ' seconds')::interval END,
                $11,$12,$13,$14,$15)
        RETURNING id
        """,
        query_text,
        query_norm,
        query_hash,
        embedding,
        answer_text,
        json.dumps(sources or []),
        model,
        json.dumps(provenance or {}),
        status,
        ttl_seconds,
        niche,
        quality_score,
        output_trainable,
        attribution_required,
        no_grounded_cache,
    )
    return str(row["id"])


async def lookup_exact(
    pool: asyncpg.Pool, *, query_hash: str, exclude: str | None = None
) -> Hit | None:
    """Exact normalized-hash match — unambiguous same-intent hit. Bumps hit_count
    (an exact match is always served). `exclude` skips one entry id — the caller's
    explicitly rejected entry must never be re-served on the retry."""
    exact = await pool.fetchrow(
        f"SELECT id, query_text, answer_text, sources, model, content_at FROM pool_entry "
        f"WHERE query_hash = $1 AND {_SERVABLE} "
        f"AND ($2::uuid IS NULL OR id <> $2::uuid) "
        f"ORDER BY updated_at DESC LIMIT 1",
        query_hash,
        exclude,
    )
    if exact is not None:
        await _bump_hit(pool, exact["id"])
        return _to_hit(exact, distance=0.0, match="exact")
    return None


async def candidates(
    pool: asyncpg.Pool,
    *,
    embedding: list[float],
    k: int = 5,
    max_distance: float | None = None,
    exclude: str | None = None,
) -> list[Hit]:
    """Top-k nearest servable entries within `max_distance`, nearest first.

    A candidate GENERATOR for the reranker — it does NOT decide the hit and does
    NOT bump hit_count (the caller bumps via mark_served only when it commits to
    serving one). All returned hits carry match='semantic'. `exclude` drops one
    entry id (the caller's explicitly rejected entry on a retry)."""
    threshold = settings.cache_max_distance if max_distance is None else max_distance
    rows = await pool.fetch(
        f"SELECT id, query_text, answer_text, sources, model, content_at, "
        f"       (embedding <=> $1) AS distance FROM pool_entry "
        f"WHERE {_SERVABLE} AND embedding IS NOT NULL "
        f"AND ($3::uuid IS NULL OR id <> $3::uuid) "
        f"ORDER BY embedding <=> $1 LIMIT $2",
        embedding,
        k,
        exclude,
    )
    return [
        _to_hit(r, distance=float(r["distance"]), match="semantic")
        for r in rows
        if r["distance"] <= threshold
    ]


async def mark_served(pool: asyncpg.Pool, entry_id) -> None:
    """Record that an entry was served as a hit (bumps hit_count + updated_at)."""
    await _bump_hit(pool, entry_id)


async def lookup(
    pool: asyncpg.Pool,
    *,
    query_hash: str,
    embedding: list[float] | None,
    max_distance: float | None = None,
    exclude: str | None = None,
) -> Hit | None:
    """Exact hash match first, then single nearest-neighbour within max_distance.

    Retained as the plain (no-rerank) lookup: used when reranking is disabled and
    by direct callers/tests. The reranked path in resolve uses lookup_exact +
    candidates + mark_served instead."""
    exact = await lookup_exact(pool, query_hash=query_hash, exclude=exclude)
    if exact is not None:
        return exact
    if embedding is None:
        return None
    threshold = settings.cache_max_distance if max_distance is None else max_distance
    top = await candidates(pool, embedding=embedding, k=1, max_distance=threshold,
                           exclude=exclude)
    if top:
        await mark_served(pool, top[0].id)
        return top[0]
    return None


async def select_refresh_candidates(
    pool: asyncpg.Pool,
    *,
    limit: int,
    min_quality: float,
    min_follow_through: float | None = None,
    min_hits_for_follow_through: int = 5,
) -> list[dict]:
    """Entries the background queue should re-distill, most-valuable first:
    never-distilled (snippet-only) → explicitly stale → below-min-quality →
    thin-answer (high click follow-through). Within that, highest follow-through,
    then most-hit and least-recently-updated first. Returns their STORED sources
    so refresh needs no new search.

    `min_follow_through` (clicks / hit_count) flags a distilled answer the agent
    keeps leaving to read sources — a thin-answer signal — but only once the entry
    has ≥ `min_hits_for_follow_through` hits so the ratio isn't trusted on noise.
    None disables the follow-through path (only the older reasons apply)."""
    import json
    import math

    # A sentinel above 1.0 disables the follow-through clause (clicks/hit ≤ 1 in
    # practice, but the column is uncapped; use +inf so nothing ever matches it).
    ft_threshold = math.inf if min_follow_through is None else min_follow_through

    rows = await pool.fetch(
        """
        WITH e AS (
            SELECT pe.id, pe.query_text, pe.query_norm, pe.query_hash, pe.sources,
                   pe.quality_score, pe.answer_text, pe.status, pe.hit_count,
                   pe.updated_at,
                   CASE WHEN pe.hit_count > 0
                        THEN COALESCE(cc.clicks, 0)::float / pe.hit_count
                        ELSE 0 END AS follow_through
              FROM pool_entry pe
              LEFT JOIN (SELECT entry_id, count(*) AS clicks
                           FROM citation_click GROUP BY entry_id) cc
                     ON cc.entry_id = pe.id
             WHERE pe.opt_in AND pe.status IN ('active','stale')
        )
        SELECT id, query_text, query_norm, query_hash, sources, quality_score,
               follow_through,
               CASE WHEN answer_text IS NULL THEN 'undistilled'
                    WHEN status = 'stale'    THEN 'stale'
                    WHEN quality_score IS NOT NULL AND quality_score < $2
                         THEN 'low_quality'
                    ELSE 'thin_answer' END AS reason
          FROM e
         WHERE answer_text IS NULL
            OR status = 'stale'
            OR (quality_score IS NOT NULL AND quality_score < $2)
            OR (answer_text IS NOT NULL AND hit_count >= $4
                AND follow_through >= $3)
         ORDER BY (answer_text IS NULL) DESC,
                  (status = 'stale') DESC,
                  follow_through DESC,
                  hit_count DESC,
                  updated_at ASC
         LIMIT $1
        """,
        limit,
        min_quality,
        ft_threshold,
        min_hits_for_follow_through,
    )
    out = []
    for r in rows:
        sources = r["sources"]
        if isinstance(sources, str):
            sources = json.loads(sources)
        out.append({
            "id": r["id"],
            "query_text": r["query_text"],
            "query_norm": r["query_norm"],
            "query_hash": r["query_hash"],
            "sources": sources or [],
            "quality_score": r["quality_score"],
            "follow_through": r["follow_through"],
            "reason": r["reason"],
        })
    return out


async def update_answer(
    pool: asyncpg.Pool,
    entry_id,
    *,
    answer_text: str,
    model: str | None,
    quality_score: float | None,
    provenance: dict | None = None,
    status: str = "active",
    output_trainable: bool = False,
    attribution_required: bool = False,
    no_grounded_cache: bool = False,
) -> None:
    """Upgrade an entry in place with a (re-)distilled answer. Leaves sources and
    embedding untouched."""
    import json

    await pool.execute(
        """
        UPDATE pool_entry
           SET answer_text = $2, model = $3, quality_score = $4,
               provenance = $5::jsonb, status = $6,
               output_trainable = $7, attribution_required = $8, no_grounded_cache = $9,
               updated_at = now(), content_at = now()
         WHERE id = $1
        """,
        entry_id,
        answer_text,
        model,
        quality_score,
        json.dumps(provenance or {}),
        status,
        output_trainable,
        attribution_required,
        no_grounded_cache,
    )


async def bump_reject(pool: asyncpg.Pool, entry_id) -> None:
    """Record an explicit reject (reject_entry retry) against an entry. hit_count's
    counterpart: a high reject/hit ratio marks a mis-serving entry for demotion."""
    await pool.execute(
        "UPDATE pool_entry SET reject_count = reject_count + 1 WHERE id = $1",
        entry_id,
    )


async def _bump_hit(pool: asyncpg.Pool, entry_id) -> None:
    await pool.execute(
        "UPDATE pool_entry SET hit_count = hit_count + 1, updated_at = now() WHERE id = $1",
        entry_id,
    )


def _to_hit(row, *, distance: float, match: str) -> Hit:
    import json

    sources = row["sources"]
    if isinstance(sources, str):
        sources = json.loads(sources)
    return Hit(
        id=str(row["id"]),
        query_text=row["query_text"],
        answer_text=row["answer_text"],
        sources=sources or [],
        model=row["model"],
        distance=distance,
        match=match,
        content_at=row["content_at"],
    )
