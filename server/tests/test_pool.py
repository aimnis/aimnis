"""Knowledge-pool / semantic-cache lookup tests.

Embeddings are injected directly (no model download) so retrieval logic is
deterministic. Cosine distance = 1 - cos(angle) between the query and stored
vectors; the default hit threshold is cache_max_distance = 0.15.
"""

from __future__ import annotations

import pytest

from aimnis import pool
from aimnis.normalize import normalize, query_hash

DIM = 384


def vec(*pairs) -> list[float]:
    v = [0.0] * DIM
    for i, x in pairs:
        v[i] = x
    return v


V_A = vec((0, 1.0))                 # reference direction
V_NEAR = vec((0, 0.9), (1, 0.436))  # cos ≈ 0.9  → distance ≈ 0.10  (hit)
V_FAR = vec((0, 0.8), (1, 0.6))     # cos = 0.8   → distance = 0.20  (miss)
V_ORTH = vec((1, 1.0))              # orthogonal  → distance = 1.0   (miss)


async def _seed(db, text="how to fix ImportError in python", *, embedding=V_A, **kw):
    norm = normalize(text)
    return await pool.insert(
        db,
        query_text=text,
        query_norm=norm,
        query_hash=query_hash(norm),
        embedding=embedding,
        answer_text=kw.pop("answer_text", "an answer"),
        **kw,
    )


async def test_exact_hash_hit_beats_vector(clean):
    text = "reset a git branch to origin"
    await _seed(clean, text, embedding=V_A, answer_text="git reset --hard")
    h = query_hash(normalize(text))

    # Even with an orthogonal embedding, the exact hash match wins.
    hit = await pool.lookup(clean, query_hash=h, embedding=V_ORTH)
    assert hit is not None
    assert hit.match == "exact"
    assert hit.distance == 0.0
    assert hit.answer_text == "git reset --hard"
    assert hit.content_at is not None  # freshness timestamp travels with the hit


async def test_semantic_hit_within_threshold(clean):
    await _seed(clean, "how to fix ImportError", embedding=V_A, answer_text="check sys.path")
    hit = await pool.lookup(clean, query_hash="no-exact-match", embedding=V_NEAR)
    assert hit is not None
    assert hit.match == "semantic"
    assert hit.distance == pytest.approx(0.10, abs=0.01)
    assert hit.answer_text == "check sys.path"


async def test_semantic_miss_outside_threshold(clean):
    await _seed(clean, "how to fix ImportError", embedding=V_A)
    assert await pool.lookup(clean, query_hash="nope", embedding=V_FAR) is None
    assert await pool.lookup(clean, query_hash="nope", embedding=V_ORTH) is None


async def test_no_embedding_only_exact(clean):
    await _seed(clean, "some query", embedding=V_A)
    assert await pool.lookup(clean, query_hash="nope", embedding=None) is None


async def test_pending_entry_not_served(clean):
    text = "pending entry"
    await _seed(clean, text, embedding=V_A, status="pending")
    h = query_hash(normalize(text))
    assert await pool.lookup(clean, query_hash=h, embedding=V_A) is None


async def test_expired_entry_not_served(clean):
    text = "stale entry"
    entry_id = await _seed(clean, text, embedding=V_A, ttl_seconds=3600)
    await clean.execute(
        "UPDATE pool_entry SET expires_at = now() - interval '1 hour' WHERE id = $1", entry_id
    )
    h = query_hash(normalize(text))
    assert await pool.lookup(clean, query_hash=h, embedding=V_A) is None


async def test_hit_count_bumps(clean):
    text = "bump me"
    entry_id = await _seed(clean, text, embedding=V_A)
    h = query_hash(normalize(text))
    await pool.lookup(clean, query_hash=h, embedding=V_A)
    await pool.lookup(clean, query_hash=h, embedding=V_A)
    count = await clean.fetchval("SELECT hit_count FROM pool_entry WHERE id = $1", entry_id)
    assert count == 2


async def test_lookup_exact_only(clean):
    text = "exact only match"
    await _seed(clean, text, embedding=V_ORTH, answer_text="yep")
    h = query_hash(normalize(text))
    hit = await pool.lookup_exact(clean, query_hash=h)
    assert hit is not None
    assert hit.match == "exact"
    assert hit.answer_text == "yep"
    assert await pool.lookup_exact(clean, query_hash="no-such-hash") is None


async def test_candidates_orders_nearest_first_and_filters(clean):
    await _seed(clean, "at", embedding=V_A, answer_text="ans-a")       # distance 0.00
    await _seed(clean, "near", embedding=V_NEAR, answer_text="ans-n")  # distance 0.10
    await _seed(clean, "far", embedding=V_FAR, answer_text="ans-f")    # distance 0.20

    wide = await pool.candidates(clean, embedding=V_A, k=5, max_distance=0.30)
    assert [c.query_text for c in wide] == ["at", "near", "far"]
    assert all(c.match == "semantic" for c in wide)

    tight = await pool.candidates(clean, embedding=V_A, k=5, max_distance=0.15)
    assert [c.query_text for c in tight] == ["at", "near"]

    topk = await pool.candidates(clean, embedding=V_A, k=1, max_distance=0.30)
    assert [c.query_text for c in topk] == ["at"]


async def test_candidates_does_not_bump_hit_count(clean):
    entry_id = await _seed(clean, "no bump", embedding=V_A)
    await pool.candidates(clean, embedding=V_A, k=5, max_distance=0.30)
    count = await clean.fetchval("SELECT hit_count FROM pool_entry WHERE id = $1", entry_id)
    assert count == 0


async def test_mark_served_bumps(clean):
    entry_id = await _seed(clean, "serve me", embedding=V_A)
    await pool.mark_served(clean, entry_id)
    await pool.mark_served(clean, entry_id)
    count = await clean.fetchval("SELECT hit_count FROM pool_entry WHERE id = $1", entry_id)
    assert count == 2


async def test_sources_round_trip(clean):
    text = "with sources"
    srcs = [{"url": "https://example.com", "title": "Example"}]
    await _seed(clean, text, embedding=V_A, sources=srcs)
    hit = await pool.lookup(clean, query_hash=query_hash(normalize(text)), embedding=V_A)
    assert hit is not None
    assert hit.sources == srcs
