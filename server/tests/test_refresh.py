"""Background refresh-queue tests. Distillation is mocked (no network); the judge
is disabled so the suite spends no quota. Exercises the real DB, quota ledger,
quality gate, candidate selection, and in-place upgrade."""

from __future__ import annotations

import pytest

from aimnis import llm, pool as pool_mod, quota, refresh
from aimnis.config import settings
from aimnis.quota import Reservation

SOURCES = [{"title": "T", "url": "http://x", "snippet": "s"}]


@pytest.fixture(autouse=True)
def _env(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "distill_enabled", True)
    monkeypatch.setattr(settings, "refresh_judge_enabled", False)  # no network judge
    monkeypatch.setattr(settings, "refresh_delay_seconds", 0)      # no pacing in tests


async def _seed(db, *, query, sources=SOURCES, answer=None, model="searxng-live",
                quality_score=None, status="active"):
    return await pool_mod.insert(
        db, query_text=query, query_norm=query, query_hash=query, embedding=None,
        answer_text=answer, sources=sources, model=model, status=status,
        quality_score=quality_score,
    )


async def test_upgrades_snippet_only_entry(clean, monkeypatch):
    eid = await _seed(clean, query="q undistilled")

    async def good(query, results, **kw):
        assert results == SOURCES  # re-distilled from the STORED sources, no new search
        return llm.DistillResult(
            answer_text="Use httpx.AsyncClient(timeout=10.0); the default is 5s [1].",
            model="meta-llama/llama-3.3-70b-instruct:free",
            prompt_tokens=10, completion_tokens=5,
        )

    monkeypatch.setattr(llm, "distill", good)
    rep = await refresh.run_refresh(clean, limit=10)

    assert rep.upgraded == 1 and rep.stopped_reason is None
    assert rep.by_reason == {"undistilled": 1}
    row = await clean.fetchrow(
        "SELECT answer_text, model, quality_score, output_trainable, provenance "
        "FROM pool_entry WHERE id = $1", eid
    )
    assert "httpx" in row["answer_text"]
    assert "llama" in row["model"] and row["quality_score"] == 1.0
    assert row["output_trainable"] is True  # llama → trainable + attribution
    # Quota was actually spent for the upgrade.
    assert await clean.fetchval(
        "SELECT count(*) FROM upstream_call WHERE status='success'") == 1


async def test_bad_answer_rejected_leaves_snippet_only(clean, monkeypatch):
    eid = await _seed(clean, query="q bad")

    async def cot(query, results, **kw):
        return llm.DistillResult(
            answer_text="We need to answer: let's examine each result carefully first.",
            model="x", prompt_tokens=1, completion_tokens=1,
        )

    monkeypatch.setattr(llm, "distill", cot)
    rep = await refresh.run_refresh(clean, limit=10)

    assert rep.upgraded == 0 and rep.rejected == 1
    assert await clean.fetchval("SELECT answer_text FROM pool_entry WHERE id=$1", eid) is None


async def test_never_downgrades_existing_answer(clean, monkeypatch):
    # Threshold raised so a decent 0.9 entry qualifies as a (low-quality) candidate.
    monkeypatch.setattr(settings, "refresh_min_quality_score", 0.95)
    eid = await _seed(clean, query="q lowq",
                      answer="Old, well-grounded answer with a citation [1].",
                      model="meta-llama/llama-3.3-70b-instruct:free", quality_score=0.9)

    async def worse(query, results, **kw):  # no citation → scores 0.7 < 0.9
        return llm.DistillResult(
            answer_text="A replacement answer with no citation but plenty of length here.",
            model="x", prompt_tokens=1, completion_tokens=1,
        )

    monkeypatch.setattr(llm, "distill", worse)
    rep = await refresh.run_refresh(clean, limit=10)

    assert rep.skipped_no_improve == 1 and rep.upgraded == 0
    kept = await clean.fetchval("SELECT answer_text FROM pool_entry WHERE id=$1", eid)
    assert kept.startswith("Old, well-grounded")


async def test_quota_exhaustion_stops_batch(clean, monkeypatch):
    await _seed(clean, query="q1")
    await _seed(clean, query="q2")

    async def denied(*a, **k):
        return Reservation(granted=False, reason="purpose_budget", call_id=None)

    async def never(*a, **k):
        raise AssertionError("distill must not run once quota is denied")

    monkeypatch.setattr(quota, "reserve", denied)
    monkeypatch.setattr(llm, "distill", never)
    rep = await refresh.run_refresh(clean, limit=10)

    assert rep.stopped_reason == "quota" and rep.upgraded == 0


async def test_candidate_ordering_undistilled_first(clean, monkeypatch):
    monkeypatch.setattr(settings, "refresh_min_quality_score", 0.95)
    await _seed(clean, query="has answer",
                answer="Answer [1], long enough to store.", model="m", quality_score=0.9)
    await _seed(clean, query="no answer")

    cands = await pool_mod.select_refresh_candidates(clean, limit=10, min_quality=0.95)
    assert [c["reason"] for c in cands][0] == "undistilled"
    assert {c["reason"] for c in cands} == {"undistilled", "low_quality"}


async def test_thin_answer_selected_by_follow_through(clean):
    # A good-quality, distilled entry that would NOT otherwise be a refresh
    # candidate — but its answer keeps sending the agent to sources (high clicks
    # per hit), so it's flagged for re-distillation as 'thin_answer'.
    eid = await _seed(clean, query="thin answer", answer="short [1]", model="m",
                      quality_score=0.9)
    await clean.execute("UPDATE pool_entry SET hit_count = 6 WHERE id = $1", eid)
    await clean.executemany(
        "INSERT INTO citation_click (entry_id, source_idx, host) VALUES ($1,0,'x.io')",
        [(eid,)] * 6,  # 6 clicks / 6 hits → follow_through 1.0
    )

    cands = await pool_mod.select_refresh_candidates(
        clean, limit=10, min_quality=0.5,
        min_follow_through=1.0, min_hits_for_follow_through=5,
    )
    assert [str(c["id"]) for c in cands] == [eid]
    assert cands[0]["reason"] == "thin_answer"
    assert cands[0]["follow_through"] >= 1.0

    # Below the hit-count floor → ratio is noise, not trusted → not selected.
    below_floor = await pool_mod.select_refresh_candidates(
        clean, limit=10, min_quality=0.5,
        min_follow_through=1.0, min_hits_for_follow_through=50,
    )
    assert below_floor == []

    # Follow-through path disabled (None) → not selected on clicks at all.
    disabled = await pool_mod.select_refresh_candidates(
        clean, limit=10, min_quality=0.5, min_follow_through=None,
    )
    assert disabled == []


async def test_disabled_when_no_key(clean, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", None)
    await _seed(clean, query="q")
    rep = await refresh.run_refresh(clean, limit=10)
    assert rep.stopped_reason == "distill_disabled" and rep.candidates == 0
