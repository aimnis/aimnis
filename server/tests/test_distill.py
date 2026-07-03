"""Distillation integration through resolve_search.

Network (SearXNG, OpenRouter) and the embedding model are mocked so these run
deterministically and offline, but they exercise the REAL quota ledger, pool
write, compliance flags, and lookup_event log against the test DB.
"""

from __future__ import annotations

import pytest

from aimnis import llm, models, quota, resolve
from aimnis.config import settings
from aimnis.quota import Reservation
from aimnis.search import SearchError, SearchResult

_RESULTS = [
    SearchResult(title="pgvector README", url="http://a", snippet="register_vector(conn)"),
    SearchResult(title="asyncpg docs", url="http://b", snippet="create_pool(init=...)"),
]


@pytest.fixture(autouse=True)
def _mock_env(monkeypatch):
    """Enable distillation with a fake key; stub embedding + live search."""
    monkeypatch.setattr(settings, "openrouter_api_key", "test-key")
    monkeypatch.setattr(settings, "distill_enabled", True)
    monkeypatch.setattr(resolve, "_embed", lambda q: [0.01] * settings.embedding_dim)

    async def fake_live_search(query, *, limit=None):
        return list(_RESULTS)

    monkeypatch.setattr(resolve, "live_search", fake_live_search)


async def test_miss_distills_and_pools_answer(clean, monkeypatch):
    async def fake_distill(query, results, **kw):
        assert len(results) == 2  # got the live snippets
        return llm.DistillResult(
            answer_text="Call register_vector on each connection [1].",
            model="meta-llama/llama-3.3-70b-instruct:free",
            prompt_tokens=100, completion_tokens=20,
        )

    monkeypatch.setattr(llm, "distill", fake_distill)

    res = await resolve.resolve_search(clean, "how to register pgvector on asyncpg")
    assert res["source"] == "live"
    assert res["distilled"] is True
    assert "register_vector" in res["answer"]

    # Pooled with the distilled answer + Llama compliance flags.
    row = await clean.fetchrow(
        "SELECT answer_text, model, output_trainable, attribution_required "
        "FROM pool_entry WHERE id = $1", res["entry_id"]
    )
    assert row["answer_text"] == res["answer"]
    assert "llama" in row["model"]
    assert row["output_trainable"] is True and row["attribution_required"] is True

    # Quota ledger recorded one successful call with token counts.
    call = await clean.fetchrow(
        "SELECT status, purpose, prompt_tokens, completion_tokens FROM upstream_call"
    )
    assert call["status"] == "success"
    assert call["purpose"] == settings.distill_purpose
    assert call["prompt_tokens"] == 100 and call["completion_tokens"] == 20

    # Served text carries the attribution and a citable Sources block.
    out = resolve.format_for_agent(res)
    assert "distilled answer" in out and "Built with Llama" in out and "Sources:" in out


async def test_quota_denied_falls_back_to_snippets(clean, monkeypatch):
    async def denied(*a, **k):
        return Reservation(granted=False, reason="purpose_budget", call_id=None)

    async def boom(*a, **k):  # distill must NOT be called when quota is denied
        raise AssertionError("distill called despite denied quota")

    monkeypatch.setattr(quota, "reserve", denied)
    monkeypatch.setattr(llm, "distill", boom)

    res = await resolve.resolve_search(clean, "denied path query")
    assert res["distilled"] is False
    assert res["answer"] is None
    row = await clean.fetchrow(
        "SELECT answer_text, model FROM pool_entry WHERE id = $1", res["entry_id"]
    )
    assert row["answer_text"] is None
    assert row["model"] == "searxng-live"


async def test_distill_timeout_records_timeout_and_degrades(clean, monkeypatch):
    async def timeout(*a, **k):
        raise llm.LLMTimeout("too slow")

    monkeypatch.setattr(llm, "distill", timeout)

    res = await resolve.resolve_search(clean, "slow distill query")
    assert res["distilled"] is False and res["answer"] is None
    call = await clean.fetchrow("SELECT status FROM upstream_call")
    assert call["status"] == "timeout"  # burned quota, correctly recorded


async def test_no_key_skips_distill_entirely(clean, monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", None)

    async def boom(*a, **k):
        raise AssertionError("distill called without a key")

    monkeypatch.setattr(llm, "distill", boom)

    res = await resolve.resolve_search(clean, "unkeyed query")
    assert res["distilled"] is False
    # No quota spent when distillation is unkeyed.
    n = await clean.fetchval("SELECT count(*) FROM upstream_call")
    assert n == 0


async def test_empty_search_is_not_pooled(clean, monkeypatch):
    """A transient empty result set must not be cached (would serve a permanent
    'no results' on every future exact match)."""
    async def empty(query, *, limit=None):
        return []

    monkeypatch.setattr(resolve, "live_search", empty)
    res = await resolve.resolve_search(clean, "query that yields nothing right now")
    assert res["source"] == "live"
    assert res["results"] == [] and res["entry_id"] is None
    assert await clean.fetchval("SELECT count(*) FROM pool_entry") == 0
    # Recorded as a non-servable outcome, not a hit/miss that poisons the pool.
    assert await clean.fetchval(
        "SELECT count(*) FROM lookup_event WHERE outcome = 'error'") == 1


async def test_bad_distillation_rejected_not_pooled(clean, monkeypatch):
    """A distillation that succeeds upstream but is low-quality (CoT leak) must be
    rejected by the quality gate: not pooled as an answer, snippets served, and
    the quota it spent still recorded."""
    async def cot_distill(query, results, **kw):
        return llm.DistillResult(
            answer_text="We need to answer: the user is asking about pooling. "
                        "Let's examine each result and figure out the best approach.",
            model="meta-llama/llama-3.3-70b-instruct:free",
            prompt_tokens=50, completion_tokens=30,
        )

    monkeypatch.setattr(llm, "distill", cot_distill)

    res = await resolve.resolve_search(clean, "how does connection pooling work")
    assert res["distilled"] is False
    assert res["distill_rejected"] is True
    assert res["answer"] is None
    assert "reasoning_leak" in res["quality_flags"]

    row = await clean.fetchrow(
        "SELECT answer_text, model, quality_score FROM pool_entry WHERE id = $1",
        res["entry_id"],
    )
    assert row["answer_text"] is None
    assert row["model"] == "searxng-live"     # degraded to snippet-only
    assert row["quality_score"] is None
    # The distill call still consumed quota (we paid; we just distrust the output).
    call = await clean.fetchrow("SELECT status FROM upstream_call")
    assert call["status"] == "success"


async def test_good_distillation_stores_quality_score(clean, monkeypatch):
    async def good_distill(query, results, **kw):
        return llm.DistillResult(
            answer_text="Register the vector type on each pooled connection via the "
                        "init hook: asyncpg.create_pool(init=register_vector) [1].",
            model="meta-llama/llama-3.3-70b-instruct:free",
            prompt_tokens=80, completion_tokens=25,
        )

    monkeypatch.setattr(llm, "distill", good_distill)
    res = await resolve.resolve_search(clean, "register pgvector on asyncpg pool")
    assert res["distilled"] is True
    assert res["quality_score"] == 1.0
    row = await clean.fetchval(
        "SELECT quality_score FROM pool_entry WHERE id = $1", res["entry_id"]
    )
    assert abs(row - 1.0) < 1e-9


async def test_secret_scrubbed_before_distill_and_pool(clean, monkeypatch):
    async def good(query, results, **kw):
        assert "AKIA" not in query  # the redacted query reaches distillation, not the secret
        return llm.DistillResult(
            answer_text="Rotate the leaked key and load it from an env var [1].",
            model="meta-llama/llama-3.3-70b-instruct:free",
            prompt_tokens=1, completion_tokens=1,
        )

    monkeypatch.setattr(llm, "distill", good)
    res = await resolve.resolve_search(clean, "why does AKIAIOSFODNN7EXAMPLE fail in boto3")

    assert res["pooled"] is True and res["scrubbed"] == {"AWS_KEY": 1}
    stored = await clean.fetchval("SELECT query_text FROM pool_entry WHERE id=$1", res["entry_id"])
    assert "AKIA" not in stored and "⟨AWS_KEY⟩" in stored


async def test_secret_dense_query_not_pooled(clean, monkeypatch):
    async def good(query, results, **kw):
        return llm.DistillResult(answer_text="Rotate all leaked credentials now [1].",
                                 model="m", prompt_tokens=1, completion_tokens=1)

    monkeypatch.setattr(llm, "distill", good)
    q = ("AKIAIOSFODNN7EXAMPLE AKIAIOSFODNN7EXAMPL2 ghp_" + "a" * 36
         + " xoxb-1234567890 all broken")
    res = await resolve.resolve_search(clean, q)

    assert res["pooled"] is False and res["entry_id"] is None
    assert await clean.fetchval("SELECT count(*) FROM pool_entry") == 0


def test_compliance_registry():
    assert models.compliance_for("meta-llama/llama-3.3-70b-instruct:free")[
        "attribution_required"] is True
    assert models.compliance_for("google/gemini-2.0-flash:free")[
        "output_trainable"] is False
    assert models.compliance_for("deepseek/deepseek-chat:free")[
        "output_trainable"] is True
    assert models.compliance_for(None) == {
        "output_trainable": False, "attribution_required": False, "no_grounded_cache": False}
    assert models.attribution_for("meta-llama/llama-3.3-70b:free") == "Built with Llama"
    assert models.attribution_for("deepseek/deepseek-chat") is None
