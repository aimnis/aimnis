"""Quality-gate tests. Heuristics are pure (no I/O); the judge is exercised
offline via httpx.MockTransport so it spends no quota and hits no network."""

from __future__ import annotations

import httpx

from aimnis import quality
from aimnis.config import settings

SOURCES = [{"title": "A", "url": "u1", "snippet": "s1"},
           {"title": "B", "url": "u2", "snippet": "s2"}]


# --- heuristics -------------------------------------------------------------- #
def test_accepts_good_grounded_answer():
    a = ("Pass timeout to httpx.AsyncClient(timeout=10.0); the default is 5s. "
         "Use httpx.Timeout for granular control [1][2].")
    r = quality.score_answer("q", a, SOURCES)
    assert r.accept and r.score == 1.0


def test_rejects_too_short():
    r = quality.score_answer("q", "yes", SOURCES)
    assert not r.accept and "too_short" in r.reasons


def test_rejects_empty_or_none():
    assert not quality.score_answer("q", None, SOURCES).accept
    assert not quality.score_answer("q", "   ", SOURCES).accept


def test_rejects_refusal_or_insufficient():
    a = "The provided search results do not contain information about this option at all."
    r = quality.score_answer("q", a, SOURCES)
    assert not r.accept and "refusal_or_insufficient" in r.reasons


def test_rejects_reasoning_leak():
    a = ("We need to answer: the user is asking about timeouts. Let's examine each "
         "result and decide what matters.")
    r = quality.score_answer("q", a, SOURCES)
    assert not r.accept and "reasoning_leak" in r.reasons


def test_rejects_prompt_echo():
    a = "Search results: [1] foo. Using only the numbered results provided here today."
    r = quality.score_answer("q", a, SOURCES)
    assert not r.accept and "prompt_echo" in r.reasons


def test_rejects_repetitive_lines():
    a = "\n".join(["The timeout is 5 seconds by default."] * 6)
    r = quality.score_answer("q", a, SOURCES)
    assert not r.accept and "repetitive_lines" in r.reasons


def test_no_citation_is_soft_penalty_but_accepts():
    a = ("Set the timeout parameter on the client constructor to change the default "
         "of five seconds applied across all operations.")
    r = quality.score_answer("q", a, SOURCES)
    assert r.accept and "no_citations" in r.reasons
    assert abs(r.score - 0.7) < 1e-9


def test_citation_out_of_range_penalised():
    a = "Use the timeout parameter to override the default value as the docs describe [9]."
    r = quality.score_answer("q", a, SOURCES)  # only 2 sources, cites [9]
    assert "citation_out_of_range" in r.reasons


# --- judge (mocked) ---------------------------------------------------------- #
def _judge_transport(content: str) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={
            "model": "judge-model",
            "choices": [{"message": {"content": content}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5},
        })
    return httpx.MockTransport(handler)


async def test_judge_accepts_high_score(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "k")
    jr = await quality.judge("q", "answer", SOURCES,
                             transport=_judge_transport('{"score": 4, "reason": "grounded"}'))
    assert jr.score == 4 and jr.accept and "grounded" in jr.reason


async def test_judge_rejects_low_score(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "k")
    jr = await quality.judge("q", "answer", SOURCES,
                             transport=_judge_transport('Sure: {"score": 1, "reason": "hallucinated"}'))
    assert jr.score == 1 and not jr.accept


async def test_judge_parses_fenced_json(monkeypatch):
    monkeypatch.setattr(settings, "openrouter_api_key", "k")
    jr = await quality.judge("q", "answer", SOURCES,
                             transport=_judge_transport('```json\n{"score": 5, "reason": "great"}\n```'))
    assert jr.score == 5 and jr.accept
