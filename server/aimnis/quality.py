"""Quality scoring — the gate that keeps the pool (and the training feed the
mission depends on) trustworthy. A distilled answer that fails is NOT pooled; the
caller degrades to serving raw snippets, so a bad answer never becomes a
permanent cached artifact or training data.

Two layers:
  1. `score_answer` — heuristics, always on, no quota. Hard-rejects degenerate /
     refusal / reasoning-leak / prompt-echo / repetitive / too-short answers, and
     computes a 0..1 soft `quality_score` from grounding signals (citations,
     length). This catches exactly the failure modes we saw live (CoT dumped into
     the answer, "the results do not contain…", empty content).
  2. `judge` — LLM-as-judge (spends quota via the caller's ledger reservation).
     Opt-in (`quality_judge_enabled`), intended for the background path where a
     second model call is affordable.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Sequence

from .config import settings

# "The (search) results do not contain…", "I don't have enough…", "unable to…".
_REFUSAL_RE = re.compile(
    r"(do not|don't|does not|doesn't|cannot|can't|could not|couldn't) "
    r"(contain|include|provide|have|mention|address|cover|find|answer|determine)"
    r"|no (relevant )?(information|answer|results|details|mention|data)"
    r"|(am|are) unable to"
    r"|(insufficient|not enough) (information|context|detail|data)",
    re.I,
)
# Reasoning / meta-commentary that leaked into the answer.
_COT_PHRASES = (
    "we need to answer", "let's examine", "let me analyze", "the user is asking",
    "the user wants", "let me think", "let's break this down", "let's see what",
    "first, let", "here's my reasoning", "step 1:",
)
_COT_LEAD_RE = re.compile(r"^(okay|ok|so|hmm|alright|well|now),?\s", re.I)
# Bits of our own prompt echoed back.
_PROMPT_ECHO_RE = re.compile(
    r"(search results:|^query:|using only the numbered|numbered web search results"
    r"|as an ai language model)",
    re.I,
)
_CITATION_RE = re.compile(r"\[(\d+)\]")


class QualityError(RuntimeError):
    pass


@dataclass(frozen=True)
class QualityResult:
    accept: bool             # may this distilled answer be pooled/served as an answer?
    score: float             # 0..1 soft quality signal (stored as pool_entry.quality_score)
    reasons: tuple[str, ...]  # flags (hard-fail reason, or soft penalties applied)


@dataclass(frozen=True)
class JudgeResult:
    score: int               # 1..5
    accept: bool
    reason: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


def score_answer(query: str, answer: str | None, sources: Sequence) -> QualityResult:
    text = (answer or "").strip()
    n_sources = len(sources or [])

    if len(text) < settings.quality_min_answer_chars:
        return QualityResult(False, 0.0, ("too_short",))

    low = text.lower()
    if _REFUSAL_RE.search(low):
        # Correct behaviour per the distill prompt, but not a poolable artifact.
        return QualityResult(False, 0.0, ("refusal_or_insufficient",))
    if _COT_LEAD_RE.match(low) or any(p in low for p in _COT_PHRASES):
        return QualityResult(False, 0.0, ("reasoning_leak",))
    if _PROMPT_ECHO_RE.search(low):
        return QualityResult(False, 0.0, ("prompt_echo",))

    # Degenerate repetition.
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if len(lines) >= 4 and len(set(lines)) / len(lines) < 0.6:
        return QualityResult(False, 0.0, ("repetitive_lines",))
    words = low.split()
    if len(words) >= 50 and len(set(words)) / len(words) < 0.35:
        return QualityResult(False, 0.0, ("low_lexical_diversity",))

    # Soft grounding signals → score.
    reasons: list[str] = []
    score = 1.0
    cites = [int(n) for n in _CITATION_RE.findall(text)]
    if not cites:
        score -= 0.3
        reasons.append("no_citations")
    elif n_sources and any(c < 1 or c > n_sources for c in cites):
        score -= 0.2
        reasons.append("citation_out_of_range")
    if len(text) > settings.quality_max_answer_chars:
        score -= 0.2
        reasons.append("very_long")

    score = max(0.0, min(1.0, score))
    accept = score >= settings.quality_min_score
    if not accept:
        reasons.append("below_min_score")
    return QualityResult(accept, score, tuple(reasons))


# --------------------------------------------------------------------------- #
# LLM-as-judge (opt-in; quota is reserved by the caller, e.g. resolve._judge)
# --------------------------------------------------------------------------- #
_JUDGE_SYSTEM = (
    "You are a strict quality judge for a knowledge cache that may feed model "
    "training. Given a QUERY, an ANSWER, and the SOURCES the answer was distilled "
    "from, rate the answer 1-5: is it grounded in the sources (no hallucination), "
    "does it directly and accurately answer the query, is it concise and free of "
    "reasoning/meta-commentary? 5 = excellent grounded answer; 1 = wrong, "
    "hallucinated, or degenerate. Respond with ONLY a JSON object: "
    '{"score": <1-5>, "reason": "<short>"}.'
)


def _build_judge_messages(query: str, answer: str, sources: Sequence) -> list[dict]:
    blocks = []
    for i, r in enumerate(sources, 1):
        title = (r.get("title") if isinstance(r, dict) else "") or ""
        snippet = (r.get("snippet") if isinstance(r, dict) else "") or ""
        blocks.append(f"[{i}] {title}\n{snippet}".strip())
    corpus = "\n\n".join(blocks) if blocks else "(no sources)"
    return [
        {"role": "system", "content": _JUDGE_SYSTEM},
        {"role": "user",
         "content": f"QUERY:\n{query}\n\nANSWER:\n{answer}\n\nSOURCES:\n{corpus}"},
    ]


def _extract_json(text: str) -> dict:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise QualityError(f"judge returned no JSON: {text[:120]}")
    try:
        return json.loads(m.group(0))
    except ValueError as exc:
        raise QualityError(f"judge JSON parse failed: {exc}") from exc


async def judge(query: str, answer: str, sources: Sequence, *, transport=None) -> JudgeResult:
    """Score an answer with an LLM judge. Raises LLMError/QualityError on failure
    (the caller records the ledger outcome). Does not touch the quota ledger itself."""
    from . import llm  # lazy

    out = await llm.complete(
        _build_judge_messages(query, answer, sources), max_tokens=200, transport=transport
    )
    data = _extract_json(out.content)
    try:
        score = int(data["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise QualityError(f"judge score missing/invalid: {data}") from exc
    score = max(1, min(5, score))
    return JudgeResult(
        score=score,
        accept=score >= settings.quality_judge_min_score,
        reason=str(data.get("reason", ""))[:300],
        model=out.model,
        prompt_tokens=out.prompt_tokens,
        completion_tokens=out.completion_tokens,
    )
