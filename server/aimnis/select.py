"""Feature-based source selection — decide WHICH sources (and in what order) to
ground on and serve, from a wider fetched candidate set.

Transparent and cold-start-safe (no training required): a weighted sum of a
provider-rank prior, lexical query↔source overlap, freshness, and — when it
exists — click follow-through debiased for the position the source was shown at.

This is deliberately the interface a learned ranker drops into later: same inputs
(query + candidate sources + accumulated click stats), same output (an ordering).
When the click log is large enough to train on, swap the scoring core in
`_score`; everything around it stays.

Design invariant it serves: selection ORDERS the stored source array (at ingest,
and later at refresh); serving is always a prefix of that order. So the served
top-K, the distilled answer's [n] citations, and the stored array all stay
aligned, and nothing re-orders under a live reply.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from math import log

from .config import settings

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str | None) -> set[str]:
    return set(_WORD.findall((text or "").lower()))


def _overlap(query_tokens: set[str], source: dict) -> float:
    """Fraction of query terms that appear in the source's title+snippet."""
    st = _tokens(f"{source.get('title', '')} {source.get('snippet', '')}")
    if not query_tokens or not st:
        return 0.0
    return len(query_tokens & st) / len(query_tokens)


def _freshness(source: dict, now: datetime) -> float:
    """1.0 for a just-fetched source, decaying with age (per-source fetched_at)."""
    ts = source.get("fetched_at")
    if not ts:
        return 0.0
    try:
        dt = datetime.fromisoformat(ts)
    except (ValueError, TypeError):
        return 0.0
    age_days = max((now - dt).total_seconds() / 86400.0, 0.0)
    return 1.0 / (1.0 + age_days)


def _click_score(source: dict, click_stats: dict | None, shown_rank: int) -> float:
    """Reward click follow-through, with extra credit for a source that earned
    clicks despite being shown in a WORSE (higher-index) position — a position-bias
    -resistant signal that it's genuinely preferred, not just first. `click_stats`
    maps url → click count (or {'clicks': n})."""
    if not click_stats:
        return 0.0
    cs = click_stats.get(source.get("url"))
    if not cs:
        return 0.0
    clicks = cs["clicks"] if isinstance(cs, dict) else cs
    return log(1.0 + clicks) * (1.0 + 0.1 * shown_rank)


def _score(query_tokens: set[str], source: dict, rank: int,
           click_stats: dict | None, now: datetime) -> float:
    return (
        settings.select_w_rank * (1.0 / (1.0 + rank))          # provider-order prior
        + settings.select_w_overlap * _overlap(query_tokens, source)
        + settings.select_w_freshness * _freshness(source, now)
        + settings.select_w_clicks * _click_score(source, click_stats, rank)
    )


def rank_sources(
    query: str,
    sources: list[dict],
    *,
    click_stats: dict | None = None,
    now: datetime | None = None,
) -> list[int]:
    """Return source indices ordered best-first. Stable: equal scores keep their
    original (provider) order, so with no signal this degrades to provider order."""
    if not sources:
        return []
    now = now or datetime.now(timezone.utc)
    qt = _tokens(query)
    scored = [
        (_score(qt, s, i, click_stats, now), i)
        for i, s in enumerate(sources)
    ]
    scored.sort(key=lambda t: (-t[0], t[1]))  # score desc, original order on ties
    return [i for _, i in scored]


def order_sources(
    query: str,
    sources: list[dict],
    *,
    click_stats: dict | None = None,
    now: datetime | None = None,
) -> list[dict]:
    """Convenience: the sources themselves in selected order."""
    return [sources[i] for i in rank_sources(query, sources, click_stats=click_stats, now=now)]
