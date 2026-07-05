"""Resolution engine.

    normalize + hash + embed
      → cache lookup (exact hash, then vector NN)
          hit  → return the pooled entry (served as a prefix of its ranked sources)
          miss → live search (provider chain) → distill into a cited answer
                 → quality-gate → opt-in pool write → return

On a miss, results are distilled into a grounded, `[n]`-cited answer via OpenRouter
`:free` when a key is set (else raw cited snippets, zero upstream spend); the answer
is quality-gated before it may enter the pool, and low-quality entries are upgraded
later by the background refresh pass. Answers are AI-generated and labeled as such in
`format_for_agent`.
"""

from __future__ import annotations

import asyncio
import hashlib
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import asyncpg

from . import models
from . import pool as pool_mod
from . import quality
from . import quota
from . import scrub as scrub_mod
from . import select
from . import stats
from .config import settings
from .normalize import normalize, query_hash
from .search import SearchError, live_search


def _embed(query: str) -> list[float]:
    from .embedding import embed

    return embed(query)


async def _distill(db: asyncpg.Pool, query: str, h: str, results: list[dict], *,
                   purpose: str | None = None, client_keys=None):
    """Distill snippets into a grounded answer, spending quota via the ledger.

    Returns a DistillResult on success, or None when distillation is
    disabled/unkeyed/has no results, or the upstream call times out/errors. Raises
    `quota.QuotaExceeded` when the reservation is DENIED (so a background batch can
    stop; the interactive caller catches it and degrades to snippets). Every
    granted call is finalized with its real ledger status so a 429/timeout counts.

    BYOK: with a client-supplied OpenRouter key (client_keys.openrouter_api_key),
    the call runs on THEIR key and quota — no ledger reservation (the ledger meters
    the service's keys; OpenRouter enforces theirs) and, deliberately, NO fallback
    to the service key on failure: a BYOK client's (much higher) miss volume must
    never drain the shared distill budget. Their failure degrades to snippets. The
    model chain is still the service's vetted distill_models — per-model compliance
    flags stay meaningful regardless of whose key paid for the call.
    """
    if not (settings.distill_enabled and results):
        return None

    from . import llm  # lazy: keeps httpx-based LLM path out of the import graph until used

    byok_key = client_keys.openrouter_api_key if client_keys else None
    if byok_key:
        try:
            return await llm.distill(query, results, api_key=byok_key)
        except llm.LLMError:  # incl. rate-limited/timeout — degrade to snippets
            return None

    if not settings.openrouter_api_key:
        return None

    primary = settings.distill_models[0] if settings.distill_models else None
    res = await quota.reserve(
        db, purpose or settings.distill_purpose, model=primary, query_hash=h
    )
    if not res.granted:
        raise quota.QuotaExceeded(res.reason)

    assert res.call_id is not None
    try:
        out = await llm.distill(query, results)
    except llm.LLMRateLimited as exc:
        await quota.record_outcome(db, res.call_id, "rate_limited", http_status=429,
                                   error=str(exc)[:500])
        return None
    except llm.LLMTimeout as exc:
        await quota.record_outcome(db, res.call_id, "timeout", error=str(exc)[:500])
        return None
    except llm.LLMError as exc:
        await quota.record_outcome(db, res.call_id, "error",
                                   http_status=exc.http_status, error=str(exc)[:500])
        return None
    except Exception as exc:  # noqa: BLE001 — never leak an in_flight reservation
        await quota.record_outcome(db, res.call_id, "error", error=str(exc)[:500])
        return None

    await quota.record_outcome(
        db, res.call_id, "success", http_status=200,
        prompt_tokens=out.prompt_tokens, completion_tokens=out.completion_tokens,
    )
    return out


async def _judge(db: asyncpg.Pool, query: str, answer: str, results: list[dict], *,
                 purpose: str | None = None):
    """LLM-as-judge, quota-gated (opt-in via quality_judge_enabled). Mirrors
    _distill: reserve before, record the real outcome after, degrade to None on
    any failure (incl. quota denial) so a judge error never blocks pooling on the
    heuristic verdict."""
    from . import llm  # lazy

    res = await quota.reserve(db, purpose or settings.quality_judge_purpose,
                              model="judge", query_hash=None)
    if not res.granted:
        return None
    assert res.call_id is not None
    try:
        jr = await quality.judge(query, answer, results)
    except llm.LLMRateLimited as exc:
        await quota.record_outcome(db, res.call_id, "rate_limited", http_status=429,
                                   error=str(exc)[:500])
        return None
    except llm.LLMTimeout as exc:
        await quota.record_outcome(db, res.call_id, "timeout", error=str(exc)[:500])
        return None
    except (llm.LLMError, quality.QualityError) as exc:
        await quota.record_outcome(db, res.call_id, "error",
                                   http_status=getattr(exc, "http_status", None),
                                   error=str(exc)[:500])
        return None
    except Exception as exc:  # noqa: BLE001 — never leak an in_flight reservation
        await quota.record_outcome(db, res.call_id, "error", error=str(exc)[:500])
        return None

    await quota.record_outcome(
        db, res.call_id, "success", http_status=200,
        prompt_tokens=jr.prompt_tokens, completion_tokens=jr.completion_tokens,
    )
    return jr


@dataclass(frozen=True)
class DistillScored:
    answer: str | None
    model: str | None
    quality_score: float | None
    flags: tuple[str, ...]
    attempted: bool   # distillation returned a result (whether or not it passed)
    rejected: bool    # attempted but the quality gate rejected it


_NO_DISTILL = DistillScored(None, None, None, (), False, False)


async def _distill_and_score(
    db: asyncpg.Pool, query: str, h: str, results: list[dict], *,
    purpose: str | None = None, judge_enabled: bool = False,
    judge_purpose: str | None = None, client_keys=None,
) -> DistillScored:
    """Distill + quality-gate (+ optional judge). Shared by the interactive path
    and the background refresh queue. May raise quota.QuotaExceeded (from _distill)
    when the reservation is denied — callers decide whether to degrade or stop."""
    distilled = await _distill(db, query, h, results, purpose=purpose,
                               client_keys=client_keys)
    if distilled is None:
        return _NO_DISTILL

    qr = quality.score_answer(query, distilled.answer_text, results)
    accept, score, flags = qr.accept, qr.score, qr.reasons
    if accept and judge_enabled:
        jr = await _judge(db, query, distilled.answer_text, results, purpose=judge_purpose)
        if jr is not None:
            accept, score = jr.accept, jr.score / 5.0
            flags = flags + (f"judge:{jr.score}",)
    if accept:
        return DistillScored(distilled.answer_text, distilled.model, score, flags, True, False)
    return DistillScored(None, None, None, flags, True, True)


async def _cache_lookup(
    db: asyncpg.Pool, query: str, h: str, embedding: list[float],
    exclude: str | None = None,
) -> tuple[pool_mod.Hit | None, float | None, float | None]:
    """Resolve a query to a cached hit. Exact hash first; then, on the semantic
    path, retrieve the top-k nearest candidates and rerank their QUESTIONS with a
    cross-encoder — accept the best only if its score clears rerank_min_score.

    Returns (hit, rerank_score, best_distance). rerank_score/best_distance describe
    the best semantic candidate whether it was accepted OR rejected, so the reject
    case can be logged for tuning; both are None for an exact hit or when reranking
    is off/unavailable. If the reranker can't load, degrades to the plain
    nearest-within-cache_max_distance gate so a reranker hiccup never breaks search."""
    exact = await pool_mod.lookup_exact(db, query_hash=h, exclude=exclude)
    if exact is not None:
        return exact, None, None

    if not settings.rerank_enabled:
        return await pool_mod.lookup(
            db, query_hash=h, embedding=embedding, exclude=exclude
        ), None, None

    cands = await pool_mod.candidates(
        db, embedding=embedding, k=settings.rerank_candidates,
        max_distance=settings.rerank_recall_max_distance, exclude=exclude,
    )
    if not cands:
        return None, None, None

    from . import rerank  # lazy: keep the cross-encoder out of the import graph until used

    try:
        ranked = await asyncio.to_thread(rerank.rank, query, [c.query_text for c in cands])
    except Exception:  # noqa: BLE001 — reranker unavailable → fall back to the distance gate
        best = cands[0]
        if best.distance <= settings.cache_max_distance:
            await pool_mod.mark_served(db, best.id)
            return best, None, best.distance
        return None, None, best.distance

    best_idx, best_score = ranked[0]
    best = cands[best_idx]
    if best_score >= settings.rerank_min_score:
        await pool_mod.mark_served(db, best.id)
        return best, best_score, best.distance
    return None, best_score, best.distance


async def resolve_search(db: asyncpg.Pool, query: str, *, niche: str | None = None,
                         client_keys=None, client_id: str | None = None,
                         reject_entry: str | None = None, miss_gate=None) -> dict:
    # BYOK: `client_keys` (apikeys.ClientKeys) carries the calling client's own
    # upstream credentials. Their misses spend their quota (search + distill);
    # everything else — cache lookup, scrubbing, pooling — is identical. Used only
    # for THIS request, never for other users' queries (ToS invariant).
    #
    # `client_id` (a DB client uuid, or "admin" for env keys) is hashed into the
    # lookup log so hit-satisfaction can sequence ONE client's lookups; the raw id
    # never lands in lookup_event. `reject_entry` is the explicit-reject retry: the
    # named entry is excluded from this lookup, its reject_count bumps, and the
    # reject is logged as a dissatisfaction label on the prior hit.
    #
    # `miss_gate` (async () -> bool, or None) is the free-tier budget hook: called
    # once IFF the cache can't answer, right before any upstream money is spent.
    # False ⇒ return source="quota" instead of searching live. Cache hits never
    # consult it — hits cost ~nothing to serve, so they stay free and unmetered
    # (that asymmetry is the whole free-tier design; see mcp_http).
    #
    # Scrub FIRST: secrets/PII must never reach a third party (Brave/OpenRouter)
    # or the pool. Everything downstream uses the redacted text; a secret-dense
    # query is served live-only (safe_to_pool=False) and never persisted.
    scrubbed = scrub_mod.scrub(query)
    query = scrubbed.text

    client_hash = (
        hashlib.sha256(client_id.encode()).hexdigest()[:16] if client_id else None
    )
    if reject_entry is not None:
        try:  # agent-supplied — a malformed id is ignored, never an error
            reject_entry = str(uuid.UUID(reject_entry))
        except ValueError:
            reject_entry = None
        else:
            await pool_mod.bump_reject(db, reject_entry)

    norm = normalize(query)
    h = query_hash(norm)
    embedding = await asyncio.to_thread(_embed, query)

    hit, rerank_score, best_distance = await _cache_lookup(
        db, query, h, embedding, exclude=reject_entry)
    if hit is not None:
        # Serve the top-K prefix of the stored (selector-ordered) sources; the
        # extras stored beyond K are the candidate pool for re-selection, not shown.
        # A prefix keeps the served list aligned with the answer's [n] citations.
        served = hit.sources[: settings.search_result_limit]
        await stats.record_event(
            db,
            query_hash=h,
            outcome="hit_exact" if hit.match == "exact" else "hit_semantic",
            distance=hit.distance,
            entry_id=hit.id,
            niche=niche,
            rerank_score=rerank_score,
            result_count=len(served),
            client_hash=client_hash,
            embedding=embedding,
            rejected_entry=reject_entry,
        )
        return {
            "source": "cache",
            "match": hit.match,
            "distance": hit.distance,
            # Echo the question this answer was actually cached for, so the calling
            # agent can catch a subtle intent/polarity mismatch (enable vs disable)
            # the reranker can't fully separate and escalate to a live search.
            "matched_query": hit.query_text,
            "rerank_score": rerank_score,
            "answer": hit.answer_text,
            "results": served,
            "model": hit.model,
            "entry_id": hit.id,
            # Content-freshness of the cached answer, so the algorithm/agent can
            # weigh how old it is and decide whether to trust it or re-search.
            "cached_at": hit.content_at.isoformat() if hit.content_at else None,
            "age_seconds": (
                (datetime.now(timezone.utc) - hit.content_at).total_seconds()
                if hit.content_at else None
            ),
        }

    # The cache can't answer — from here on, upstream money gets spent. A gated
    # (keyless) caller out of daily miss budget stops HERE: no live search, no
    # distill, and no lookup_event (quota refusals aren't misses; logging them
    # would depress the hit-rate metric the flywheel is judged by).
    if miss_gate is not None and not await miss_gate():
        return {"source": "quota", "results": []}

    try:
        # Fetch WIDE (search_fetch_limit) so the selector has a candidate pool.
        results = [asdict(r) for r in await live_search(
            query, limit=settings.search_fetch_limit, client_keys=client_keys)]
    except SearchError as exc:
        await stats.record_event(db, query_hash=h, outcome="error", niche=niche,
                                 client_hash=client_hash, embedding=embedding,
                                 rejected_entry=reject_entry)
        return {"source": "error", "error": str(exc), "results": []}

    # Stamp each source with when we fetched it, so per-source freshness travels
    # with the source through storage/refresh (and into future source-selection).
    fetched_iso = datetime.now(timezone.utc).isoformat()
    for r in results:
        r["fetched_at"] = fetched_iso

    # A transient empty result set (e.g. all upstream engines rate-limited) must
    # NOT be pooled — it would be served as a permanent exact-hit "no results".
    if not results:
        await stats.record_event(db, query_hash=h, outcome="error", niche=niche,
                                 client_hash=client_hash, embedding=embedding,
                                 rejected_entry=reject_entry)
        return {"source": "live", "results": [], "answer": None, "distilled": False,
                "entry_id": None}

    # Dedupe by URL, then let the feature selector ORDER the candidates (cold start:
    # provider-rank prior + query/source overlap + freshness; no clicks yet on a
    # brand-new fetch). Store the full ordered set (wide); ground/serve the top-K
    # prefix. Ordering the stored array — not re-ordering at serve — keeps the
    # served list, the answer's citations, and storage all aligned.
    seen: set[str] = set()
    deduped = []
    for r in results:
        u = r.get("url")
        if u and u not in seen:
            seen.add(u)
            deduped.append(r)
    results = [deduped[i] for i in select.rank_sources(query, deduped)]
    served = results[: settings.search_result_limit]

    # Distill + quality-gate from the top-K. A failing answer degrades to snippets
    # so it never poisons the pool. Quota denial degrades to snippets rather than
    # erroring.
    try:
        scored = await _distill_and_score(
            db, query, h, served,
            purpose=settings.distill_purpose,
            judge_enabled=settings.quality_judge_enabled,
            judge_purpose=settings.quality_judge_purpose,
            client_keys=client_keys,
        )
    except quota.QuotaExceeded:
        scored = _NO_DISTILL

    answer = scored.answer
    answer_model = scored.model
    quality_score = scored.quality_score
    quality_flags = scored.flags

    model = answer_model if answer else "searxng-live"
    flags = models.compliance_for(model) if answer else {}

    # ToS-taint tracking: entries produced under BYOK credentials are provenance-
    # tagged so a later per-provider ToS finding can filter or purge them — the
    # corpus never carries anonymous license taint. byok_search is conservatively
    # set whenever the client HAD a search key (the service chain may have served
    # as fallback; over-tagging errs on the purgeable side).
    byok_prov = {}
    if client_keys is not None:
        if client_keys.search_provider and client_keys.search_api_key:
            byok_prov["byok_search"] = client_keys.search_provider
        if client_keys.openrouter_api_key and answer:
            byok_prov["byok_distill"] = True

    # Secret-dense queries are served live-only and never persisted (redaction may
    # be incomplete, and such a query is unlikely to be reusable knowledge).
    entry_id = None
    if scrubbed.safe_to_pool:
        entry_id = await pool_mod.insert(
            db,
            query_text=query,
            query_norm=norm,
            query_hash=h,
            embedding=embedding,
            answer_text=answer,
            sources=results,
            model=model,
            provenance={"source": "searxng-live", "distilled": bool(answer),
                        "quality_flags": list(quality_flags), **byok_prov},
            status="active",
            niche=niche,
            quality_score=quality_score,
            output_trainable=flags.get("output_trainable", False),
            attribution_required=flags.get("attribution_required", False),
            no_grounded_cache=flags.get("no_grounded_cache", False),
        )
    # If a semantic candidate was found but REJECTED by the reranker, log its
    # score/distance on the miss so rerank_min_score can be tuned against the
    # rejected-near-miss distribution (both are None for a true no-candidate miss).
    await stats.record_event(
        db, query_hash=h, outcome="miss", entry_id=entry_id, niche=niche,
        distance=best_distance, rerank_score=rerank_score,
        result_count=len(served),
        client_hash=client_hash, embedding=embedding, rejected_entry=reject_entry,
    )
    return {
        "source": "live",
        "results": served,
        "answer": answer,
        "model": answer_model,
        "distilled": bool(answer),
        "distill_rejected": scored.rejected,
        "quality_score": quality_score,
        "quality_flags": list(quality_flags),
        "pooled": entry_id is not None,
        "scrubbed": scrubbed.findings or None,
        "entry_id": entry_id,
        "cached_at": fetched_iso,  # freshly fetched now
        "age_seconds": 0.0,
    }


def _humanize_age(seconds: float | None) -> str | None:
    """Compact relative age, e.g. '5 days ago', '3 hours ago', 'just now'."""
    if seconds is None:
        return None
    s = int(seconds)
    if s < 90:
        return "just now"
    if s < 5400:  # < 1.5h
        return f"{max(s // 60, 1)} min ago"
    if s < 129600:  # < 1.5d
        return f"{s // 3600} hours ago"
    return f"{s // 86400} days ago"


def format_for_agent(res: dict) -> str:
    """Render a resolution result as text for an agent's search tool."""
    src = res["source"]
    if src == "error":
        return f"Aimnis: live search unavailable ({res['error']})."
    if src == "quota":
        # Keyless caller out of today's live-search budget. Cached answers stay
        # free — this query just isn't in the pool yet. Only produced on the
        # hosted MCP edge (see mcp_http), so the copy can name the register tool.
        return (
            "[Aimnis · daily free live-search limit reached] This query isn't in the "
            "shared knowledge pool yet, and today's free live searches from your "
            "network are used up (cached answers remain free and unlimited; the limit "
            "resets daily, UTC). To keep searching live now, get a free API key: call "
            "the `register` tool with your user's email address (the key is returned "
            "immediately), or register at https://aimnis.com/register — then send it "
            "as 'Authorization: Bearer aim_...'. Please relay this to the user."
        )

    answer = res.get("answer")
    # A present `answer` is model-distilled → disclose it as AI-generated. Raw-snippet
    # results (no answer) are not, so they carry no such claim.
    ai = ", AI-generated" if answer else ""
    lines: list[str] = []
    if src == "cache":
        matched = res.get("matched_query")
        # Freshness phrase from the cached answer's content_at (date + relative age)
        # so the agent can judge staleness itself.
        cached_at = res.get("cached_at")
        age = _humanize_age(res.get("age_seconds"))
        when = ""
        if cached_at:
            when = f" cached {cached_at[:10]}" + (f" ({age})" if age else "")
        if res.get("match") == "semantic" and matched:
            # Surface the exact question this answer was cached for so the agent
            # can verify it matches its intent (esp. polarity: enable vs disable),
            # and give it a sanctioned escape hatch: retrying with reject_entry
            # skips this entry AND labels the mis-serve for ranking (explicit
            # dissatisfaction signal — see stats.hit_satisfaction).
            entry = res.get("entry_id")
            reject_hint = (
                f' If that is not what you are asking, search again with '
                f'reject_entry="{entry}" to skip this cached answer and search live.'
                if entry else
                " If that is not what you are asking, disregard it and request a "
                "live search."
            )
            lines.append(
                f'[Aimnis · cache hit (semantic match){ai} —{when}; this answer was cached '
                f'for the question: "{matched}".{reject_hint} Verify time-sensitive facts.]'
            )
        else:
            lines.append(
                f"[Aimnis · cache hit ({res['match']} match){ai} —{when}; "
                f"verify time-sensitive facts]"
            )
    elif answer:
        lines.append("[Aimnis · distilled answer (live), AI-generated — verify time-sensitive facts]")
    else:
        lines.append("[Aimnis · live results]")

    if answer:
        lines += [answer, ""]

    # Sources: numbered so the answer's [n] citations line up. When citation
    # routing is on and the entry is pooled, the printed link is the signed
    # /r/<token> redirect (so following it feeds the ranking signal); the real
    # host is shown inline in parentheses so the agent can still judge/trust the
    # source domain. Unpooled/live results and disabled routing keep the raw URL.
    from . import citations

    entry_id = res.get("entry_id")
    label = "Sources:" if answer else None
    if label:
        lines.append(label)
    for i, r in enumerate(res.get("results", []), 1):
        title = r.get("title") or "(no title)"
        url = r.get("url")
        routed = citations.route_url(entry_id, url)
        if routed:
            host = citations.host_of(url) if url else None
            head = f"{i}. {title} ({host})" if host else f"{i}. {title}"
            block = f"{head}\n   {routed}"
        else:
            block = f"{i}. {title}\n   {url}"
        if r.get("snippet"):
            block += f"\n   {r['snippet']}"
        lines.append(block)

    attribution = models.attribution_for(res.get("model"))
    if attribution:
        lines += ["", attribution]

    if not res.get("results") and not answer:
        lines.append("(no results)")
    return "\n".join(lines)
