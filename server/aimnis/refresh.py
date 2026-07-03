"""Background re-distill / refresh queue.

Fixes the gap where an entry pooled snippet-only during a 429 blip is served from
cache forever and never re-distilled. This batch runner selects such entries (plus
explicitly-stale and below-min-quality ones), re-distills from their ALREADY-STORED
sources (no new search spend), quality-gates + judges the result, and upgrades the
entry in place — but only when the new answer is at least as good as the old, so a
refresh never makes an entry worse.

Spends the reserved background quota budget (config.refresh_purpose). When that
budget is exhausted the batch stops cleanly. Run one-shot (cron / manual):

    python -m aimnis.refresh
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field

import asyncpg

from . import models
from . import pool as pool_mod
from . import quota
from . import resolve
from .config import settings

log = logging.getLogger("aimnis.refresh")


@dataclass
class RefreshReport:
    candidates: int = 0
    upgraded: int = 0
    rejected: int = 0            # distilled but failed the quality gate
    failed: int = 0             # distill errored / empty
    skipped_no_improve: int = 0  # new answer scored below the existing one
    stopped_reason: str | None = None  # 'quota' | 'distill_disabled' | None (ran to completion)
    by_reason: dict = field(default_factory=dict)


async def run_refresh(
    db: asyncpg.Pool, *, limit: int | None = None, purpose: str | None = None,
    judge_enabled: bool | None = None,
) -> RefreshReport:
    limit = limit or settings.refresh_batch_limit
    purpose = purpose or settings.refresh_purpose
    judge_enabled = settings.refresh_judge_enabled if judge_enabled is None else judge_enabled

    report = RefreshReport()
    if not (settings.distill_enabled and settings.openrouter_api_key):
        report.stopped_reason = "distill_disabled"
        return report

    candidates = await pool_mod.select_refresh_candidates(
        db, limit=limit, min_quality=settings.refresh_min_quality_score,
        min_follow_through=settings.refresh_min_follow_through,
        min_hits_for_follow_through=settings.refresh_follow_through_min_hits,
    )
    report.candidates = len(candidates)

    for i, c in enumerate(candidates):
        if i and settings.refresh_delay_seconds:
            await asyncio.sleep(settings.refresh_delay_seconds)  # pace to avoid provider 429s
        report.by_reason[c["reason"]] = report.by_reason.get(c["reason"], 0) + 1
        try:
            scored = await resolve._distill_and_score(
                db, c["query_text"], c["query_hash"], c["sources"],
                purpose=purpose, judge_enabled=judge_enabled,
                judge_purpose=settings.quality_judge_purpose,
            )
        except quota.QuotaExceeded as exc:
            # Background budget exhausted — stop cleanly, leaving the rest for next run.
            report.stopped_reason = "quota"
            log.info("refresh stopped: quota (%s) after %d upgrades", exc.reason, report.upgraded)
            break

        if scored.answer is None:
            if scored.rejected:
                report.rejected += 1
            else:
                report.failed += 1
            continue

        # Never make an entry worse: only replace an existing answer if the new
        # score is at least as high. (Snippet-only entries have no prior score.)
        old = c["quality_score"]
        if old is not None and (scored.quality_score or 0.0) < old:
            report.skipped_no_improve += 1
            continue

        flags = models.compliance_for(scored.model)
        await pool_mod.update_answer(
            db, c["id"],
            answer_text=scored.answer, model=scored.model,
            quality_score=scored.quality_score,
            provenance={"source": "refresh", "distilled": True,
                        "quality_flags": list(scored.flags), "reason": c["reason"]},
            status="active",
            output_trainable=flags.get("output_trainable", False),
            attribution_required=flags.get("attribution_required", False),
            no_grounded_cache=flags.get("no_grounded_cache", False),
        )
        report.upgraded += 1

    return report


async def _main() -> None:
    from .db import close_pool, get_pool

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    db = await get_pool()
    try:
        rep = await run_refresh(db)
    finally:
        await close_pool()
    print(
        f"refresh: {rep.upgraded} upgraded, {rep.rejected} rejected, "
        f"{rep.failed} failed, {rep.skipped_no_improve} no-improve "
        f"of {rep.candidates} candidates {dict(rep.by_reason)}"
        + (f" — stopped: {rep.stopped_reason}" if rep.stopped_reason else "")
    )


def main() -> None:
    asyncio.run(_main())


if __name__ == "__main__":
    main()
