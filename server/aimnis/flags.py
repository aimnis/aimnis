"""Live operator toggles (service_flag table) — flipped without a redeploy.

Currently just `registration_open`: gates self-serve eval-key issuance. Defaults to
open if the row is missing so a fresh DB isn't accidentally locked out.
"""

from __future__ import annotations

import asyncpg

REGISTRATION_OPEN = "registration_open"


async def get_flag(pool: asyncpg.Pool, name: str, *, default: bool = True) -> bool:
    val = await pool.fetchval("SELECT enabled FROM service_flag WHERE name=$1", name)
    return default if val is None else bool(val)


async def set_flag(pool: asyncpg.Pool, name: str, enabled: bool) -> None:
    await pool.execute(
        "INSERT INTO service_flag (name, enabled) VALUES ($1,$2) "
        "ON CONFLICT (name) DO UPDATE SET enabled=EXCLUDED.enabled, updated_at=now()",
        name, enabled,
    )


async def registration_open(pool: asyncpg.Pool) -> bool:
    return await get_flag(pool, REGISTRATION_OPEN, default=True)
