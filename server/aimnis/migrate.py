"""Minimal forward-only migration runner: applies migrations/*.sql in name order,
recording applied files in schema_migrations. Idempotent.

Usage:  python -m aimnis.migrate
"""

from __future__ import annotations

import asyncio
import pathlib

import asyncpg

from .config import settings

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parent.parent / "migrations"


async def migrate(dsn: str | None = None) -> None:
    conn = await asyncpg.connect(dsn or settings.database_url)
    try:
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  filename text PRIMARY KEY,"
            "  applied_at timestamptz NOT NULL DEFAULT now())"
        )
        applied = {r["filename"] for r in await conn.fetch("SELECT filename FROM schema_migrations")}

        for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
            if path.name in applied:
                continue
            async with conn.transaction():
                await conn.execute(path.read_text())
                await conn.execute(
                    "INSERT INTO schema_migrations (filename) VALUES ($1)", path.name
                )
            print(f"applied {path.name}")
        print("migrations up to date")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(migrate())
