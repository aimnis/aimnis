"""Test fixtures. Integration tests run against a real Postgres (pgvector image).

Set AIMNIS_DATABASE_URL (or use the docker-compose default) before running; if no
DB is reachable the suite skips rather than fails.
"""

from __future__ import annotations

from urllib.parse import urlparse, urlunparse

import asyncpg
import pytest
import pytest_asyncio
from pgvector.asyncpg import register_vector

from aimnis.config import settings
from aimnis.migrate import migrate

# Tests run against a SEPARATE database so `pytest` never truncates the dev /
# dogfood pool that the live MCP server reads. Derived from the configured DSN by
# suffixing the db name (aimnis -> aimnis_test); created on first use.
_parts = urlparse(settings.database_url)
TEST_DSN = urlunparse(_parts._replace(path=(_parts.path.rstrip("/") or "/aimnis") + "_test"))

_migrated = False


async def _ensure_schema() -> bool:
    global _migrated
    try:
        admin = await asyncpg.connect(settings.database_url, timeout=3)
    except Exception:
        return False
    try:
        test_db = urlparse(TEST_DSN).path.lstrip("/")
        exists = await admin.fetchval(
            "SELECT 1 FROM pg_database WHERE datname = $1", test_db
        )
        if not exists:
            await admin.execute(f'CREATE DATABASE "{test_db}"')
    finally:
        await admin.close()
    if not _migrated:
        await migrate(TEST_DSN)
        _migrated = True
    return True


@pytest_asyncio.fixture
async def clean():
    """A pool with deterministic test keys and an empty upstream_call table."""
    if not await _ensure_schema():
        pytest.skip("Postgres not available at AIMNIS_DATABASE_URL")

    pool = await asyncpg.create_pool(
        TEST_DSN, min_size=1, max_size=5, init=register_vector
    )

    # test-rpm: tiny per-minute limit, no purpose budgets.
    await pool.execute(
        "INSERT INTO quota_key (label, provider, rpm_limit, rpd_limit) "
        "VALUES ('test-rpm', 'openrouter', 2, 1000) "
        "ON CONFLICT (label) DO UPDATE SET rpm_limit = 2, rpd_limit = 1000, active = true"
    )
    # test-day: tiny per-day limit + a purpose budget of 2 for background_precompute.
    await pool.execute(
        "INSERT INTO quota_key (label, provider, rpm_limit, rpd_limit) "
        "VALUES ('test-day', 'openrouter', 100, 3) "
        "ON CONFLICT (label) DO UPDATE SET rpm_limit = 100, rpd_limit = 3, active = true"
    )
    await pool.execute(
        "INSERT INTO quota_budget (key_id, purpose, daily_limit) "
        "SELECT id, 'background_precompute', 2 FROM quota_key WHERE label = 'test-day' "
        "ON CONFLICT (key_id, purpose) DO UPDATE SET daily_limit = 2"
    )
    # test-inactive: exercises the disabled-key path.
    await pool.execute(
        "INSERT INTO quota_key (label, provider, rpm_limit, rpd_limit, active) "
        "VALUES ('test-inactive', 'openrouter', 10, 10, false) "
        "ON CONFLICT (label) DO UPDATE SET active = false"
    )

    await pool.execute("TRUNCATE upstream_call")
    await pool.execute("TRUNCATE pool_entry CASCADE")  # cascades to citation_click (FK)
    await pool.execute("TRUNCATE lookup_event")
    await pool.execute("TRUNCATE api_client CASCADE")  # cascades to api_request (FK)
    await pool.execute("TRUNCATE waitlist")
    # Registration defaults to open; reset in case a test paused it.
    await pool.execute(
        "INSERT INTO service_flag (name, enabled) VALUES ('registration_open', true) "
        "ON CONFLICT (name) DO UPDATE SET enabled = true"
    )
    try:
        yield pool
    finally:
        await pool.close()
