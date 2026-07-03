"""Query normalization + hashing for exact dedup / single-flight keys."""

from __future__ import annotations

import hashlib
import re

_WS = re.compile(r"\s+")


def normalize(query: str) -> str:
    """Canonical form for exact-match dedup: lowercased, whitespace-collapsed, trimmed."""
    return _WS.sub(" ", query.strip().lower())


def query_hash(query_norm: str) -> str:
    """Stable hash of the normalized query (exact-dedup / single-flight key)."""
    return hashlib.sha256(query_norm.encode("utf-8")).hexdigest()
