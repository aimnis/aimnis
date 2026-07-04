"""Live-fallback web search for cache misses on the interactive path.

An ordered FALLBACK CHAIN of pluggable backends (config.search_preference):
  - Brave Search API — keyed, reliable, rich `extra_snippets` (good grounding).
  - Tavily — keyed, agent/RAG-oriented, clean extractable page content.
  - Exa — keyed, neural/embeddings search, distinct semantic recall.
  - SearXNG — keyless self-host floor; fine for dev but its scraping engines
    rate-limit/CAPTCHA under load, so it sits LAST, not as a load-bearing prod path.

live_search tries each usable (keyed, or keyless for SearXNG) backend in order and,
on a 429 / error / empty result, falls through to the next — so exhausting one
provider's free tier degrades gracefully instead of returning nothing. Keys unset
⇒ that provider is skipped; SearXNG always remains as the floor.
"""

from __future__ import annotations

import html
import logging
import re
from dataclasses import dataclass, field

import httpx

from .config import settings

log = logging.getLogger("aimnis.search")

_TAG_RE = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str
    # Extra page excerpts (Brave provides these); richer grounding for distill.
    extra_snippets: tuple[str, ...] = field(default_factory=tuple)


class SearchError(RuntimeError):
    pass


def _clean(text: str | None) -> str:
    """Strip HTML tags and unescape entities from a snippet."""
    if not text:
        return ""
    return html.unescape(_TAG_RE.sub("", text)).strip()


# --------------------------------------------------------------------------- #
# Backends
# --------------------------------------------------------------------------- #
async def _searxng_search(query: str, limit: int) -> list[SearchResult]:
    try:
        async with httpx.AsyncClient(timeout=settings.search_timeout_seconds) as client:
            resp = await client.get(
                f"{settings.searxng_url}/search",
                params={"q": query, "format": "json"},
                headers={"Accept": "application/json"},
            )
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchError(f"searxng search failed: {exc}") from exc

    results: list[SearchResult] = []
    for item in data.get("results", [])[:limit]:
        url = item.get("url")
        if not url:
            continue
        results.append(
            SearchResult(title=item.get("title") or "", url=url,
                         snippet=_clean(item.get("content")))
        )
    return results


def _parse_brave(data: dict, limit: int) -> list[SearchResult]:
    results: list[SearchResult] = []
    for item in (data.get("web") or {}).get("results", [])[:limit]:
        url = item.get("url")
        if not url:
            continue
        extra = tuple(_clean(s) for s in (item.get("extra_snippets") or []) if _clean(s))
        results.append(
            SearchResult(title=_clean(item.get("title")), url=url,
                         snippet=_clean(item.get("description")), extra_snippets=extra[:3])
        )
    return results


async def _brave_search(query: str, limit: int, api_key: str | None = None) -> list[SearchResult]:
    api_key = api_key or settings.brave_api_key
    if not api_key:
        raise SearchError("brave backend selected but AIMNIS_BRAVE_API_KEY is unset")
    headers = {
        "X-Subscription-Token": api_key,
        "Accept": "application/json",
        "Accept-Encoding": "gzip",
    }
    try:
        async with httpx.AsyncClient(timeout=settings.search_timeout_seconds) as client:
            resp = await client.get(
                settings.brave_endpoint,
                params={"q": query, "count": min(limit, 20)},
                headers=headers,
            )
            # Surface remaining monthly/second quota so usage is trackable.
            rl = resp.headers.get("x-ratelimit-remaining")
            if rl:
                log.info("brave quota remaining: %s", rl)
            if resp.status_code == 429:
                raise SearchError("brave rate limited (429)")
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchError(f"brave search failed: {exc}") from exc
    return _parse_brave(data, limit)


async def _tavily_search(query: str, limit: int, api_key: str | None = None) -> list[SearchResult]:
    api_key = api_key or settings.tavily_api_key
    if not api_key:
        raise SearchError("tavily backend selected but AIMNIS_TAVILY_API_KEY is unset")
    try:
        async with httpx.AsyncClient(timeout=settings.search_timeout_seconds) as client:
            resp = await client.post(
                settings.tavily_endpoint,
                headers={"Authorization": f"Bearer {api_key}",
                         "Content-Type": "application/json"},
                json={"query": query, "max_results": min(limit, 20),
                      "search_depth": "basic"},
            )
            if resp.status_code == 429:
                raise SearchError("tavily rate limited (429)")
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchError(f"tavily search failed: {exc}") from exc

    results: list[SearchResult] = []
    for item in (data.get("results") or [])[:limit]:
        url = item.get("url")
        if not url:
            continue
        results.append(
            SearchResult(title=_clean(item.get("title")), url=url,
                         snippet=_clean(item.get("content")))
        )
    return results


async def _exa_search(query: str, limit: int, api_key: str | None = None) -> list[SearchResult]:
    api_key = api_key or settings.exa_api_key
    if not api_key:
        raise SearchError("exa backend selected but AIMNIS_EXA_API_KEY is unset")
    try:
        async with httpx.AsyncClient(timeout=settings.search_timeout_seconds) as client:
            resp = await client.post(
                settings.exa_endpoint,
                headers={"x-api-key": api_key,
                         "Content-Type": "application/json"},
                json={"query": query, "numResults": min(limit, 20), "type": "auto",
                      "contents": {"text": {"maxCharacters": 500}}},
            )
            if resp.status_code == 429:
                raise SearchError("exa rate limited (429)")
            resp.raise_for_status()
            data = resp.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise SearchError(f"exa search failed: {exc}") from exc

    results: list[SearchResult] = []
    for item in (data.get("results") or [])[:limit]:
        url = item.get("url")
        if not url:
            continue
        results.append(
            SearchResult(title=_clean(item.get("title")), url=url,
                         snippet=_clean(item.get("text")))
        )
    return results


# --------------------------------------------------------------------------- #
# Dispatch + fallback chain
# --------------------------------------------------------------------------- #
# Name → module-level function NAME (resolved via globals() at call time, so
# tests can monkeypatch e.g. search._brave_search and have it take effect).
_BACKENDS = {
    "brave": "_brave_search",
    "tavily": "_tavily_search",
    "exa": "_exa_search",
    "searxng": "_searxng_search",
}

# A provider is skipped unless usable. SearXNG is keyless, so it's always usable
# and guarantees the chain is never empty.
_KEY_ATTR = {
    "brave": "brave_api_key",
    "tavily": "tavily_api_key",
    "exa": "exa_api_key",
}


def _usable(backend: str) -> bool:
    attr = _KEY_ATTR.get(backend)
    return attr is None or bool(getattr(settings, attr))


def _provider_chain() -> list[str]:
    """Ordered backends to try. `auto` → the full preference list; an explicit
    backend goes first, then the remaining preference list as fallback (so even a
    pinned backend degrades). Unknown names are ignored; SearXNG is always appended
    as the keyless floor if not already present."""
    pref = list(settings.search_preference)
    chosen = settings.search_backend
    if chosen != "auto" and chosen in _BACKENDS:
        order = [chosen] + [b for b in pref if b != chosen]
    else:
        order = pref
    order = [b for b in order if b in _BACKENDS]
    if "searxng" not in order:
        order.append("searxng")
    return order


async def live_search(
    query: str, *, limit: int | None = None, client_keys=None
) -> list[SearchResult]:
    """Run the fallback chain. With BYOK `client_keys` (apikeys.ClientKeys), the
    client's own provider+key is tried FIRST — their miss spends their quota — then
    the service chain as a graceful fallback. The client key is used only for this
    request (never for other users' queries — the BYOK ToS invariant)."""
    limit = limit or settings.search_result_limit

    # (backend, api_key_override) pairs; None ⇒ the service's configured key.
    attempts: list[tuple[str, str | None]] = []
    if client_keys is not None and client_keys.search_provider and client_keys.search_api_key:
        if client_keys.search_provider in _BACKENDS:
            attempts.append((client_keys.search_provider, client_keys.search_api_key))
    attempts += [(b, None) for b in _provider_chain()]

    last_err: SearchError | None = None
    tried = False
    for backend, key_override in attempts:
        if key_override is None and not _usable(backend):
            continue  # no service key → skip silently
        tried = True
        try:
            fn = globals()[_BACKENDS[backend]]
            if key_override is not None:
                results = await fn(query, limit, api_key=key_override)
            else:
                results = await fn(query, limit)
            if results:
                return results
            last_err = SearchError(f"{backend} returned no results")
        except SearchError as exc:
            last_err = exc
            log.warning("search backend %s failed: %s", backend, exc)
    if last_err:
        raise last_err
    if not tried:  # should be unreachable — searxng is always usable
        raise SearchError("no usable search backend configured")
    return []
