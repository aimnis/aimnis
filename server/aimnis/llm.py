"""OpenRouter chat client — distills live-search snippets into a grounded answer.

This is the first quota-spending path. The caller reserves quota BEFORE invoking
`distill` and records the outcome after (see resolve._distill), because a failed
call — including a 429 — still burns real quota.

Errors are classified so the caller can record the right ledger status:
  - LLMRateLimited  → 'rate_limited' (HTTP 429)
  - LLMTimeout      → 'timeout'
  - LLMError        → 'error' (other non-2xx / transport / parse failures)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import httpx

from .config import settings

_SYSTEM = (
    "You are a research assistant for software engineers. Using ONLY the numbered "
    "web search results provided, write a concise, accurate answer to the query. "
    "Cite sources inline as [n] matching the result numbers. Prefer exact API names, "
    "flags, and short code over prose. If the results do not contain the answer, say "
    "so in one sentence — do not invent facts, URLs, or APIs.\n"
    "Output ONLY the final answer. Do not include any reasoning, planning, preamble, "
    "or meta-commentary (no 'We need to…', no 'Let's examine…')."
)


class LLMError(RuntimeError):
    def __init__(self, message: str, *, http_status: int | None = None) -> None:
        super().__init__(message)
        self.http_status = http_status


class LLMRateLimited(LLMError):
    pass


class LLMTimeout(LLMError):
    pass


@dataclass(frozen=True)
class ChatResult:
    content: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass(frozen=True)
class DistillResult:
    answer_text: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None


async def complete(
    messages: list[dict],
    *,
    models: list[str] | None = None,
    max_tokens: int | None = None,
    temperature: float = 0.2,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,  # for tests (MockTransport)
) -> ChatResult:
    """One chat completion via OpenRouter, routed across the provider-diverse
    fallback chain. Shared by distillation and the quality judge."""
    chain = (models or settings.distill_models)[:3]  # OpenRouter caps `models` at 3
    api_key = api_key or settings.openrouter_api_key
    base_url = base_url or settings.openrouter_base_url
    timeout = timeout if timeout is not None else settings.distill_timeout_seconds
    max_tokens = max_tokens if max_tokens is not None else settings.distill_max_tokens
    if not api_key:
        raise LLMError("no OpenRouter API key configured")
    if not chain:
        raise LLMError("no models configured")

    payload = {
        "messages": messages, "max_tokens": max_tokens, "temperature": temperature,
        # Several free-tier chain members (nemotron, gpt-oss) are reasoning models
        # that otherwise spend the completion budget on hidden reasoning tokens and
        # leave `content` empty. Capping reasoning effort low forces them to reserve
        # room for the actual answer; non-reasoning models ignore this field.
        "reasoning": {"effort": "low", "exclude": True},
    }
    # A single request OpenRouter routes across the chain (first available wins).
    if len(chain) > 1:
        payload["models"] = chain
    else:
        payload["model"] = chain[0]
    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": settings.openrouter_referer,
        "X-Title": settings.openrouter_title,
    }

    try:
        async with httpx.AsyncClient(
            base_url=base_url, timeout=timeout, transport=transport
        ) as client:
            resp = await client.post("/chat/completions", json=payload, headers=headers)
    except httpx.TimeoutException as exc:
        raise LLMTimeout(f"chat timed out after {timeout}s") from exc
    except httpx.HTTPError as exc:
        raise LLMError(f"chat transport error: {exc}") from exc

    if resp.status_code == 429:
        raise LLMRateLimited("OpenRouter rate limited", http_status=429)
    if resp.status_code >= 400:
        raise LLMError(
            f"OpenRouter returned {resp.status_code}: {resp.text[:200]}",
            http_status=resp.status_code,
        )

    try:
        data = resp.json()
        msg = data["choices"][0]["message"]
        # Use the final answer only; a null content (reasoning-only response) is
        # treated as empty so the caller degrades rather than serving scratchpad.
        content = (msg.get("content") or "").strip()
    except (ValueError, KeyError, IndexError, TypeError, AttributeError) as exc:
        raise LLMError(f"unparseable OpenRouter response: {exc}") from exc
    if not content:
        # Name the actual responding model + finish_reason (not just the reserved
        # primary) — this is the dominant failure mode on reasoning-capable free
        # models and was previously invisible in the ledger's error column.
        finish_reason = (data.get("choices") or [{}])[0].get("finish_reason")
        raise LLMError(
            f"OpenRouter returned an empty answer "
            f"(model={data.get('model')!r}, finish_reason={finish_reason!r})"
        )

    usage = data.get("usage") or {}
    return ChatResult(
        content=content,
        model=data.get("model") or chain[0],
        prompt_tokens=usage.get("prompt_tokens"),
        completion_tokens=usage.get("completion_tokens"),
    )


def _build_messages(query: str, results: Sequence[Mapping]) -> list[dict]:
    blocks = []
    for i, r in enumerate(results, 1):
        title = r.get("title") or "(no title)"
        url = r.get("url") or ""
        snippet = r.get("snippet") or ""
        # Fold in richer page excerpts when the backend provides them (Brave).
        extra = r.get("extra_snippets") or ()
        body = "\n".join([snippet, *extra]).strip()
        blocks.append(f"[{i}] {title}\n{url}\n{body}".strip())
    corpus = "\n\n".join(blocks) if blocks else "(no results)"
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"Query: {query}\n\nSearch results:\n{corpus}"},
    ]


async def distill(
    query: str,
    results: Sequence[Mapping],
    *,
    models: list[str] | None = None,
    timeout: float | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    transport: httpx.AsyncBaseTransport | None = None,  # for tests (MockTransport)
) -> DistillResult:
    out = await complete(
        _build_messages(query, results),
        models=models, timeout=timeout, api_key=api_key,
        base_url=base_url, transport=transport,
    )
    return DistillResult(
        answer_text=out.content, model=out.model,
        prompt_tokens=out.prompt_tokens, completion_tokens=out.completion_tokens,
    )
