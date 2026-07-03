"""Ingress secret/PII scrubbing — runs at the TOP of resolution, before a query
is embedded, sent to any third party (Brave/OpenRouter), or persisted.

The coding niche is the reason this matters: agent queries carry code context that
can include API keys, tokens, private keys, connection strings, and PII. We redact
in place with typed placeholders (⟨AWS_KEY⟩) so the query stays semantically
searchable and two queries differing only in their secret collapse to the same
cache key. If a query is secret-DENSE (redaction may be incomplete, and it's
unlikely to be reusable knowledge anyway) it is served live-only and never pooled.

First cut: high-precision key-format regexes + connection strings + assignments +
PII (email) + a conservative Shannon-entropy pass for unknown-format secrets.
Broader coverage (detect-secrets / TruffleHog / GitGuardian rule packs) can be
layered in later; the entropy threshold is a tunable that belongs with the private
anti-abuse config.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from math import log2

from .config import settings

# Ordered most-specific first. Each entry: (type, compiled regex). The matched
# span is replaced with ⟨TYPE⟩. For assignments we redact only the value.
_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("PRIVATE_KEY", re.compile(
        r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----.*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        re.S)),
    ("CONNECTION_STRING", re.compile(
        r"\b(?:postgres(?:ql)?|mysql|mongodb(?:\+srv)?|redis|amqp|https?)://"
        r"[^\s:/@]+:[^\s:/@]+@[^\s]+")),  # scheme://user:pass@host
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]+\.eyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+")),
    ("AWS_KEY", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("GITHUB_TOKEN", re.compile(r"\b(?:gh[posru]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{22,})")),
    ("SLACK_TOKEN", re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}")),
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("STRIPE_KEY", re.compile(r"\b[sr]k_(?:live|test)_[A-Za-z0-9]{16,}")),
    # OpenAI / OpenRouter / Anthropic-style sk- keys (incl. sk-proj-, sk-or-v1-, sk-ant-).
    ("API_KEY", re.compile(r"\bsk-(?:proj-|or-v1-|ant-)?[A-Za-z0-9_\-]{20,}")),
    ("BRAVE_KEY", re.compile(r"\bBSA[A-Za-z0-9_\-]{20,}")),
    ("BEARER_TOKEN", re.compile(r"\b[Bb]earer\s+[A-Za-z0-9._\-]{16,}")),
]

# key=value / key: value assignments — redact the VALUE only.
_ASSIGN_RE = re.compile(
    r"(?i)\b(password|passwd|pwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token|token)"
    r"(\s*[:=]\s*)(['\"]?)([^\s'\"⟨⟩]{4,})(\3)")  # exclude ⟨⟩ so we don't re-redact placeholders

_EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b")

# Candidate runs for the entropy pass (12 = cheap prefilter floor; the real
# min length is enforced against settings in code).
_TOKEN_RE = re.compile(r"[A-Za-z0-9_\-+/=]{12,}")

_HAS_ALPHA = re.compile(r"[A-Za-z]")
_HAS_DIGIT = re.compile(r"[0-9]")


@dataclass(frozen=True)
class ScrubResult:
    text: str                          # redacted, semantically-searchable query
    findings: dict = field(default_factory=dict)  # type -> count
    secret_count: int = 0
    safe_to_pool: bool = True          # False when secret-dense → live-only, never persist


def _entropy(s: str) -> float:
    if not s:
        return 0.0
    n = len(s)
    return -sum((c / n) * log2(c / n) for c in Counter(s).values())


def scrub(text: str) -> ScrubResult:
    if not settings.scrub_enabled or not text:
        return ScrubResult(text=text)

    findings: Counter = Counter()

    def _tally(kind: str):
        findings[kind] += 1

    out = text
    for kind, rx in _PATTERNS:
        out = rx.sub(lambda m, k=kind: (_tally(k) or f"⟨{k}⟩"), out)

    # Assignments: keep the key + operator, redact the value.
    out = _ASSIGN_RE.sub(lambda m: (_tally("SECRET") or f"{m.group(1)}{m.group(2)}⟨SECRET⟩"), out)

    out = _EMAIL_RE.sub(lambda m: (_tally("EMAIL") or "⟨EMAIL⟩"), out)

    # Conservative entropy pass for unknown-format secrets: long, mixed-charset,
    # high-entropy runs only (leaves normal code identifiers and prose intact).
    if settings.scrub_entropy_enabled:
        def _maybe(m: re.Match) -> str:
            tok = m.group(0)
            if (len(tok) >= settings.scrub_entropy_min_len
                    and _HAS_ALPHA.search(tok) and _HAS_DIGIT.search(tok)
                    and _entropy(tok) >= settings.scrub_entropy_threshold):
                _tally("HIGH_ENTROPY")
                return "⟨HIGH_ENTROPY⟩"
            return tok
        out = _TOKEN_RE.sub(_maybe, out)

    secret_count = sum(findings.values())
    safe_to_pool = secret_count <= settings.scrub_max_secrets_to_pool
    return ScrubResult(
        text=out, findings=dict(findings),
        secret_count=secret_count, safe_to_pool=safe_to_pool,
    )
