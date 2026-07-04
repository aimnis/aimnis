"""Runtime configuration. Values come from the environment (AIMNIS_ prefix) / .env.

Anti-abuse thresholds are intentionally NOT here — they load from a separate
config that ships with safe example defaults; tuned production values are injected
from a private overlay at deploy, so publishing this file doesn't reveal them.
"""

import os
import pathlib
from typing import Annotated

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


def _resolve_env_file() -> str | None:
    """Locate server/.env robustly.

    The MCP server is spawned from the project root, so a bare ".env" (CWD-relative)
    would miss the key. We can't just use `__file__.parent.parent` either: under a
    NON-editable install that path points into site-packages, where there is no
    .env — which silently disables distillation AND drops the Brave backend (the
    exact bug hit on 2026-07-02). So: honour an explicit AIMNIS_ENV_FILE, else walk
    up from this module looking for a real .env. Return None if none is found (env
    vars still work; pydantic just has no file to read)."""
    override = os.environ.get("AIMNIS_ENV_FILE")
    if override:
        return override
    here = pathlib.Path(__file__).resolve()
    for parent in here.parents:
        candidate = parent / ".env"
        if candidate.is_file():
            return str(candidate)
    return None


_ENV_FILE = _resolve_env_file()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="AIMNIS_", env_file=_ENV_FILE, extra="ignore"
    )

    database_url: str = "postgresql://aimnis:aimnis@localhost:5432/aimnis"

    # Local embedding model (never spend upstream quota on embeddings).
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = 384

    # Semantic-cache hit threshold as MAX cosine distance (pgvector `<=>`).
    # distance = 1 - cosine_similarity; 0.15 ⇒ similarity ≥ 0.85. Used as the
    # single-shot accept threshold ONLY when reranking is disabled; with reranking
    # on it is superseded by rerank_recall_max_distance + rerank_min_score below.
    cache_max_distance: float = 0.15

    # Semantic-match reranking (cross-encoder precision over bi-encoder recall).
    # The bi-encoder (bge-small) embeds each query independently, so it is weak at
    # separating look-alike opposite-intent queries ("enable X" vs "disable X").
    # So: ANN retrieves the top-k nearest candidates, then a cross-encoder rescores
    # the candidate QUESTIONS against the incoming query (it jointly encodes the
    # pair, which is far better at polarity). The reranker ranks + filters gross
    # mismatches; the residual hard case (a single flipped word) is delegated to the
    # calling agent by echoing the matched question in the tool output, so the agent
    # makes the final polarity call. Runs locally on CPU (fastembed ONNX, no GPU, no
    # upstream spend). Degrades to the plain cache_max_distance gate if it can't load.
    rerank_enabled: bool = True
    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # ~80 MB CPU ONNX cross-encoder
    rerank_candidates: int = 5           # top-k pulled from ANN to feed the reranker
    rerank_recall_max_distance: float = 0.30  # generous ANN recall filter (cos sim ≥ 0.70)
    rerank_min_score: float = 0.5        # sigmoid(logit) accept floor for a semantic hit

    # Live-fallback search for cache misses. This is an ordered FALLBACK CHAIN, not
    # a single backend: each keyed provider is tried in turn and, on a 429/error/
    # empty result, live_search falls through to the next usable one, so a throttled
    # or dry provider degrades instead of returning nothing.
    #   "auto"    → full chain, preference order below, keyless SearXNG last
    #   "brave"|"tavily"|"exa"|"searxng" → that provider first, then the rest as fallback
    # Preference order (auto): Brave (rich extra_snippets) → Tavily (clean RAG
    # content) → Exa (neural recall) → SearXNG (keyless self-host floor). A provider
    # with no key is skipped; SearXNG needs none, so the chain never has zero backends.
    # This exists so launch-day traffic that exhausts one free tier (e.g. Brave's
    # ~2k/mo) rolls onto the next instead of failing.
    search_backend: str = "auto"
    search_preference: tuple[str, ...] = ("brave", "tavily", "exa", "searxng")
    searxng_url: str = "http://localhost:8888"
    brave_api_key: str | None = None
    brave_endpoint: str = "https://api.search.brave.com/res/v1/web/search"
    # Tavily — agent/RAG-oriented search, clean extractable page content.
    tavily_api_key: str | None = None
    tavily_endpoint: str = "https://api.tavily.com/search"
    # Exa — neural/embeddings search, good semantic recall as a distinct angle.
    exa_api_key: str | None = None
    exa_endpoint: str = "https://api.exa.ai/search"
    search_timeout_seconds: float = 10.0
    # Two caps: fetch WIDE from providers, then serve/ground on the best K. The
    # extra candidates are stored (not discarded at the cap) so the source selector
    # — and later re-selection on click feedback — has a pool to pick from.
    search_fetch_limit: int = 20         # candidates pulled from providers + stored
    search_result_limit: int = 8         # top-K served, cited, and distilled from

    # Feature-based source selector weights (aimnis.select). Transparent, cold-start
    # -safe priors; a learned ranker later replaces the scoring core, not the shape.
    # Clicks are weighted high but only contribute once follow-through data exists.
    select_w_rank: float = 1.0           # provider-order prior (earlier = better)
    select_w_overlap: float = 0.6        # lexical query↔source term overlap
    select_w_freshness: float = 0.3      # newer fetched_at preferred
    select_w_clicks: float = 1.0         # click follow-through (beats-its-position)

    # Default quota key label used by the ledger.
    quota_key_label: str = "primary-free"

    # Public flywheel dashboard (Gate 1 instrument).
    dashboard_host: str = "127.0.0.1"
    dashboard_port: int = 8080

    # HTTP gateway (the remote REST/MCP edge) — exposes POST /v1/search + GET /v1/stats
    # so remote agents can use the HOSTED pool without ever touching Postgres directly.
    # FAIL-CLOSED: with no keys configured the /v1 routes refuse (503), so a public
    # deploy can never accidentally serve unauthenticated, quota-spending search to the
    # whole internet. Set AIMNIS_GATEWAY_API_KEYS=key1,key2 (comma-separated) to enable.
    # NoDecode: skip pydantic-settings' default JSON decoding of this list-typed
    # field from the env so the validator below can accept a plain comma-separated
    # string (AIMNIS_GATEWAY_API_KEYS=key1,key2).
    gateway_api_keys: Annotated[list[str], NoDecode] = []

    # Client side of the same edge: when AIMNIS_GATEWAY_URL is set, the MCP server acts
    # as a thin CLIENT of a remote Aimnis gateway (HTTP) instead of resolving against a
    # local Postgres — this is how a user points Claude Code / OpenCode at the hosted
    # service. Unset ⇒ local/self-host mode (resolve against the local pool). The client
    # sends AIMNIS_GATEWAY_CLIENT_API_KEY as its bearer token.
    gateway_url: str | None = None
    gateway_client_api_key: str | None = None
    gateway_timeout_seconds: float = 30.0

    # Self-serve eval portal (aimnis.com). Client keys issued by the portal live in
    # the DB (api_client) and are metered; the env-var gateway_api_keys above remain
    # an ADMIN/bootstrap path (unlimited, unmetered). Default per-key caps ration the
    # shared upstream quota — keep rpd well under the ~1,000/day service ceiling so a
    # single key can't starve the pool for everyone; hits are free, only misses spend.
    client_default_rpm: int = 20
    client_default_rpd: int = 200
    # BYOK — a client may attach their own OpenRouter / search-provider keys; their
    # misses then spend THEIR quota, so they get much higher caps without touching
    # the shared ceiling ("your keys, your limits"). Keys are pgcrypto-encrypted
    # under byok_secret; unset ⇒ BYOK is disabled fail-closed (form hidden, nothing
    # stored). Caps stay finite as an abuse backstop — hits still cost us compute.
    byok_secret: str | None = None
    byok_rpm: int = 60
    byok_rpd: int = 5000
    # Admin key for the portal's operator endpoints (pause/resume registration, etc.).
    # Unset ⇒ those endpoints are disabled (fail-closed) — flip flags in the DB instead.
    admin_api_key: str | None = None
    # Public base URL of the portal, used in issued-key / waitlist emails and links.
    portal_base_url: str = "https://aimnis.com"

    # Transactional email (Resend). Real inbox delivery needs a provider + DNS auth
    # (SPF/DKIM/DMARC on aimnis.com) — the Railway box can't send mail directly. With
    # no key set, email is a logged no-op so the portal still works in dev / self-host.
    resend_api_key: str | None = None
    resend_endpoint: str = "https://api.resend.com/emails"
    email_from: str = "Aimnis <support@aimnis.com>"
    # Anti-abuse: max portal form submissions (/register, /waitlist) per client IP
    # per hour. In-process counter — sufficient for a single-instance deploy.
    portal_ip_hourly: int = 5
    # Optional shared try-out key, pre-filled in the public MCP server card's config
    # form so directory visitors (e.g. Smithery playground) can try the server without
    # registering. DELIBERATELY PUBLIC — must be a dedicated metered DB client key
    # (standard caps bound the abuse; revoke by prefix to rotate). Never the admin key.
    demo_api_key: str | None = None

    # Citation routing — cited source links can be routed through a signed
    # /r/<token> redirect that logs a click (aggregate relevance signal: which
    # pooled entries earn follow-through, which sources are dead weight) before
    # 302-ing to the destination. FAIL-CLOSED and privacy-scoped:
    #   * No routing unless citation_signing_secret is set (HMAC-signed tokens are
    #     forgery-proof, and we only ever resolve URLs already in our own pool — so
    #     /r can never be abused as an open redirector).
    #   * The link base is citation_public_base_url (else gateway_url). Unset ⇒ no
    #     routing at all, raw URLs are emitted. Self-host points this at your OWN
    #     gateway, so clicks stay on your box and feed your own ranking.
    #   * The click log records entry+source+host+timestamp ONLY — never IP/UA/user.
    citation_routing_enabled: bool = True
    citation_signing_secret: str | None = None
    citation_public_base_url: str | None = None  # e.g. https://api.aimnis.org

    @field_validator("gateway_api_keys", mode="before")
    @classmethod
    def _split_api_keys(cls, v):
        # Accept a comma-separated string from the environment (pydantic would otherwise
        # demand JSON for a list field) as well as a real list.
        if isinstance(v, str):
            return [k.strip() for k in v.split(",") if k.strip()]
        return v

    # Ingress secret/PII scrubbing (runs before embed/search/distill/persist).
    # The entropy thresholds are the tunable, false-positive-sensitive part — real
    # values belong in the private anti-abuse config; these are safe defaults.
    scrub_enabled: bool = True
    scrub_max_secrets_to_pool: int = 3      # more redactions than this → live-only, never pool
    scrub_entropy_enabled: bool = True
    scrub_entropy_min_len: int = 20
    scrub_entropy_threshold: float = 3.5

    # LLM distillation of live-search misses into a pooled answer (OpenRouter :free).
    # Off unless an API key is present, so the service degrades to raw snippets
    # (and spends zero quota) without one. Runs on the interactive miss path under
    # a tight timeout; on quota-denied / timeout / error it falls back to snippets.
    openrouter_api_key: str | None = None
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    distill_enabled: bool = True
    # Ordered fallback chain (OpenRouter caps the `models` array at 3). :free
    # models 429 constantly because popular ones share saturated providers
    # (Venice/OpenInference), so we deliberately pick models on DISTINCT dedicated
    # providers — one request routes to the first available. Lineup rotates; verify
    # with `GET /models` and swap ids here (no code change). Order favours the
    # coding niche (a code-tuned model first).
    distill_models: list[str] = [
        "cohere/north-mini-code:free",                 # Cohere — code-tuned
        "nvidia/nemotron-3-super-120b-a12b:free",      # Nvidia — large general
        "openai/gpt-oss-20b:free",                     # Darkbloom — fast fallback
    ]
    distill_timeout_seconds: float = 20.0   # free models are slow; degrade to snippets past this
    distill_max_tokens: int = 1024          # headroom for reasoning models' hidden reasoning tokens
    distill_purpose: str = "interactive_fallback"  # quota sub-budget this spends from
    # Courtesy attribution headers OpenRouter uses for its app leaderboard.
    openrouter_referer: str = "https://aimnis.com"
    openrouter_title: str = "Aimnis"

    # Quality gate — decides whether a distilled answer is trustworthy enough to
    # POOL (and, per the mission, eventually feed training). Heuristics always run
    # (cheap, no quota); a bad answer degrades to serving raw snippets instead of
    # poisoning the pool. The LLM judge is opt-in (spends quota) and intended for
    # the background path.
    quality_min_answer_chars: int = 40
    quality_max_answer_chars: int = 4000
    quality_min_score: float = 0.5          # heuristic soft-score acceptance floor (0..1)
    quality_judge_enabled: bool = False     # LLM-as-judge; off on the interactive path
    quality_judge_purpose: str = "quality_judge"  # quota purpose for judge calls
    quality_judge_min_score: int = 3        # 1..5; below this the answer is rejected

    # Background re-distill / refresh queue. Upgrades snippet-only entries (pooled
    # during a 429 blip) and low-quality entries to distilled+judged answers,
    # re-using their ALREADY-STORED sources (no new search spend). Spends the
    # reserved background quota budget; the judge is affordable here (off the
    # interactive path).
    refresh_batch_limit: int = 25
    refresh_purpose: str = "background_precompute"   # quota purpose for refresh distills
    refresh_min_quality_score: float = 0.6           # entries below this (with an answer) are candidates
    refresh_judge_enabled: bool = True               # run the LLM judge on the background path
    # Follow-through signal: an entry whose distilled answer keeps sending the agent
    # on to a cited source (clicks-per-hit high) is likely THIN — re-distill it to a
    # fuller answer. This is the only sound serving-quality use of the click signal:
    # 0 clicks can mean a perfect self-sufficient answer, so clicks never demote or
    # reorder a served result — they only flag which answers to improve. Gated on a
    # min hit count so a 1-hit/1-click ratio isn't trusted as noise.
    refresh_min_follow_through: float = 1.0          # clicks-per-hit at/above which a distilled entry is a re-distill candidate
    refresh_follow_through_min_hits: int = 5         # require this many hits before trusting the ratio
    # Pace between candidates — free providers 429 when a batch bursts calls at
    # them faster than their per-second limits (distill + judge = 2 calls each).
    refresh_delay_seconds: float = 2.0


settings = Settings()
