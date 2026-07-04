"""Self-serve eval portal (aimnis.com).

Public marketing landing + self-serve eval-key registration, a terms page, and a
waitlist for when intake is paused. Server-rendered HTML (inline CSS, no external
deps — same constraints as the flywheel dashboard).

Routes:
    GET  /                     landing — what Aimnis is + why
    GET  /setup                per-agent setup instructions (hosted MCP endpoint + REST)
    GET  /llms.txt             plain-text summary for LLM/agent crawlers
    GET  /.well-known/api-catalog  RFC 9727 linkset of machine-usable endpoints
                               (advertised via an RFC 8288 Link header on /)
    GET  /terms                terms of use
    GET  /register             registration form (or "at capacity" + waitlist if paused)
    POST /register             issue a key (open) → email it (email-only in prod); else waitlist
    POST /waitlist             capture an email for capacity notifications
    POST /admin/registration   operator pause/resume toggle (X-Admin-Key)
    GET  /admin/clients        registered clients + usage counts (X-Admin-Key)

Issuance is INSTANT when `registration_open`. Each issued key is metered (per-key
rate + daily caps, see apikeys/gateway) and revocable at any time. Operators reserve
the right to revoke any key at any time — stated in the terms and on the success page.
"""

from __future__ import annotations

import hmac
import html
import re
import time

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response

from . import apikeys, db, email as email_mod, flags, stats
from .config import settings

router = APIRouter()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _valid_email(e: str) -> bool:
    return bool(e) and len(e) <= 254 and _EMAIL_RE.match(e) is not None


# --------------------------------------------------------------------------- #
# Per-IP form throttle (anti key-farming)
# --------------------------------------------------------------------------- #
# Keys are handed out self-serve, so the scarce thing to protect is upstream
# quota: without this, a script minting keys with throwaway emails multiplies
# the per-key caps arbitrarily. In-process state — fine for one instance.
_form_hits: dict[str, list[float]] = {}


def _client_ip(request: Request) -> str:
    # Railway terminates TLS at its proxy; the client is the first hop of
    # X-Forwarded-For. Fall back to the socket peer for dev / direct access.
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def _throttled(request: Request) -> bool:
    """True if this IP has exceeded the hourly form budget (and record the hit)."""
    now = time.monotonic()
    ip = _client_ip(request)
    hits = [t for t in _form_hits.get(ip, []) if now - t < 3600]
    if len(hits) >= settings.portal_ip_hourly:
        _form_hits[ip] = hits
        return True
    hits.append(now)
    _form_hits[ip] = hits
    if len(_form_hits) > 10_000:  # bound memory under address-spread abuse
        _form_hits.clear()
    return False


def _too_many(title: str) -> HTMLResponse:
    body = """
  <h1>Too many requests</h1>
  <p class="tag">Please slow down.</p>
  <p>We've seen several submissions from your network in the last hour. Try again later —
  or if you're behind a shared corporate NAT, reply to any Aimnis email and we'll sort you out.</p>
"""
    return HTMLResponse(_page(title, body), status_code=429)


# --------------------------------------------------------------------------- #
# Shared rendering
# --------------------------------------------------------------------------- #
# Brand mark (assets/brand/aimnis-icon.svg, embedded so the portal stays
# single-process with no static-file mount): ripple-lens magnifier.
_FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512">
<defs>
<radialGradient id="bg" cx="38%" cy="32%" r="90%">
<stop offset="0%" stop-color="#131c29"/><stop offset="100%" stop-color="#0b0f14"/>
</radialGradient>
<linearGradient id="mark" x1="0%" y1="0%" x2="100%" y2="100%">
<stop offset="0%" stop-color="#2dd4bf"/><stop offset="100%" stop-color="#58a6ff"/>
</linearGradient>
</defs>
<rect width="512" height="512" rx="114" fill="url(#bg)"/>
<g fill="none" stroke="url(#mark)">
<circle cx="226" cy="226" r="120" stroke-width="30"/>
<circle cx="226" cy="226" r="74" stroke-width="14" opacity="0.6"
 stroke-linecap="round" stroke-dasharray="330 135" transform="rotate(-35 226 226)"/>
<circle cx="226" cy="226" r="38" stroke-width="12" opacity="0.85"
 stroke-linecap="round" stroke-dasharray="175 65" transform="rotate(115 226 226)"/>
</g>
<circle cx="226" cy="226" r="14" fill="#2dd4bf"/>
<line x1="318" y1="318" x2="404" y2="404" stroke="url(#mark)"
 stroke-width="44" stroke-linecap="round"/>
</svg>"""

FAVICON_LINK = '<link rel="icon" href="/favicon.svg" type="image/svg+xml">'


@router.get("/favicon.svg")
@router.get("/favicon.ico")  # browsers probe this path unprompted; SVG works there too
async def favicon() -> Response:
    return Response(
        _FAVICON_SVG,
        media_type="image/svg+xml",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@router.get("/robots.txt")
async def robots() -> Response:
    base = settings.portal_base_url.rstrip("/")
    # /r/ redirects feed the aggregate click log — keep crawlers out of it.
    body = (
        "User-agent: *\n"
        "Disallow: /r/\n"
        "Disallow: /admin/\n"
        "Allow: /\n"
        f"Sitemap: {base}/sitemap.xml\n"
    )
    return Response(body, media_type="text/plain",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/llms.txt")
async def llms_txt() -> Response:
    """Plain-text site summary for LLM/agent crawlers (the llms.txt convention) —
    an agent-directory crawler probed for this on 2026-07-04, so something reads it."""
    base = settings.portal_base_url.rstrip("/")
    body = f"""# Aimnis

> Collaborative, cache-first web search for AI agents. Ask over MCP; semantically
> similar questions answered before (by anyone) return a distilled, source-cited
> answer instantly. New questions are searched live, distilled, cited, and pooled
> for the next agent. Search once, answer everyone.

MCP endpoint: {base}/mcp (streamable HTTP; Authorization: Bearer aim_YOUR_KEY,
or X-API-Key). Free eval key by email: {base}/register
Server card: {base}/.well-known/mcp/server-card.json

## Docs

- [Agent setup]({base}/setup): config snippets for OpenCode, OpenClaw, Hermes, Pi, Claude Code, REST
- [Live flywheel]({base}/flywheel): public dashboard — cache hit rate vs corpus size
- [Terms]({base}/terms): AI-generated answers, provenance, free personal-data removal
- [Source](https://github.com/aimnis/aimnis): AGPLv3 server, Apache-2.0 clients
"""
    return Response(body, media_type="text/plain; charset=utf-8",
                    headers={"Cache-Control": "public, max-age=86400"})


# RFC 8288 Link header advertised on the homepage so agents can discover the
# machine-readable surfaces without scraping HTML. Relative refs resolve against
# the request URI per the RFC. The rels are IANA-registered: api-catalog
# (RFC 9727), service-desc (the MCP server card describes the service),
# service-doc (human/agent-readable setup docs), describedby (llms.txt summary).
_DISCOVERY_LINKS = (
    '</.well-known/api-catalog>; rel="api-catalog", '
    '</.well-known/mcp/server-card.json>; rel="service-desc", '
    '</setup>; rel="service-doc", '
    '</llms.txt>; rel="describedby"'
)


@router.get("/.well-known/api-catalog")
async def api_catalog() -> Response:
    """RFC 9727 API catalog: a linkset enumerating the machine-usable endpoints
    (hosted MCP, REST gateway, public stats) and where each is described."""
    import json

    base = settings.portal_base_url.rstrip("/")
    linkset = {"linkset": [
        {
            "anchor": f"{base}/mcp",
            "service-desc": [{"href": f"{base}/.well-known/mcp/server-card.json",
                              "type": "application/json"}],
            "service-doc": [{"href": f"{base}/setup", "type": "text/html"}],
        },
        {
            "anchor": f"{base}/v1/search",
            "service-doc": [{"href": f"{base}/setup", "type": "text/html"}],
        },
        {
            "anchor": f"{base}/api/stats",
            "service-doc": [{"href": f"{base}/flywheel", "type": "text/html"}],
        },
    ]}
    return Response(json.dumps(linkset), media_type="application/linkset+json",
                    headers={"Cache-Control": "public, max-age=86400"})


@router.get("/sitemap.xml")
async def sitemap() -> Response:
    base = settings.portal_base_url.rstrip("/")
    urls = "".join(
        f"<url><loc>{base}{path}</loc></url>"
        for path in ("/", "/register", "/setup", "/flywheel", "/terms")
    )
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        f"{urls}</urlset>"
    )
    return Response(body, media_type="application/xml",
                    headers={"Cache-Control": "public, max-age=86400"})


_STYLE = """
  :root { color-scheme: dark; }
  * { box-sizing: border-box; }
  body { margin:0; background:#0b0f14; color:#e6edf3;
         font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }
  .wrap { max-width:760px; margin:0 auto; padding:40px 20px 64px; }
  h1 { font-size:30px; margin:0 0 6px; letter-spacing:-.01em; }
  h2 { font-size:19px; color:#c9d6e5; margin:34px 0 10px; }
  .tag { color:#8aa0bd; font-size:18px; margin:0 0 28px; }
  p { margin:0 0 14px; }
  a { color:#58a6ff; }
  ol, ul { margin:0 0 14px; padding-left:22px; }
  li { margin:6px 0; }
  .card { background:#111820; border:1px solid #20304a; border-radius:12px; padding:22px 24px; margin:22px 0; }
  .btn { display:inline-block; background:#238636; color:#fff; text-decoration:none;
         padding:11px 20px; border-radius:8px; font-weight:600; border:0; font-size:16px; cursor:pointer; }
  .btn.secondary { background:#21304a; }
  label { display:block; color:#c9d6e5; font-size:14px; margin:14px 0 6px; }
  input[type=email], input[type=text], input[type=password], textarea, select {
    width:100%; background:#0b0f14; border:1px solid #20304a; border-radius:8px;
    color:#e6edf3; padding:11px 12px; font:inherit; }
  input:focus, textarea:focus, select:focus { outline:none; border-color:#58a6ff; }
  input::placeholder, textarea::placeholder { color:#5c6f88; }
  input[type=checkbox] { accent-color:#238636; width:16px; height:16px; }
  textarea { min-height:78px; resize:vertical; }
  .row { display:flex; gap:10px; flex-wrap:wrap; }
  .row select { flex:0 1 200px; }
  .row input { flex:1 1 220px; }
  code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  pre { background:#0b0f14; border:1px solid #20304a; border-radius:8px; padding:14px;
        overflow-x:auto; font-size:13px; color:#c9d6e5; }
  .key { font-size:15px; word-break:break-all; color:#2dd4bf; }
  .err { color:#f85149; font-size:14px; margin:8px 0; }
  .muted { color:#5c6f88; font-size:13px; }
  .steps { color:#8aa0bd; }
  footer { color:#5c6f88; font-size:13px; margin-top:40px; border-top:1px solid #182234; padding-top:16px; }
  nav a { margin-right:16px; }
  .wrap.wide { max-width:920px; }
  .hero { display:grid; grid-template-columns:1.1fr 1fr; gap:30px; align-items:center; margin:6px 0 8px; }
  @media (max-width:720px) { .hero { grid-template-columns:1fr; } }
  .proof { background:#111820; border:1px solid #20304a; border-radius:12px; padding:18px 20px; }
  .proof .pill { color:#2dd4bf; font-size:12px; text-transform:uppercase; letter-spacing:.05em; }
  .proof .pill::before { content:"●"; margin-right:6px; }
  .proof .q { color:#e6edf3; font-weight:600; margin:10px 0 8px; }
  .proof .a { color:#8aa0bd; font-size:14px; margin:0 0 10px; }
  .proof .src { font-size:13px; margin:0; }
  .proof .ms { color:#5c6f88; font-size:12px; margin:8px 0 0; }
  .proofline { color:#8aa0bd; font-size:14px; margin:10px 2px 0; }
  .benefits { display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:12px; margin:0 0 14px; }
  .benefit { background:#111820; border:1px solid #20304a; border-radius:10px; padding:14px 16px; }
  .benefit b { display:block; margin-bottom:4px; }
  .benefit p { color:#8aa0bd; font-size:14px; margin:0; }
"""


def _page(title: str, body: str, *, wide: bool = False) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
{FAVICON_LINK}
<style>{_STYLE}</style></head>
<body><div class="wrap{' wide' if wide else ''}">{body}
<footer><nav><a href="/">Home</a><a href="/register">Get a key</a>
<a href="/setup">Agent setup</a><a href="/flywheel">Live flywheel</a><a href="/terms">Terms</a></nav>
<p class="muted">Aimnis is an evaluation preview — best-effort, no SLA. Answers are AI-generated;
verify time-sensitive facts. Don't send secrets or personal data in queries.</p></footer>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# Landing
# --------------------------------------------------------------------------- #
@router.get("/", response_class=HTMLResponse)
async def landing() -> HTMLResponse:
    # Live aggregate proof for the hero. Best-effort: the landing page must render
    # even if the DB is unreachable. Aggregates only — raw pool query text is never
    # published (a scrubber miss could leak a secret), so the Q→A sample below is a
    # hand-picked REAL pool entry, hardcoded rather than fetched.
    proofline = '<a href="/flywheel">Watch the pool grow live →</a>'
    try:
        s = await stats.gather(await db.get_pool())
        if s.corpus_servable:
            parts = [f"<b>{s.corpus_servable}</b> answers pooled"]
            if s.lookups_total:
                parts.append(f"<b>{s.hit_rate:.0%}</b> of questions answered instantly")
            proofline = (" · ".join(parts)
                         + ' · <a href="/flywheel">watch it live →</a>')
    except Exception:  # noqa: BLE001 — proof numbers are decoration, never a 500
        pass

    body = f"""
  <div class="hero">
    <div>
      <h1>Aimnis</h1>
      <p class="tag">Collaborative search for agents. Search once, answer everyone.</p>
      <p>Your coding agent keeps re-searching the same things — the new library version, the
      same error message, the same API change — and you wait and pay every time. <b>Aimnis
      gives your agent a shared memory of the web.</b> Anything that anyone's agent has asked
      before comes back instantly, with sources.</p>
      <p style="margin-top:18px">
        <a class="btn" href="/register">Get an eval API key</a>
        <a class="btn secondary" href="/setup">Agent setup</a>
      </p>
    </div>
    <div>
      <div class="proof">
        <p class="pill" style="margin:0">answered from the pool</p>
        <p class="q">pgvector: HNSW or IVFFlat for a table under 1M rows?</p>
        <p class="a">Use HNSW as the default — it works on empty tables, handles writes
        without rebuilds, and gives high recall with modest tuning
        (<code>hnsw.ef_search ≈ 80–120</code>)…</p>
        <p class="src">[1] github.com/pgvector &nbsp;·&nbsp; [2] supabase.com/blog</p>
        <p class="ms">a real pooled answer — searched once, instant for every agent since</p>
      </div>
      <p class="proofline">{proofline}</p>
    </div>
  </div>

  <h2>What you get</h2>
  <div class="benefits">
    <div class="benefit"><b>⚡ Faster sessions</b>
      <p>Known questions come back in milliseconds, not a full web search. Your agent spends
      its time coding.</p></div>
    <div class="benefit"><b>💰 Lower cost</b>
      <p>Live search runs only on genuinely new questions. Nobody pays for the same answer
      twice.</p></div>
    <div class="benefit"><b>🔍 Answers you can check</b>
      <p>Every answer shows its sources and its age — you can tell where it came from and how
      fresh it is.</p></div>
    <div class="benefit"><b>📈 Better every day</b>
      <p>Every question anyone asks becomes an instant answer for you — and yours help
      everyone else.</p></div>
  </div>

  <div class="card">
    <p style="margin:0 0 14px"><b>Try it free.</b> Eval key by email, no card. Your agent
    connects with just a URL and the key — nothing to install, set up in two minutes.</p>
    <a class="btn" href="/register">Get an eval API key</a>
    <a class="btn secondary" href="/setup">Agent setup</a>
    <a class="btn secondary" href="/flywheel">See the live flywheel</a>
  </div>

  <p class="muted">Skeptical? Good. The make-or-break number — how often the shared memory
  answers without a live search — is published live on the
  <a href="/flywheel">flywheel dashboard</a>. Power users can attach their own search/model
  keys at registration for much higher daily limits.</p>

  <p class="muted">Works with any MCP-capable agent — setup guides for
  <a href="/setup#opencode">OpenCode</a>, <a href="/setup#openclaw">OpenClaw</a>,
  <a href="/setup#hermes">Hermes Agent</a>, <a href="/setup#pi">Pi</a>,
  <a href="/setup#claude-code">Claude Code</a>, or <a href="/setup#rest">plain REST</a>.</p>

  <p class="muted">By requesting a key you agree to the <a href="/terms">Terms of Use</a>.
  Operators reserve the right to revoke any key at any time.</p>
"""
    return HTMLResponse(_page("Aimnis · Collaborative Search for Agents", body, wide=True),
                        headers={"Link": _DISCOVERY_LINKS})


# --------------------------------------------------------------------------- #
# Setup — per-agent instructions
# --------------------------------------------------------------------------- #
def _snip(text: str) -> str:
    """A copy-pasteable <pre> block (escaped verbatim snippet)."""
    return f"<pre>{html.escape(text)}</pre>"


@router.get("/setup", response_class=HTMLResponse)
async def setup() -> HTMLResponse:
    base = settings.portal_base_url.rstrip("/")
    mcp_url = f"{base}/mcp"

    opencode = _snip(f"""// opencode.json (project) — or the mcp block of your global config
{{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {{
    "aimnis": {{
      "type": "remote",
      "url": "{mcp_url}",
      "enabled": true,
      "headers": {{ "Authorization": "Bearer aim_YOUR_KEY" }}
    }}
  }}
}}""")

    openclaw = _snip(f"""// ~/.openclaw/openclaw.json — under mcp.servers
{{
  "mcp": {{
    "servers": {{
      "aimnis": {{
        "transport": "streamable-http",
        "url": "{mcp_url}",
        "headers": {{ "Authorization": "Bearer aim_YOUR_KEY" }}
      }}
    }}
  }}
}}""")
    openclaw_prefer = _snip("""// same file — make Aimnis the only web search:
{ "tools": { "web": { "search": { "enabled": false } } } }""")

    hermes = _snip(f"""# ~/.hermes/config.yaml  (put AIMNIS_KEY=aim_YOUR_KEY in ~/.hermes/.env)
mcp_servers:
  aimnis:
    url: "{mcp_url}"
    headers:
      Authorization: "Bearer ${{AIMNIS_KEY}}"
""")
    hermes_prefer = _snip("""# same file — hide the built-in web_search / web_extract:
agent:
  disabled_toolsets:
    - web""")

    pi_mcp = _snip("pi install npm:@zhafron/pi-mcp-tools") + _snip(f"""// ~/.pi/agent/settings.json (project: .pi/settings.json)
{{
  "mcp": {{
    "aimnis": {{
      "type": "remote",
      "url": "{mcp_url}"
    }}
  }}
}}""")
    pi_curl = _snip(f"""export AIMNIS_KEY=aim_YOUR_KEY
curl -s {base}/v1/search \\
  -H "Authorization: Bearer $AIMNIS_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "your question"}}'""")

    claude_code = _snip(f"""claude mcp add --transport http aimnis {mcp_url} \\
    --header "Authorization: Bearer aim_YOUR_KEY\"""")
    claude_prefer = _snip("""// .claude/settings.json — prefer Aimnis over the built-in web search:
{ "permissions": { "deny": ["WebSearch"] } }""")

    rest = _snip(f"""curl -s {base}/v1/search \\
  -H "Authorization: Bearer aim_YOUR_KEY" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "how do I undo the last git commit but keep changes staged"}}'""")

    body = f"""
  <h1>Set up your agent</h1>
  <p class="tag">Aimnis is a hosted MCP server — most agents connect with just a URL and your key.</p>

  <div class="card">
    <p style="margin:0"><b>Endpoint</b>: <code>{html.escape(mcp_url)}</code> (MCP, streamable HTTP)<br>
    <b>Auth</b>: <code>Authorization: Bearer aim_…</code> or <code>X-API-Key: aim_…</code><br>
    <b>Tools</b>: <code>search</code> (cached/live web answers, cited) · <code>stats</code> (pool metrics)<br>
    No key yet? <a href="/register">Get one</a> — it takes ten seconds.</p>
  </div>

  <h2 id="opencode">OpenCode</h2>
  <p>Add a remote MCP server to your <code>opencode.json</code>:</p>
  {opencode}
  <p class="muted">OpenCode's built-in <code>websearch</code> is off unless you use the OpenCode
  provider or set <code>OPENCODE_ENABLE_EXA</code> — leave those off and the model reaches for
  Aimnis naturally. Tools appear as <code>aimnis_search</code> / <code>aimnis_stats</code>.</p>

  <h2 id="openclaw">OpenClaw</h2>
  <p>Register Aimnis under <code>mcp.servers</code> in <code>~/.openclaw/openclaw.json</code>
  (or: <code>openclaw mcp add aimnis --transport streamable-http --url {html.escape(mcp_url)}</code>):</p>
  {openclaw}
  <p>To make Aimnis the only web search, disable the managed one:</p>
  {openclaw_prefer}
  <p class="muted">Verify connectivity with <code>openclaw mcp doctor --probe</code>.</p>

  <h2 id="hermes">Hermes Agent</h2>
  <p>Add Aimnis under <code>mcp_servers</code> in <code>~/.hermes/config.yaml</code>; Hermes
  resolves <code>${{VAR}}</code> placeholders from <code>~/.hermes/.env</code>:</p>
  {hermes}
  <p>To route all search through Aimnis, disable the built-in web toolset:</p>
  {hermes_prefer}
  <p class="muted">Tools appear as <code>mcp_aimnis_search</code> / <code>mcp_aimnis_stats</code>.
  Reload a live session with <code>/reload-mcp</code>.</p>

  <h2 id="pi">Pi</h2>
  <p>Pi has no native MCP client; the <code>pi-mcp-tools</code> extension bridges it:</p>
  {pi_mcp}
  <p class="muted">If your pi-mcp-tools version doesn't support auth headers on remote servers
  yet, use the REST route below instead.</p>
  <p>Or skip MCP entirely — Pi's idiomatic path is a CLI the agent calls via bash. Put this in
  your project's <code>AGENTS.md</code> ("use this command for web searches") :</p>
  {pi_curl}

  <h2 id="claude-code">Claude Code</h2>
  {claude_code}
  {claude_prefer}

  <h2 id="rest">Any other agent (REST)</h2>
  <p>Anything that can make an HTTP request can use Aimnis — <code>POST /v1/search</code>
  returns structured JSON plus a ready-to-render <code>formatted</code> field:</p>
  {rest}

  <h2>Good to know</h2>
  <ul>
    <li><b>Metering</b>: only actual searches count (MCP handshakes are free). Default limits:
        {settings.client_default_rpm}/min, {settings.client_default_rpd}/day per key.</li>
    <li><b>Cache hits are instant</b>; misses run a live search + distill and are slower
        (seconds). Every answer is AI-generated and cites its sources.</li>
    <li><b>Need more than the default daily limit?</b> Re-register with your own
        OpenRouter / search keys attached (<a href="/register">BYOK</a>) — your misses then run
        on your quota and your cap rises to {settings.byok_rpd:,}/day.</li>
    <li><b>Don't send secrets</b> in queries — scrubbing is best-effort
        (<a href="/terms">Terms</a>).</li>
  </ul>
"""
    return HTMLResponse(_page("Aimnis · Agent setup", body))


# --------------------------------------------------------------------------- #
# Terms
# --------------------------------------------------------------------------- #
@router.get("/terms", response_class=HTMLResponse)
async def terms() -> HTMLResponse:
    body = """
  <h1>Terms of Use</h1>
  <p class="muted">Evaluation preview. These terms govern use of the hosted Aimnis service and
  evaluation API keys.</p>

  <h2>The service</h2>
  <ul>
    <li>Aimnis is provided <b>as-is, best-effort, with no SLA or warranty</b> during the
        evaluation preview. Availability, limits, and features may change at any time.</li>
    <li>Access is metered per key (per-minute and per-day request caps).
        <b>Operators reserve the right to revoke any key at any time</b>, with or without notice —
        e.g. for abuse, to protect shared capacity, or when the preview ends.</li>
  </ul>

  <h2>Answers</h2>
  <ul>
    <li>Answers are <b>AI-generated</b> and may be wrong or out of date. Verify anything
        time-sensitive or consequential against the cited sources.</li>
    <li>Answers are distilled from third-party web sources, cited in each response. Aimnis does not
        warrant the accuracy, completeness, or licensing of upstream content.</li>
  </ul>

  <h2>What you send us</h2>
  <ul>
    <li><b>Do not send secrets, credentials, or personal data in queries.</b> Aimnis runs a
        best-effort secret/PII scrubber before processing, but it is not a guarantee — treat every
        query as if it may be retained.</li>
    <li>Distilled, scrubbed question/answer pairs may be added to the <b>shared knowledge pool</b>
        so other users benefit. Raw queries are not published; the pool stores distilled answers
        with provenance.</li>
    <li>Upstream free-tier model providers may log prompts on their side; pooled content was never
        private upstream.</li>
  </ul>

  <h2>Telemetry &amp; privacy</h2>
  <ul>
    <li>Citation clicks are logged in <b>aggregate only</b> — which pooled entry and source were
        followed, plus destination host and time. <b>No IP, no user-agent, no per-user tracking.</b></li>
    <li>We store the <b>email</b> you register with, to deliver your key and capacity notices.
        <b>Removal of your own personal data is free, always</b> — email the operator to have your
        email and key deleted.</li>
  </ul>

  <h2>Bring-your-own-keys (BYOK)</h2>
  <ul>
    <li>You may optionally attach your own OpenRouter and/or search-provider API keys. They are
        stored <b>encrypted at rest</b>, used <b>exclusively to serve your own requests</b> —
        never other users' — and <b>deleted</b> when your Aimnis key is revoked or rotated.
        You can remove them any time by re-registering without keys, or by asking us.</li>
    <li>You must own the attached keys, and your provider plans must permit using results for
        AI answering and contributing distilled answers to a shared pool. Your relationship
        with those providers (quotas, billing, their terms) remains your own.</li>
    <li>Answers produced under your keys are provenance-tagged in the pool, so they can be
        removed if a licensing problem ever surfaces.</li>
  </ul>

  <h2>Acceptable use</h2>
  <ul>
    <li>Don't attempt to exceed or evade rate limits, register keys in bulk, or resell access.</li>
    <li>Don't use Aimnis to generate or distribute unlawful content, or to poison the pool.</li>
    <li>Don't attach API keys you don't own or aren't licensed to use this way.</li>
  </ul>
"""
    return HTMLResponse(_page("Aimnis · Terms of Use", body))


# --------------------------------------------------------------------------- #
# Register
# --------------------------------------------------------------------------- #
def _byok_fields() -> str:
    """The optional bring-your-own-keys section of the registration form. Only
    rendered when BYOK is enabled server-side (encryption secret configured)."""
    if not apikeys.byok_enabled():
        return ""
    return f"""
    <details style="margin-top:18px">
      <summary style="cursor:pointer;color:#2dd4bf;font-weight:600">
        Bring your own keys — {settings.byok_rpd:,} requests/day instead of {settings.client_default_rpd}</summary>
      <p class="muted" style="margin-top:10px">Optional. Attach your own upstream keys and your
      cache <i>misses</i> run on <b>your</b> quota instead of the shared pool's — so you get far
      higher limits ("your keys, your limits"). Keys are stored encrypted, used <b>only for your
      own requests</b>, and deleted when your key is revoked or rotated. Cache hits stay free
      and don't touch your keys.</p>
      <label for="openrouter_key">OpenRouter API key (for answer distillation)</label>
      <input type="password" id="openrouter_key" name="openrouter_key"
             placeholder="sk-or-v1-…" autocomplete="off">
      <label for="search_provider">Search provider + key (for live search on misses)</label>
      <div class="row">
        <select id="search_provider" name="search_provider">
          <option value="">— none —</option>
          <option value="brave">Brave Search API</option>
          <option value="tavily">Tavily</option>
          <option value="exa">Exa</option>
        </select>
        <input type="password" name="search_key" placeholder="search API key"
               autocomplete="off">
      </div>
      <label style="display:flex;gap:8px;align-items:flex-start;margin-top:14px;font-size:13px;color:#8aa0bd">
        <input type="checkbox" name="byok_ack" value="yes" style="margin-top:3px">
        <span>I own these keys and my provider plans permit using results for AI answering and
        contributing distilled answers to a shared pool.</span>
      </label>
    </details>"""


def _register_form(error: str = "", email: str = "") -> str:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return f"""
  <h1>Get an eval API key</h1>
  <p class="tag">Self-serve — your key arrives by email. One active key per email.</p>
  {err}
  <form method="post" action="/register" class="card">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required value="{html.escape(email)}"
           placeholder="you@example.com" autocomplete="email">
    <label for="use_case">What are you evaluating it for? (optional)</label>
    <textarea id="use_case" name="use_case" placeholder="e.g. WebSearch for Claude Code on a local model"></textarea>
    {_byok_fields()}
    <p style="margin:18px 0 0"><button class="btn" type="submit">Create my key</button></p>
  </form>
  <p class="muted">By creating a key you agree to the <a href="/terms">Terms of Use</a>.
  Your key is metered and revocable at any time. Re-register with the same email to rotate
  your key or change attached keys.</p>
"""


def _at_capacity(error: str = "", email: str = "") -> str:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return f"""
  <h1>Aimnis is at capacity</h1>
  <p class="tag">Evaluation keys are paused right now.</p>
  <p>We cap the number of active evaluators so the shared pool stays fast and within its
  upstream quota. Leave your email and we'll notify you the moment capacity opens up.</p>
  {err}
  <form method="post" action="/waitlist" class="card">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required value="{html.escape(email)}"
           placeholder="you@example.com" autocomplete="email">
    <p style="margin:18px 0 0"><button class="btn" type="submit">Notify me</button></p>
  </form>
"""


@router.get("/register", response_class=HTMLResponse)
async def register_form() -> HTMLResponse:
    pool = await db.get_pool()
    if not await flags.registration_open(pool):
        return HTMLResponse(_page("Aimnis · At capacity", _at_capacity()))
    return HTMLResponse(_page("Aimnis · Get an eval key", _register_form()))


def _key_sent_page(issued: apikeys.IssuedKey) -> str:
    """Production success page: the key travels ONLY by email (possession of the
    inbox is the gate — on-screen delivery would make throwaway-email key farming
    trivial). Shows the prefix so the user can match it to the email."""
    return f"""
  <h1>Check your email</h1>
  <div class="card">
    <p style="margin:0">Your eval API key (<code>{html.escape(issued.prefix)}…</code>) was sent to
    <b>{html.escape(issued.email or "you")}</b>. It's only ever delivered by email — if it
    doesn't arrive within a few minutes, check spam, then re-register to rotate a fresh key.</p>
  </div>
  <h2>While you wait</h2>
  <p><a class="btn" href="/setup">Per-agent setup — OpenCode, OpenClaw, Hermes, Pi, Claude Code</a></p>
  <p class="muted">Limits: {issued.rpm_limit} requests/min, {issued.rpd_limit:,} requests/day.{
    " Your own keys are attached — misses run on your quota; keys are stored encrypted,"
    " used only for your requests, and wiped on revoke/rotate." if issued.byok else ""}
  The key is revocable at any time — see the <a href="/terms">Terms</a>.</p>
"""


def _email_failed_page(email_addr: str) -> str:
    return f"""
  <h1>We couldn't send your key</h1>
  <p class="tag">Nothing was issued.</p>
  <p>Delivering your key to <b>{html.escape(email_addr)}</b> failed, so the key was discarded
  (keys are only ever delivered by email). Please try again in a few minutes.</p>
  <p><a class="btn" href="/register">Try again</a></p>
"""


def _key_issued_page(request: Request, issued: apikeys.IssuedKey) -> str:
    """Dev / self-host success page (no email provider configured): the key is
    shown once on-screen. Production always takes the email-only path."""
    gateway_url = str(request.base_url).rstrip("/")
    return f"""
  <h1>Your eval API key</h1>
  <div class="card">
    <p style="margin:0 0 8px"><b>Copy it now — it's shown once and never again.</b></p>
    <p class="key">{html.escape(issued.key)}</p>
  </div>

  <h2>Point your agent at Aimnis</h2>
  <p>Aimnis is a hosted MCP server — nothing to install. Add it to your agent:</p>
  <pre>URL:     {html.escape(gateway_url)}/mcp        (MCP, streamable HTTP)
Header:  Authorization: Bearer {html.escape(issued.key)}</pre>
  <p><a class="btn" href="/setup">Per-agent setup — OpenCode, OpenClaw, Hermes, Pi, Claude Code</a></p>
  <p>Or call the REST endpoint directly:</p>
  <pre>curl -s {html.escape(gateway_url)}/v1/search \\
  -H "Authorization: Bearer {html.escape(issued.key)}" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "how do I undo the last git commit but keep changes staged"}}'</pre>

  <p class="muted">Limits: {issued.rpm_limit} requests/min, {issued.rpd_limit:,} requests/day.{
    " Your own keys are attached — misses run on your quota; keys are stored encrypted,"
    " used only for your requests, and wiped on revoke/rotate." if issued.byok else ""}
  Cache hits are instant; only misses count against the daily upstream budget. The key is
  revocable at any time — see the <a href="/terms">Terms</a>.</p>
"""


def _key_email_html(issued: apikeys.IssuedKey, gateway_url: str) -> str:
    setup_url = f"{settings.portal_base_url.rstrip('/')}/setup"
    return f"""<div style="font-family:system-ui,sans-serif;line-height:1.6;color:#111">
  <h2>Your Aimnis eval API key</h2>
  <p>Here's your evaluation key. Keep it secret; it's metered and revocable at any time.</p>
  <p style="font-family:monospace;font-size:15px;word-break:break-all;background:#f4f4f4;padding:12px;border-radius:6px">{html.escape(issued.key)}</p>
  <p>Aimnis is a hosted MCP server — nothing to install. Add it to your coding agent:</p>
  <pre style="background:#f4f4f4;padding:12px;border-radius:6px;overflow-x:auto">URL:     {html.escape(gateway_url)}/mcp
Header:  Authorization: Bearer {html.escape(issued.key)}</pre>
  <p>Per-agent instructions (OpenCode, OpenClaw, Hermes, Pi, Claude Code, REST):
  <a href="{html.escape(setup_url)}">{html.escape(setup_url)}</a></p>
  <p>Limits: {issued.rpm_limit} requests/min, {issued.rpd_limit} requests/day. Cache hits are free.</p>
  <p style="color:#666;font-size:13px">Answers are AI-generated — verify time-sensitive facts.
  Don't send secrets or personal data in queries. Removal of your data is free — just reply to ask.</p>
</div>"""


@router.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    email: str = Form(...),
    use_case: str = Form(default=""),
    openrouter_key: str = Form(default=""),
    search_provider: str = Form(default=""),
    search_key: str = Form(default=""),
    byok_ack: str = Form(default=""),
) -> HTMLResponse:
    if _throttled(request):
        return _too_many("Aimnis · Too many requests")

    pool = await db.get_pool()
    email = email.strip()

    if not await flags.registration_open(pool):
        # Paused between form render and submit — send them to the waitlist instead.
        return HTMLResponse(_page("Aimnis · At capacity", _at_capacity(email=email)))

    def _bad(msg: str) -> HTMLResponse:
        return HTMLResponse(
            _page("Aimnis · Get an eval key", _register_form(msg, email)), status_code=400
        )

    if not _valid_email(email):
        return _bad("Please enter a valid email address.")

    # Optional BYOK: validate only if any key material was submitted.
    openrouter_key = openrouter_key.strip()
    search_provider = search_provider.strip()
    search_key = search_key.strip()
    client_keys = None
    if openrouter_key or search_key:
        if not apikeys.byok_enabled():
            return _bad("Bring-your-own-keys is not available right now.")
        if search_key and search_provider not in apikeys.SEARCH_PROVIDERS:
            return _bad("Pick a search provider for your search key.")
        if byok_ack != "yes":
            return _bad("Please confirm the bring-your-own-keys terms checkbox.")
        client_keys = apikeys.ClientKeys(
            openrouter_api_key=openrouter_key or None,
            search_provider=search_provider if search_key else None,
            search_api_key=search_key or None,
        )

    label = use_case.strip()[:500] or None
    issued = await apikeys.issue(pool, email=email, label=label, client_keys=client_keys)

    # No email provider configured (dev / self-host): show the key once on-screen.
    if not settings.resend_api_key:
        return HTMLResponse(_page("Aimnis · Your eval key", _key_issued_page(request, issued)))

    # Production: the key is delivered ONLY by email — a working inbox is the
    # anti-farming gate. If the send fails, revoke so no orphan key floats around.
    gateway_url = str(request.base_url).rstrip("/")
    sent = await email_mod.send_email(
        email, "Your Aimnis eval API key", _key_email_html(issued, gateway_url)
    )
    if not sent:
        await apikeys.revoke(pool, email=email)
        return HTMLResponse(
            _page("Aimnis · Delivery failed", _email_failed_page(email)), status_code=502
        )
    return HTMLResponse(_page("Aimnis · Check your email", _key_sent_page(issued)))


# --------------------------------------------------------------------------- #
# Waitlist
# --------------------------------------------------------------------------- #
@router.post("/waitlist", response_class=HTMLResponse)
async def waitlist_submit(request: Request, email: str = Form(...)) -> HTMLResponse:
    if _throttled(request):
        return _too_many("Aimnis · Too many requests")
    pool = await db.get_pool()
    email = email.strip()
    if not _valid_email(email):
        return HTMLResponse(
            _page("Aimnis · At capacity",
                  _at_capacity("Please enter a valid email address.", email)),
            status_code=400,
        )
    await pool.execute(
        "INSERT INTO waitlist (email) VALUES ($1) "
        "ON CONFLICT (lower(email)) DO NOTHING",
        email,
    )
    await email_mod.send_email(
        email, "You're on the Aimnis waitlist",
        "<p>Thanks — you're on the Aimnis evaluation waitlist. "
        "We'll email you the moment capacity opens up.</p>",
    )
    body = """
  <h1>You're on the list</h1>
  <p>We'll email you the moment evaluation capacity opens up. Thanks for your patience.</p>
  <p><a class="btn secondary" href="/">Back home</a></p>
"""
    return HTMLResponse(_page("Aimnis · Waitlisted", body))


# --------------------------------------------------------------------------- #
# Directory well-knowns
# --------------------------------------------------------------------------- #
@router.get("/.well-known/glama.json")
async def glama_wellknown() -> JSONResponse:
    """Claim file for the Glama MCP directory (glama.ai) — proves the hosted
    connector's maintainer. The contact is the send-from mailbox (monitored)."""
    addr = re.sub(r"^.*<|>.*$", "", settings.email_from).strip()
    return JSONResponse({
        "$schema": "https://glama.ai/mcp/schemas/connector.json",
        "maintainers": [{"email": addr}],
    })


@router.get("/.well-known/mcp/server-card.json")
async def mcp_server_card() -> JSONResponse:
    """Static MCP server card: capability metadata for directory scanners
    (e.g. Smithery) that can't get past /mcp's bearer-key auth wall — we don't
    implement OAuth, so their authenticated scan can't run. Tool names/schemas
    must mirror mcp_server.py."""
    # Smithery session-config: prompt the user for their key and forward it
    # as the X-API-Key header (accepted by our /mcp edge alongside Bearer).
    api_key_field: dict = {
        "type": "string",
        "title": "Aimnis API key",
        "description": "Free eval key (aim_…) from https://aimnis.com/register",
        "x-from": {"header": "x-api-key"},
    }
    if settings.demo_api_key:
        # Deliberately-public try-out key (its own DB client, standard tight
        # caps, revocable by prefix). Pre-fills the directory config form so
        # visitors can try the server without registering first.
        api_key_field["default"] = settings.demo_api_key
        api_key_field["description"] += " — or keep the pre-filled shared demo key"
    return JSONResponse({
        "serverInfo": {"name": "Aimnis", "version": "0.1.0"},
        "authentication": {"required": True, "schemes": ["bearer"]},
        "configSchema": {
            "type": "object",
            "required": ["apiKey"],
            "properties": {"apiKey": api_key_field},
        },
        "tools": [
            {
                "name": "search",
                "description": (
                    "Search the web via Aimnis. Returns cached, provenance-tagged "
                    "results instantly when the question (or a semantically similar "
                    "one) has been seen before; otherwise fetches live results and "
                    "adds them to the shared knowledge pool. Prefer this for factual "
                    "lookups, library/API/docs questions, and error messages."
                ),
                "inputSchema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
            {
                "name": "stats",
                "description": (
                    "Report Aimnis flywheel statistics: knowledge-pool (cache) size, "
                    "cache hit rate (all-time and recent), and the most-reused queries."
                ),
                "inputSchema": {"type": "object", "properties": {}},
            },
        ],
        "resources": [],
        "prompts": [],
    })


# --------------------------------------------------------------------------- #
# Admin: pause/resume registration, list registered clients
# --------------------------------------------------------------------------- #
def _require_admin(x_admin_key: str | None) -> None:
    if not settings.admin_api_key:
        raise HTTPException(status_code=404, detail="not found")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="invalid admin key")


@router.get("/admin/clients")
async def admin_list_clients(
    active: bool = False,
    x_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    """Registered clients (email, key prefix, status, limits, usage). Full keys are
    never stored, so they can never be listed — the prefix is the operator handle."""
    _require_admin(x_admin_key)
    pool = await db.get_pool()
    clients = await apikeys.list_clients(pool, active_only=active)
    usage = {
        str(r["client_id"]): (r["today"], r["total"])
        for r in await pool.fetch(
            "SELECT client_id, count(*) FILTER (WHERE created_at >= now() - interval '24 hours')"
            " AS today, count(*) AS total FROM api_request GROUP BY client_id"
        )
    }
    for c in clients:
        c["id"] = str(c["id"])
        c["created_at"] = c["created_at"].isoformat()
        c["revoked_at"] = c["revoked_at"].isoformat() if c["revoked_at"] else None
        c["requests_24h"], c["requests_total"] = usage.get(c["id"], (0, 0))
    return JSONResponse({"count": len(clients), "clients": clients})


@router.post("/admin/registration")
async def admin_set_registration(
    open: bool = Form(...),
    x_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    """Flip the registration_open flag live (no redeploy). Requires AIMNIS_ADMIN_API_KEY;
    if that's unset the endpoint is disabled (fail-closed 404)."""
    _require_admin(x_admin_key)
    pool = await db.get_pool()
    await flags.set_flag(pool, flags.REGISTRATION_OPEN, open)
    return JSONResponse({"registration_open": open})
