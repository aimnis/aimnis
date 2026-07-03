"""Self-serve eval portal (aimnis.com).

Public marketing landing + self-serve eval-key registration, a terms page, and a
waitlist for when intake is paused. Server-rendered HTML (inline CSS, no external
deps — same constraints as the flywheel dashboard).

Routes:
    GET  /                     landing — what Aimnis is + why
    GET  /terms                terms of use
    GET  /register             registration form (or "at capacity" + waitlist if paused)
    POST /register             issue a key (open) → show it once + email it; else waitlist
    POST /waitlist             capture an email for capacity notifications
    POST /admin/registration   operator pause/resume toggle (X-Admin-Key)

Issuance is INSTANT when `registration_open`. Each issued key is metered (per-key
rate + daily caps, see apikeys/gateway) and revocable at any time. Operators reserve
the right to revoke any key at any time — stated in the terms and on the success page.
"""

from __future__ import annotations

import hmac
import html
import re

from fastapi import APIRouter, Form, Header, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse

from . import apikeys, db, email as email_mod, flags
from .config import settings

router = APIRouter()

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _valid_email(e: str) -> bool:
    return bool(e) and len(e) <= 254 and _EMAIL_RE.match(e) is not None


# --------------------------------------------------------------------------- #
# Shared rendering
# --------------------------------------------------------------------------- #
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
  input[type=email], input[type=text], textarea {
    width:100%; background:#0b0f14; border:1px solid #20304a; border-radius:8px;
    color:#e6edf3; padding:11px 12px; font:inherit; }
  textarea { min-height:78px; resize:vertical; }
  code, pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
  pre { background:#0b0f14; border:1px solid #20304a; border-radius:8px; padding:14px;
        overflow-x:auto; font-size:13px; color:#c9d6e5; }
  .key { font-size:15px; word-break:break-all; color:#2dd4bf; }
  .err { color:#f85149; font-size:14px; margin:8px 0; }
  .muted { color:#5c6f88; font-size:13px; }
  .steps { color:#8aa0bd; }
  footer { color:#5c6f88; font-size:13px; margin-top:40px; border-top:1px solid #182234; padding-top:16px; }
  nav a { margin-right:16px; }
"""


def _page(title: str, body: str) -> str:
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>{_STYLE}</style></head>
<body><div class="wrap">{body}
<footer><nav><a href="/">Home</a><a href="/register">Get a key</a>
<a href="/flywheel">Live flywheel</a><a href="/terms">Terms</a></nav>
<p class="muted">Aimnis is an evaluation preview — best-effort, no SLA. Answers are AI-generated;
verify time-sensitive facts. Don't send secrets or personal data in queries.</p></footer>
</div></body></html>"""


# --------------------------------------------------------------------------- #
# Landing
# --------------------------------------------------------------------------- #
@router.get("/", response_class=HTMLResponse)
async def landing() -> HTMLResponse:
    body = """
  <h1>Aimnis</h1>
  <p class="tag">Collaborative search for agents. Search once, answer everyone.</p>

  <p>Aimnis is a <b>cache-first knowledge gateway</b> for AI agents. Every question an agent
  asks is answered from a <b>shared, continuously-growing pool</b> of distilled, cited answers.
  When a question is new, Aimnis runs a live web search, distills a cited answer, and adds it to
  the pool — so the <i>next</i> agent that asks anything similar gets it instantly, for free.</p>

  <h2>How it works</h2>
  <ol>
    <li>Your agent asks a question through the Aimnis MCP tool or REST endpoint.</li>
    <li><b>Cache hit</b> (exact or semantic) → an instant, cited answer from the pool.</li>
    <li><b>Miss</b> → live search → distilled, source-cited answer → added to the pool.</li>
    <li>The pool compounds: the more it's used, the more it already knows.</li>
  </ol>

  <h2>Why it's different</h2>
  <ul>
    <li><b>A continuous-learning layer</b>, not a per-vendor search bolt-on — one shared corpus
        across agents that captures what happened after any model's training cutoff.</li>
    <li><b>Cheaper and faster</b> than hitting a live web-search API on every query — you pay the
        live cost once, then serve it from cache to everyone.</li>
    <li><b>Honest provenance</b>: every answer is labeled AI-generated and carries its sources and
        freshness, so your agent can decide when to escalate to a live search.</li>
  </ul>

  <div class="card">
    <p style="margin:0 0 14px"><b>Try it.</b> Grab an evaluation API key and point your coding
    agent at the hosted gateway.</p>
    <a class="btn" href="/register">Get an eval API key</a>
    <a class="btn secondary" href="/flywheel">See the live flywheel</a>
  </div>

  <p class="muted">By requesting a key you agree to the <a href="/terms">Terms of Use</a>.
  Operators reserve the right to revoke any key at any time.</p>
"""
    return HTMLResponse(_page("Aimnis · Collaborative Search for Agents", body))


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

  <h2>Acceptable use</h2>
  <ul>
    <li>Don't attempt to exceed or evade rate limits, register keys in bulk, or resell access.</li>
    <li>Don't use Aimnis to generate or distribute unlawful content, or to poison the pool.</li>
  </ul>
"""
    return HTMLResponse(_page("Aimnis · Terms of Use", body))


# --------------------------------------------------------------------------- #
# Register
# --------------------------------------------------------------------------- #
def _register_form(error: str = "", email: str = "") -> str:
    err = f'<p class="err">{html.escape(error)}</p>' if error else ""
    return f"""
  <h1>Get an eval API key</h1>
  <p class="tag">Instant, self-serve. One active key per email.</p>
  {err}
  <form method="post" action="/register" class="card">
    <label for="email">Email</label>
    <input type="email" id="email" name="email" required value="{html.escape(email)}"
           placeholder="you@example.com" autocomplete="email">
    <label for="use_case">What are you evaluating it for? (optional)</label>
    <textarea id="use_case" name="use_case" placeholder="e.g. WebSearch for Claude Code on a local model"></textarea>
    <p style="margin:18px 0 0"><button class="btn" type="submit">Create my key</button></p>
  </form>
  <p class="muted">By creating a key you agree to the <a href="/terms">Terms of Use</a>.
  Your key is metered and revocable at any time.</p>
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


def _key_issued_page(request: Request, issued: apikeys.IssuedKey) -> str:
    gateway_url = str(request.base_url).rstrip("/")
    return f"""
  <h1>Your eval API key</h1>
  <div class="card">
    <p style="margin:0 0 8px"><b>Copy it now — it's shown once and never again.</b></p>
    <p class="key">{html.escape(issued.key)}</p>
  </div>
  <p>We've also emailed it to {html.escape(issued.email or "you")}.</p>

  <h2>Point your agent at Aimnis</h2>
  <p>Run the local MCP server in remote mode (it forwards to the hosted gateway; your queries
  never touch a database directly):</p>
  <pre>AIMNIS_GATEWAY_URL={html.escape(gateway_url)} \\
AIMNIS_GATEWAY_CLIENT_API_KEY={html.escape(issued.key)} \\
    python -m aimnis.mcp_server</pre>
  <p>Register that command in your coding agent (Claude Code / OpenCode) as an MCP server.
  Or call the REST endpoint directly:</p>
  <pre>curl -s {html.escape(gateway_url)}/v1/search \\
  -H "Authorization: Bearer {html.escape(issued.key)}" \\
  -H "Content-Type: application/json" \\
  -d '{{"query": "how do I undo the last git commit but keep changes staged"}}'</pre>

  <p class="muted">Limits: {issued.rpm_limit} requests/min, {issued.rpd_limit} requests/day.
  Cache hits are instant; only misses count against the daily upstream budget. The key is
  revocable at any time — see the <a href="/terms">Terms</a>.</p>
"""


def _key_email_html(issued: apikeys.IssuedKey, gateway_url: str) -> str:
    return f"""<div style="font-family:system-ui,sans-serif;line-height:1.6;color:#111">
  <h2>Your Aimnis eval API key</h2>
  <p>Here's your evaluation key. Keep it secret; it's metered and revocable at any time.</p>
  <p style="font-family:monospace;font-size:15px;word-break:break-all;background:#f4f4f4;padding:12px;border-radius:6px">{html.escape(issued.key)}</p>
  <p>Point your coding agent at the hosted gateway:</p>
  <pre style="background:#f4f4f4;padding:12px;border-radius:6px;overflow-x:auto">AIMNIS_GATEWAY_URL={html.escape(gateway_url)}
AIMNIS_GATEWAY_CLIENT_API_KEY={html.escape(issued.key)}
python -m aimnis.mcp_server</pre>
  <p>Limits: {issued.rpm_limit} requests/min, {issued.rpd_limit} requests/day. Cache hits are free.</p>
  <p style="color:#666;font-size:13px">Answers are AI-generated — verify time-sensitive facts.
  Don't send secrets or personal data in queries. Removal of your data is free — just reply to ask.</p>
</div>"""


@router.post("/register", response_class=HTMLResponse)
async def register_submit(
    request: Request,
    email: str = Form(...),
    use_case: str = Form(default=""),
) -> HTMLResponse:
    pool = await db.get_pool()
    email = email.strip()

    if not await flags.registration_open(pool):
        # Paused between form render and submit — send them to the waitlist instead.
        return HTMLResponse(_page("Aimnis · At capacity", _at_capacity(email=email)))

    if not _valid_email(email):
        return HTMLResponse(
            _page("Aimnis · Get an eval key",
                  _register_form("Please enter a valid email address.", email)),
            status_code=400,
        )

    label = use_case.strip()[:500] or None
    issued = await apikeys.issue(pool, email=email, label=label)

    gateway_url = str(request.base_url).rstrip("/")
    await email_mod.send_email(
        email, "Your Aimnis eval API key", _key_email_html(issued, gateway_url)
    )
    return HTMLResponse(_page("Aimnis · Your eval key", _key_issued_page(request, issued)))


# --------------------------------------------------------------------------- #
# Waitlist
# --------------------------------------------------------------------------- #
@router.post("/waitlist", response_class=HTMLResponse)
async def waitlist_submit(email: str = Form(...)) -> HTMLResponse:
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
# Admin: pause/resume registration
# --------------------------------------------------------------------------- #
@router.post("/admin/registration")
async def admin_set_registration(
    open: bool = Form(...),
    x_admin_key: str | None = Header(default=None),
) -> JSONResponse:
    """Flip the registration_open flag live (no redeploy). Requires AIMNIS_ADMIN_API_KEY;
    if that's unset the endpoint is disabled (fail-closed 404)."""
    if not settings.admin_api_key:
        raise HTTPException(status_code=404, detail="not found")
    if not x_admin_key or not hmac.compare_digest(x_admin_key, settings.admin_api_key):
        raise HTTPException(status_code=401, detail="invalid admin key")
    pool = await db.get_pool()
    await flags.set_flag(pool, flags.REGISTRATION_OPEN, open)
    return JSONResponse({"registration_open": open})
