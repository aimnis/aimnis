"""Public flywheel dashboard — the Gate 1 pass/kill instrument, made visible.

Serves a self-contained HTML page (server-rendered inline SVG, no external deps)
plotting cache hit rate against cumulative unique queries, plus a JSON metrics
endpoint for build-in-public / programmatic access.

    uvicorn aimnis.api:app          # or: python -m aimnis.api  /  aimnis-dashboard
"""

from __future__ import annotations

import html
from contextlib import asynccontextmanager
from dataclasses import asdict

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse

from . import citations, db, stats
from .config import settings

# Gate 1 pass line: hit rate should climb past ~30% as the corpus grows.
_TARGET = 0.30


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.get_pool()
    yield
    await db.close_pool()


app = FastAPI(title="Aimnis Flywheel", lifespan=lifespan)

# The remote REST edge (POST /v1/search, GET /v1/stats). Same process as the
# dashboard so the hosted deploy is a single web service.
from .gateway import router as gateway_router  # noqa: E402

app.include_router(gateway_router)


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.get("/r/{token}")
async def redirect_citation(token: str):
    """Signed citation redirect: log the click (aggregate signal) and 302 to the
    stored destination. A forged/expired/out-of-range token 404s rather than
    redirecting, so this endpoint can never be used as an open redirector."""
    pool = await db.get_pool()
    url = await citations.resolve_click(pool, token)
    if url is None:
        return JSONResponse({"error": "unknown or invalid citation link"}, status_code=404)
    # 302 (temporary): the pool entry can be re-distilled, so the mapping isn't permanent.
    return RedirectResponse(url, status_code=302)


@app.get("/api/stats")
async def api_stats():
    """PUBLIC metrics — aggregates only. The raw query text (`top_queries`) and the
    per-host / per-entry click detail are withheld here and served only from the
    authenticated `/v1/stats`: a query string can echo a secret the best-effort
    scrubber missed, so we don't broadcast it to the open internet."""
    pool = await db.get_pool()
    s = await stats.gather(pool)
    series = await stats.flywheel_series(pool)
    calib = await stats.rerank_calibration(pool)
    clicks = await stats.click_analytics(pool)
    storage = await stats.storage_stats(pool)
    public = asdict(s)
    public.pop("top_queries", None)  # gated → /v1/stats
    return JSONResponse({
        **public,
        "target_hit_rate": _TARGET,
        "series": [asdict(p) for p in series],
        "rerank_calibration": asdict(calib),
        # aggregate click counts only; per-host/per-entry lists are gated to /v1/stats
        "click_analytics": {
            "clicks_total": clicks.clicks_total,
            "follow_through": clicks.follow_through,
        },
        "storage": asdict(storage),
    })


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    pool = await db.get_pool()
    s = await stats.gather(pool)
    series = await stats.flywheel_series(pool)
    clicks = await stats.click_analytics(pool)
    storage = await stats.storage_stats(pool)
    asof = await pool.fetchval("SELECT now()")
    return _render_page(s, series, clicks, storage, str(asof))


# --------------------------------------------------------------------------- #
# Server-side rendering (inline SVG, inline CSS — no external requests)
# --------------------------------------------------------------------------- #
def _svg(series: list) -> str:
    W, H = 760, 380
    ml, mr, mt, mb = 54, 20, 20, 44
    pw, ph = W - ml - mr, H - mt - mb

    def x(i, n):
        return ml + (pw * i / (n - 1) if n > 1 else 0)

    def y(rate):
        return mt + ph * (1 - rate)  # rate 0..1, 100% at top

    parts = [f'<svg viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Cache hit rate versus cumulative unique queries" '
             f'style="width:100%;height:auto">']

    # y gridlines + labels (0,25,50,75,100%)
    for pct in (0, 25, 50, 75, 100):
        gy = y(pct / 100)
        parts.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml + pw}" y2="{gy:.1f}" '
                     f'stroke="#20304a" stroke-width="1"/>')
        parts.append(f'<text x="{ml - 8}" y="{gy + 4:.1f}" text-anchor="end" '
                     f'fill="#8aa0bd" font-size="12">{pct}%</text>')

    # target reference line
    ty = y(_TARGET)
    parts.append(f'<line x1="{ml}" y1="{ty:.1f}" x2="{ml + pw}" y2="{ty:.1f}" '
                 f'stroke="#e3b341" stroke-width="1.5" stroke-dasharray="6 4"/>')
    parts.append(f'<text x="{ml + pw}" y="{ty - 6:.1f}" text-anchor="end" '
                 f'fill="#e3b341" font-size="12">Gate 1 target ~{int(_TARGET*100)}%</text>')

    n = len(series)
    if n == 0:
        parts.append(f'<text x="{W/2}" y="{H/2}" text-anchor="middle" fill="#8aa0bd" '
                     f'font-size="14">No lookups yet — run some searches.</text>')
    else:
        cum = " ".join(f"{x(i, n):.1f},{y(p.hit_rate):.1f}" for i, p in enumerate(series))
        roll = " ".join(f"{x(i, n):.1f},{y(p.rolling_hit_rate):.1f}" for i, p in enumerate(series))
        parts.append(f'<polyline points="{roll}" fill="none" stroke="#2dd4bf" stroke-width="2"/>')
        parts.append(f'<polyline points="{cum}" fill="none" stroke="#58a6ff" stroke-width="2.5"/>')
        # x-axis end labels (unique-query count)
        parts.append(f'<text x="{ml}" y="{H - 14}" fill="#8aa0bd" font-size="12">0</text>')
        parts.append(f'<text x="{ml + pw}" y="{H - 14}" text-anchor="end" fill="#8aa0bd" '
                     f'font-size="12">{series[-1].unique_queries} unique queries</text>')

    parts.append(f'<text x="{ml + pw/2}" y="{H - 14}" text-anchor="middle" fill="#8aa0bd" '
                 f'font-size="12">cumulative unique queries →</text>')
    parts.append("</svg>")
    return "".join(parts)


def _tile(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{html.escape(sub)}</div>' if sub else ""
    return (f'<div class="tile"><div class="label">{html.escape(label)}</div>'
            f'<div class="value">{html.escape(value)}</div>{sub_html}</div>')


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit in ("B", "KB") else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _render_page(s, series: list, clicks, storage, asof: str) -> str:
    per_entry = storage.bytes_per_entry
    proj_1m = _human_bytes(per_entry * 1_000_000) if per_entry else "—"
    tiles = "".join([
        _tile("Cache hit rate", f"{s.hit_rate:.0%}", f"{s.hits} hits / {s.lookups_total} lookups"),
        _tile("Recent hit rate", f"{s.recent_hit_rate:.0%}", f"last {s.recent_window} lookups"),
        _tile("Corpus", f"{s.corpus_total}", f"{s.corpus_servable} servable"),
        _tile("Hit mix", f"{s.hits_exact} + {s.hits_semantic}", "exact + semantic"),
        _tile("Sources / reply", f"{s.avg_results_per_reply:.1f}", "avg cited per answer"),
        _tile("Citations followed", f"{clicks.clicks_total}",
              f"{clicks.follow_through:.2f} per hit"),
        _tile("Pool storage", _human_bytes(storage.total_bytes),
              f"~{_human_bytes(per_entry)}/entry · ~{proj_1m} at 1M"),
    ])
    # NB: the most-reused-query / most-followed-source lists are intentionally NOT
    # rendered on this public page — raw query text could surface a secret the
    # best-effort scrubber missed. That detail lives behind the API key (/v1/stats).

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Aimnis · Flywheel</title>
<style>
  :root {{ color-scheme: dark; }}
  * {{ box-sizing: border-box; }}
  body {{ margin:0; background:#0b0f14; color:#e6edf3;
         font:15px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif; }}
  .wrap {{ max-width:840px; margin:0 auto; padding:28px 20px 48px; }}
  h1 {{ font-size:22px; margin:0 0 2px; }}
  .tag {{ color:#8aa0bd; margin:0 0 24px; }}
  .tiles {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr));
           gap:12px; margin-bottom:24px; }}
  .tile {{ background:#111820; border:1px solid #20304a; border-radius:10px; padding:14px 16px; }}
  .tile .label {{ color:#8aa0bd; font-size:12px; text-transform:uppercase; letter-spacing:.04em; }}
  .tile .value {{ font-size:26px; font-weight:650; margin-top:4px; }}
  .tile .sub {{ color:#8aa0bd; font-size:12px; margin-top:2px; }}
  .card {{ background:#111820; border:1px solid #20304a; border-radius:10px; padding:18px; }}
  .legend {{ display:flex; gap:18px; color:#8aa0bd; font-size:13px; margin:10px 2px 0; }}
  .legend b {{ display:inline-block; width:22px; height:3px; border-radius:2px; vertical-align:middle; margin-right:6px; }}
  h2 {{ font-size:15px; color:#c9d6e5; margin:28px 0 8px; }}
  ol {{ margin:0; padding-left:0; list-style:none; }}
  ol li {{ padding:6px 0; border-top:1px solid #182234; }}
  ol .n {{ color:#2dd4bf; font-variant-numeric:tabular-nums; margin-right:10px; }}
  ol .dim {{ color:#5c6f88; font-size:12px; }}
  .cols {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:0 28px; }}
  footer {{ color:#5c6f88; font-size:12px; margin-top:28px; }}
  a {{ color:#58a6ff; }}
</style></head>
<body><div class="wrap">
  <h1>Aimnis · Flywheel</h1>
  <p class="tag">Cache hit rate vs. cumulative unique queries — the Gate 1 test. It passes if the curve slopes up.</p>
  <div class="tiles">{tiles}</div>
  <div class="card">
    {_svg(series)}
    <div class="legend">
      <span><b style="background:#58a6ff"></b>cumulative hit rate</span>
      <span><b style="background:#2dd4bf"></b>recent (rolling) hit rate</span>
      <span><b style="background:#e3b341"></b>Gate 1 target</span>
    </div>
  </div>
  <footer>as of {html.escape(asof)} · <a href="/api/stats">/api/stats</a> (JSON, aggregate) ·
    per-query and per-source detail is available to API-key holders at <code>/v1/stats</code> ·
    click signal is aggregate (source + host + time) — no per-user data</footer>
</div></body></html>"""


def main() -> None:
    import uvicorn
    uvicorn.run("aimnis.api:app", host=settings.dashboard_host, port=settings.dashboard_port)


if __name__ == "__main__":
    main()
