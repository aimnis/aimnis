"""Feature-based source selector — ordering logic, cold-start fallback, and the
click 'beats-its-position' signal. All offline."""

from __future__ import annotations

from datetime import datetime, timezone

from aimnis import select
from aimnis.config import settings

NOW = datetime(2026, 7, 3, tzinfo=timezone.utc)


def _src(url, title="", snippet="", fetched="2026-07-03T00:00:00+00:00"):
    return {"url": url, "title": title, "snippet": snippet, "fetched_at": fetched}


def test_empty():
    assert select.rank_sources("q", []) == []


def test_cold_start_preserves_provider_order(monkeypatch):
    # No overlap/clicks differences and equal freshness → stable provider order.
    monkeypatch.setattr(settings, "select_w_overlap", 0.0)
    monkeypatch.setattr(settings, "select_w_freshness", 0.0)
    srcs = [_src("https://a"), _src("https://b"), _src("https://c")]
    assert select.rank_sources("anything", srcs, now=NOW) == [0, 1, 2]


def test_lexical_overlap_promotes_relevant_source(monkeypatch):
    monkeypatch.setattr(settings, "select_w_rank", 1.0)
    monkeypatch.setattr(settings, "select_w_overlap", 5.0)  # make overlap decisive
    monkeypatch.setattr(settings, "select_w_freshness", 0.0)
    srcs = [
        _src("https://irrelevant", title="cooking pasta at home"),
        _src("https://relevant", title="enable HTTP/2 in nginx", snippet="listen 443 ssl http2"),
    ]
    order = select.rank_sources("how to enable http2 in nginx", srcs, now=NOW)
    assert order[0] == 1  # the relevant source outranks the earlier-but-off-topic one


def test_click_beats_position(monkeypatch):
    # A source shown LAST that nonetheless earned clicks should outrank an
    # unclicked source shown first — position-bias-resistant.
    monkeypatch.setattr(settings, "select_w_rank", 1.0)
    monkeypatch.setattr(settings, "select_w_overlap", 0.0)
    monkeypatch.setattr(settings, "select_w_freshness", 0.0)
    monkeypatch.setattr(settings, "select_w_clicks", 2.0)
    srcs = [_src("https://first"), _src("https://second"), _src("https://clicked-last")]
    clicks = {"https://clicked-last": 9}
    order = select.rank_sources("q", srcs, click_stats=clicks, now=NOW)
    assert order[0] == 2  # the clicked-but-last source wins


def test_order_sources_returns_dicts(monkeypatch):
    monkeypatch.setattr(settings, "select_w_overlap", 0.0)
    monkeypatch.setattr(settings, "select_w_freshness", 0.0)
    srcs = [_src("https://a"), _src("https://b")]
    assert [s["url"] for s in select.order_sources("q", srcs, now=NOW)] == ["https://a", "https://b"]
