"""Signed citation redirect: token round-trip, tamper rejection, fail-closed
routing, and the routed rendering in format_for_agent. All offline."""

from __future__ import annotations

from aimnis import citations, resolve
from aimnis.config import settings

EID = "11111111-2222-3333-4444-555555555555"
URL = "https://docs.python.org/3/library/os.html"


def _sign_on(monkeypatch, *, base="https://api.aimnis.org", secret="s3cr3t"):
    monkeypatch.setattr(settings, "citation_routing_enabled", True)
    monkeypatch.setattr(settings, "citation_signing_secret", secret)
    monkeypatch.setattr(settings, "citation_public_base_url", base)


def test_token_round_trips(monkeypatch):
    _sign_on(monkeypatch)
    tok = citations.token_for(EID, URL)
    assert tok and "/" not in tok  # urlsafe, unpadded
    # Token carries the entry id + URL hash (stable identity, not a position).
    assert citations.verify(tok) == (EID, citations.url_hash(URL))


def test_token_is_position_independent(monkeypatch):
    # Same (entry, url) → same token regardless of where the source sits, so a
    # link survives re-ordering the entry's sources.
    _sign_on(monkeypatch)
    assert citations.token_for(EID, URL) == citations.token_for(EID, URL)
    assert citations.token_for(EID, URL) != citations.token_for(EID, "https://other.example")


def test_verify_rejects_tampered_and_forged(monkeypatch):
    _sign_on(monkeypatch)
    tok = citations.token_for(EID, URL)
    assert citations.verify(tok[:-2] + ("aa" if tok[-2:] != "aa" else "bb")) is None
    assert citations.verify("not-base64-@@@") is None
    assert citations.verify("") is None
    # A token signed with a different secret must not verify under ours.
    monkeypatch.setattr(settings, "citation_signing_secret", "other-secret")
    assert citations.verify(tok) is None


def test_route_url_fail_closed(monkeypatch):
    # No secret → no routing (raw URLs emitted by callers).
    monkeypatch.setattr(settings, "citation_routing_enabled", True)
    monkeypatch.setattr(settings, "citation_signing_secret", None)
    monkeypatch.setattr(settings, "citation_public_base_url", "https://api.aimnis.org")
    assert citations.route_url(EID, URL) is None

    _sign_on(monkeypatch)
    # Disabled switch → no routing even with a secret.
    monkeypatch.setattr(settings, "citation_routing_enabled", False)
    assert citations.route_url(EID, URL) is None

    # No entry_id (unpooled live result) or no URL → no routing.
    monkeypatch.setattr(settings, "citation_routing_enabled", True)
    assert citations.route_url(None, URL) is None
    assert citations.route_url(EID, None) is None


def test_route_url_falls_back_to_gateway_url(monkeypatch):
    monkeypatch.setattr(settings, "citation_routing_enabled", True)
    monkeypatch.setattr(settings, "citation_signing_secret", "s")
    monkeypatch.setattr(settings, "citation_public_base_url", None)
    monkeypatch.setattr(settings, "gateway_url", "https://gw.example.org/")
    url = citations.route_url(EID, URL)
    assert url and url.startswith("https://gw.example.org/r/")  # trailing slash trimmed


def test_host_of():
    assert citations.host_of("https://docs.python.org/3/library/os.html") == "docs.python.org"
    assert citations.host_of("not a url") is None


def test_format_routes_urls_and_shows_host(monkeypatch):
    _sign_on(monkeypatch)
    out = resolve.format_for_agent({
        "source": "cache", "match": "exact", "answer": "do X [1]",
        "entry_id": EID,
        "results": [{"title": "Enable HTTP/2", "url": "https://nginx.org/en/docs/http2.html",
                     "snippet": "http2 on;"}],
    })
    assert "https://api.aimnis.org/r/" in out       # link is routed
    assert "nginx.org" in out                        # real host still shown inline
    assert "https://nginx.org/en/docs/http2.html" not in out  # raw URL not the link


def test_format_keeps_raw_url_when_unpooled(monkeypatch):
    _sign_on(monkeypatch)
    out = resolve.format_for_agent({
        "source": "live", "answer": None, "entry_id": None,
        "results": [{"title": "T", "url": "https://x.example/y", "snippet": "s"}],
    })
    assert "https://x.example/y" in out
    assert "/r/" not in out
