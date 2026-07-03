"""Transactional email via Resend (HTTP API).

Why a provider and not SMTP-from-the-box: Railway (like most hosts) blocks outbound
port 25 and its IPs sit on shared/blocklisted ranges, so mail sent directly from the
app lands in spam or bounces. "Spam clearance" = a provider with warmed IP reputation
PLUS domain authentication (SPF + DKIM + DMARC DNS records on aimnis.com). Resend gives
both over a plain HTTPS call. Set up: add the domain in Resend, add the DKIM/SPF records
it prints to aimnis.com's DNS, then set AIMNIS_RESEND_API_KEY.

With no key configured this is a logged no-op — the portal still issues keys (and shows
them on-screen), it just doesn't email them, so dev / self-host works without a provider.
"""

from __future__ import annotations

import logging

import httpx

from .config import settings

log = logging.getLogger("aimnis.email")


async def send_email(to: str, subject: str, html: str) -> bool:
    """Send one email. Returns True if handed off to Resend, False if no-op/failed."""
    if not settings.resend_api_key:
        log.info("email no-op (no RESEND_API_KEY): to=%s subject=%r", to, subject)
        return False
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            r = await client.post(
                settings.resend_endpoint,
                headers={"Authorization": f"Bearer {settings.resend_api_key}"},
                json={"from": settings.email_from, "to": [to], "subject": subject, "html": html},
            )
        if r.status_code >= 300:
            log.warning("resend send failed: %s %s", r.status_code, r.text[:300])
            return False
        return True
    except Exception as exc:  # noqa: BLE001 — email must never crash the request path
        log.warning("resend send error: %s", exc)
        return False
