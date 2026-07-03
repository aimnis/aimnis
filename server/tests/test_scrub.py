"""Ingress scrubber tests (pure, no I/O)."""

from __future__ import annotations

from aimnis import scrub


def test_no_secrets_unchanged():
    q = "how to use functools.lru_cache in python"
    r = scrub.scrub(q)
    assert r.text == q and r.secret_count == 0 and r.safe_to_pool and r.findings == {}


def test_aws_key_redacted():
    r = scrub.scrub("why does AKIAIOSFODNN7EXAMPLE fail in boto3")
    assert "AKIA" not in r.text and "⟨AWS_KEY⟩" in r.text
    assert r.findings == {"AWS_KEY": 1}


def test_github_token_redacted():
    r = scrub.scrub("remote uses ghp_" + "a" * 36 + " for auth")
    assert "ghp_" not in r.text and "⟨GITHUB_TOKEN⟩" in r.text


def test_openai_style_key_redacted():
    r = scrub.scrub("OpenAI(api_key='sk-proj-" + "A1b2C3d4" * 4 + "')")
    assert "sk-proj-" not in r.text and "⟨API_KEY⟩" in r.text


def test_private_key_block_redacted():
    key = "-----BEGIN RSA PRIVATE KEY-----\nMIIBOgIBAAJ...\n-----END RSA PRIVATE KEY-----"
    r = scrub.scrub("this key errors: " + key)
    assert "BEGIN RSA" not in r.text and "⟨PRIVATE_KEY⟩" in r.text


def test_connection_string_redacted():
    r = scrub.scrub("connect to postgres://admin:s3cr3tpw@db.internal:5432/app")
    assert "s3cr3tpw" not in r.text and "⟨CONNECTION_STRING⟩" in r.text


def test_password_assignment_keeps_key_redacts_value():
    r = scrub.scrub("config has password=hunter2000 set")
    assert "hunter2000" not in r.text and "⟨SECRET⟩" in r.text and "password" in r.text


def test_email_pii_redacted():
    r = scrub.scrub("ping jane.doe@example.com about it")
    assert "jane.doe@example.com" not in r.text and "⟨EMAIL⟩" in r.text


def test_jwt_redacted():
    r = scrub.scrub("Authorization eyJhbGciOiJIUzI.eyJzdWIiOiIxMjM.SflKxwRJSMeKKF2QT4")
    assert "⟨JWT⟩" in r.text and "eyJhbGci" not in r.text


def test_high_entropy_unknown_secret_redacted():
    r = scrub.scrub("the value is X7k9Qm2Rp4Zt8Wn6Ld3Vb5Yc1Hf0Gj here")
    assert "⟨HIGH_ENTROPY⟩" in r.text and "X7k9Qm2" not in r.text


def test_benign_code_identifier_not_over_redacted():
    q = "def authentication_middleware_handler(request): return process(request)"
    r = scrub.scrub(q)  # long, but all-alpha (no digit) → left intact
    assert r.text == q and r.secret_count == 0


def test_density_gate_blocks_pooling():
    q = ("keys AKIAIOSFODNN7EXAMPLE AKIAIOSFODNN7EXAMPL2 ghp_" + "a" * 36
         + " xoxb-1234567890 leaked")
    r = scrub.scrub(q)
    assert r.secret_count >= 4 and r.safe_to_pool is False


def test_disabled_passthrough(monkeypatch):
    from aimnis.config import settings
    monkeypatch.setattr(settings, "scrub_enabled", False)
    r = scrub.scrub("AKIAIOSFODNN7EXAMPLE")
    assert r.text == "AKIAIOSFODNN7EXAMPLE" and r.secret_count == 0
