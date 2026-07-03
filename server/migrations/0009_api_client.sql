-- 0009_api_client.sql — DB-backed client API keys for self-serve eval access.
--
-- Until now the gateway's key allowlist lived in an env var (AIMNIS_GATEWAY_API_KEYS),
-- so issuing or revoking a key meant a redeploy. A self-serve portal can't redeploy
-- per signup, and "revoke any key at any time" must be instant — so client keys move
-- here. The env-var keys survive as an ADMIN/bootstrap path (unlimited, no metering);
-- these DB keys are the metered, revocable eval keys handed out by the portal.
--
-- Each key carries its own per-minute + per-day request cap. That is the safety rail
-- that makes handing keys to strangers survivable: ALL eval traffic shares one upstream
-- OpenRouter :free quota (~1,000 req/day for the whole service), so a single key — or a
-- leaked one — must not be able to drain it for everyone. The caps ration that fixed pie.

CREATE TABLE IF NOT EXISTS api_client (
    id           uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    -- sha256(presented key), hex. The plaintext key is shown to the registrant ONCE at
    -- issue and never stored; we only ever compare hashes (preimage-resistant, so an
    -- indexed hash lookup leaks nothing the way a byte-by-byte plaintext compare would).
    key_hash     text NOT NULL UNIQUE,
    key_prefix   text NOT NULL,          -- leading chars of the key, for operator identification in listings
    label        text,                   -- optional human label / use-case blurb
    email        text,                   -- registrant contact (key delivery, revocation notice); PII, deletable on request
    status       text NOT NULL DEFAULT 'active' CHECK (status IN ('active','revoked')),
    rpm_limit    integer NOT NULL DEFAULT 20,    -- per-minute request cap
    rpd_limit    integer NOT NULL DEFAULT 200,   -- per-day request cap (rations the shared upstream quota)
    notes        text,
    created_at   timestamptz NOT NULL DEFAULT now(),
    revoked_at   timestamptz
);

CREATE INDEX IF NOT EXISTS api_client_email_idx ON api_client (lower(email));
-- At most one ACTIVE key per email (re-registering rotates it — see apikeys.issue).
CREATE UNIQUE INDEX IF NOT EXISTS api_client_active_email_idx
    ON api_client (lower(email)) WHERE status = 'active' AND email IS NOT NULL;

-- Per-request log — the counter behind the rate limits, and coarse per-key usage.
-- Aggregate only: no query text, no IP/UA — consistent with the aggregate-only
-- telemetry stance elsewhere (see citation_click).
CREATE TABLE IF NOT EXISTS api_request (
    id          bigserial PRIMARY KEY,
    client_id   uuid NOT NULL REFERENCES api_client(id) ON DELETE CASCADE,
    created_at  timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS api_request_client_created_idx ON api_request (client_id, created_at);

-- ----------------------------------------------------------------------------
-- reserve_client_request: atomic authenticate-and-reserve for a client key.
-- Mirrors reserve_upstream_call — serialize per client via an advisory lock so
-- concurrent requests can't race past the caps — but keys off the sha256 hash.
-- On grant it logs one api_request row (which counts immediately) and returns the
-- client id. Reason is one of: granted | unknown_key | revoked | rate_minute | rate_day.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION reserve_client_request(p_key_hash text)
RETURNS TABLE (granted boolean, reason text, client_id uuid)
LANGUAGE plpgsql AS $$
DECLARE
    c              api_client%ROWTYPE;
    v_minute_count integer;
    v_day_start    timestamptz;
    v_day_count    integer;
BEGIN
    SELECT * INTO c FROM api_client WHERE key_hash = p_key_hash;
    IF NOT FOUND THEN
        RETURN QUERY SELECT false, 'unknown_key', NULL::uuid; RETURN;
    END IF;
    IF c.status <> 'active' THEN
        RETURN QUERY SELECT false, 'revoked', c.id; RETURN;
    END IF;

    PERFORM pg_advisory_xact_lock(hashtext('aimnis_client_' || c.id::text));

    -- Qualify api_request.client_id — the OUT column is also named client_id.
    SELECT count(*) INTO v_minute_count
      FROM api_request
     WHERE api_request.client_id = c.id AND created_at >= now() - interval '60 seconds';
    IF v_minute_count >= c.rpm_limit THEN
        RETURN QUERY SELECT false, 'rate_minute', c.id; RETURN;
    END IF;

    v_day_start := date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
    SELECT count(*) INTO v_day_count
      FROM api_request
     WHERE api_request.client_id = c.id AND created_at >= v_day_start;
    IF v_day_count >= c.rpd_limit THEN
        RETURN QUERY SELECT false, 'rate_day', c.id; RETURN;
    END IF;

    INSERT INTO api_request (client_id) VALUES (c.id);
    RETURN QUERY SELECT true, 'granted', c.id;
END;
$$;

-- ============================================================================
-- Waitlist — captured when eval registration is paused ("at capacity"). Email
-- only + a timestamp; notified_at is stamped when capacity is offered back.
-- ============================================================================
CREATE TABLE IF NOT EXISTS waitlist (
    id           bigserial PRIMARY KEY,
    email        text NOT NULL,
    created_at   timestamptz NOT NULL DEFAULT now(),
    notified_at  timestamptz
);
CREATE UNIQUE INDEX IF NOT EXISTS waitlist_email_idx ON waitlist (lower(email));

-- ============================================================================
-- Service flags — small key/value toggles the operator flips live (no redeploy).
-- 'registration_open' gates self-serve eval-key issuance; false ⇒ the portal shows
-- "at capacity" and offers the waitlist.
-- ============================================================================
CREATE TABLE IF NOT EXISTS service_flag (
    name        text PRIMARY KEY,
    enabled     boolean NOT NULL,
    updated_at  timestamptz NOT NULL DEFAULT now()
);
INSERT INTO service_flag (name, enabled) VALUES ('registration_open', true)
    ON CONFLICT (name) DO NOTHING;
