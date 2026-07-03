-- 0001_init.sql — Aimnis M1 core schema.
-- Two concerns: the knowledge pool (== semantic cache) and the quota ledger.

CREATE EXTENSION IF NOT EXISTS vector;    -- pgvector: semantic cache lookup
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- gen_random_uuid()

-- ============================================================================
-- Knowledge pool == semantic cache. One row per distilled Q&A entry.
-- The same table backs both cache hits and the licensed corpus; there is no
-- separate cache to keep in sync.
-- ============================================================================
CREATE TABLE IF NOT EXISTS pool_entry (
    id                    uuid PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Query, ALREADY scrubbed of secrets/PII at ingress (never store raw).
    query_text            text NOT NULL,
    query_norm            text NOT NULL,            -- normalized (lowercased, ws-collapsed)
    query_hash            text NOT NULL,            -- hash(query_norm): exact dedup / single-flight key
    embedding             vector(384),              -- local embedding (bge-small); NULL until embedded

    -- Answer + provenance.
    answer_text           text,
    sources               jsonb NOT NULL DEFAULT '[]'::jsonb,
    model                 text,                     -- upstream model that produced the answer
    provenance            jsonb NOT NULL DEFAULT '{}'::jsonb,
    quality_score         real,

    status                text NOT NULL DEFAULT 'pending'
                            CHECK (status IN ('pending','active','stale','rejected')),
    opt_in                boolean NOT NULL DEFAULT true,

    -- Compliance flags (per-source upstream ToS; see memory/tos-audit.md).
    output_trainable      boolean NOT NULL DEFAULT false,  -- may enter the training-data feed
    attribution_required  boolean NOT NULL DEFAULT false,  -- e.g. "Built with Llama"
    no_grounded_cache     boolean NOT NULL DEFAULT false,  -- e.g. Gemini grounded results — never cache

    -- Freshness / stats.
    niche                 text,
    hit_count             integer NOT NULL DEFAULT 0,
    ttl_seconds           integer,
    expires_at            timestamptz,
    created_at            timestamptz NOT NULL DEFAULT now(),
    updated_at            timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS pool_entry_query_hash_idx ON pool_entry (query_hash);
CREATE INDEX IF NOT EXISTS pool_entry_status_idx     ON pool_entry (status);
CREATE INDEX IF NOT EXISTS pool_entry_expires_idx    ON pool_entry (expires_at);
-- HNSW cosine index for semantic lookup. NULL embeddings are skipped.
CREATE INDEX IF NOT EXISTS pool_entry_embedding_idx
    ON pool_entry USING hnsw (embedding vector_cosine_ops);

-- ============================================================================
-- Quota ledger. Governs upstream (OpenRouter :free) usage.
-- Enforces three limits: per-minute, per-day (calendar-UTC or rolling-24h),
-- and per-purpose daily sub-budgets so background precompute never starves the
-- interactive path (and vice versa). Failed attempts (incl. HTTP 429) COUNT —
-- they burn real quota, so the reservation is taken BEFORE the upstream call.
-- ============================================================================
CREATE TABLE IF NOT EXISTS quota_key (
    id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    provider         text NOT NULL DEFAULT 'openrouter',
    label            text NOT NULL UNIQUE,
    rpm_limit        integer NOT NULL,          -- keep < nominal 20 for safety margin
    rpd_limit        integer NOT NULL,          -- keep < nominal 1000 for safety margin
    day_window_kind  text NOT NULL DEFAULT 'calendar_utc'
                       CHECK (day_window_kind IN ('calendar_utc','rolling_24h')),
    active           boolean NOT NULL DEFAULT true,
    notes            text,
    created_at       timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS quota_budget (
    key_id       uuid NOT NULL REFERENCES quota_key(id) ON DELETE CASCADE,
    purpose      text NOT NULL,                 -- background_precompute | stale_refresh | interactive_fallback | ...
    daily_limit  integer NOT NULL,
    PRIMARY KEY (key_id, purpose)
);

CREATE TABLE IF NOT EXISTS upstream_call (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    key_id            uuid NOT NULL REFERENCES quota_key(id) ON DELETE CASCADE,
    purpose           text NOT NULL,
    model             text,
    query_hash        text,                     -- what this call was for (nullable)
    status            text NOT NULL DEFAULT 'in_flight'
                        CHECK (status IN ('in_flight','success','rate_limited','error','timeout','abandoned')),
    http_status       integer,
    prompt_tokens     integer,
    completion_tokens integer,
    error             text,
    created_at        timestamptz NOT NULL DEFAULT now(),
    completed_at      timestamptz
);

-- Counting queries always exclude 'abandoned'; partial indexes match them.
CREATE INDEX IF NOT EXISTS upstream_call_key_created_idx
    ON upstream_call (key_id, created_at) WHERE status <> 'abandoned';
CREATE INDEX IF NOT EXISTS upstream_call_key_purpose_created_idx
    ON upstream_call (key_id, purpose, created_at) WHERE status <> 'abandoned';

-- ----------------------------------------------------------------------------
-- reserve_upstream_call: atomic check-and-reserve. Serialized per key via an
-- advisory lock so concurrent workers can't race past the limits (which would
-- cause 429s that themselves burn quota). Returns a granted flag + reason, and
-- on grant inserts an 'in_flight' row that counts immediately.
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION reserve_upstream_call(
    p_label       text,
    p_purpose     text,
    p_model       text DEFAULT NULL,
    p_query_hash  text DEFAULT NULL
)
RETURNS TABLE (granted boolean, reason text, call_id uuid)
LANGUAGE plpgsql AS $$
DECLARE
    k               quota_key%ROWTYPE;
    v_day_start     timestamptz;
    v_minute_count  integer;
    v_day_count     integer;
    v_purpose_count integer;
    v_purpose_limit integer;
    v_new_id        uuid;
BEGIN
    SELECT * INTO k FROM quota_key WHERE label = p_label;
    IF NOT FOUND THEN
        RETURN QUERY SELECT false, 'unknown_key', NULL::uuid; RETURN;
    END IF;
    IF NOT k.active THEN
        RETURN QUERY SELECT false, 'key_inactive', NULL::uuid; RETURN;
    END IF;

    -- Serialize all reservations for this key.
    PERFORM pg_advisory_xact_lock(hashtext('aimnis_quota_' || k.id::text));

    IF k.day_window_kind = 'rolling_24h' THEN
        v_day_start := now() - interval '24 hours';
    ELSE
        v_day_start := date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
    END IF;

    -- Per-minute limit.
    SELECT count(*) INTO v_minute_count
      FROM upstream_call
     WHERE key_id = k.id AND status <> 'abandoned'
       AND created_at >= now() - interval '60 seconds';
    IF v_minute_count >= k.rpm_limit THEN
        RETURN QUERY SELECT false, 'rate_minute', NULL::uuid; RETURN;
    END IF;

    -- Per-day limit.
    SELECT count(*) INTO v_day_count
      FROM upstream_call
     WHERE key_id = k.id AND status <> 'abandoned'
       AND created_at >= v_day_start;
    IF v_day_count >= k.rpd_limit THEN
        RETURN QUERY SELECT false, 'rate_day', NULL::uuid; RETURN;
    END IF;

    -- Per-purpose daily budget (only if one is configured for this purpose).
    SELECT daily_limit INTO v_purpose_limit
      FROM quota_budget WHERE key_id = k.id AND purpose = p_purpose;
    IF FOUND THEN
        SELECT count(*) INTO v_purpose_count
          FROM upstream_call
         WHERE key_id = k.id AND purpose = p_purpose AND status <> 'abandoned'
           AND created_at >= v_day_start;
        IF v_purpose_count >= v_purpose_limit THEN
            RETURN QUERY SELECT false, 'purpose_budget', NULL::uuid; RETURN;
        END IF;
    END IF;

    INSERT INTO upstream_call (key_id, purpose, model, query_hash, status)
         VALUES (k.id, p_purpose, p_model, p_query_hash, 'in_flight')
      RETURNING id INTO v_new_id;

    RETURN QUERY SELECT true, 'granted', v_new_id;
END;
$$;

-- Finalize a reserved call with its real outcome.
CREATE OR REPLACE FUNCTION record_upstream_outcome(
    p_call_id           uuid,
    p_status            text,
    p_http_status       integer DEFAULT NULL,
    p_prompt_tokens     integer DEFAULT NULL,
    p_completion_tokens integer DEFAULT NULL,
    p_error             text DEFAULT NULL
)
RETURNS void LANGUAGE sql AS $$
    UPDATE upstream_call
       SET status            = p_status,
           http_status       = p_http_status,
           prompt_tokens     = p_prompt_tokens,
           completion_tokens = p_completion_tokens,
           error             = p_error,
           completed_at      = now()
     WHERE id = p_call_id;
$$;

-- Current usage snapshot for a key (for the dashboard / pre-flight checks).
CREATE OR REPLACE FUNCTION quota_usage(p_label text)
RETURNS TABLE (minute_used integer, minute_limit integer,
               day_used integer, day_limit integer, day_start timestamptz)
LANGUAGE plpgsql AS $$
DECLARE
    k           quota_key%ROWTYPE;
    v_day_start timestamptz;
BEGIN
    SELECT * INTO k FROM quota_key WHERE label = p_label;
    IF NOT FOUND THEN RETURN; END IF;

    IF k.day_window_kind = 'rolling_24h' THEN
        v_day_start := now() - interval '24 hours';
    ELSE
        v_day_start := date_trunc('day', now() AT TIME ZONE 'UTC') AT TIME ZONE 'UTC';
    END IF;

    RETURN QUERY
      SELECT
        (SELECT count(*)::int FROM upstream_call
          WHERE key_id = k.id AND status <> 'abandoned'
            AND created_at >= now() - interval '60 seconds'),
        k.rpm_limit,
        (SELECT count(*)::int FROM upstream_call
          WHERE key_id = k.id AND status <> 'abandoned'
            AND created_at >= v_day_start),
        k.rpd_limit,
        v_day_start;
END;
$$;
