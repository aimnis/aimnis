-- 0012_anon_usage.sql — per-IP daily budgets for KEY-LESS use of the hosted edge.
--
-- Keyless search is the front door of the funnel: an agent that discovers /mcp can
-- search immediately, no registration. Cache hits are near-free to serve, so they
-- are NOT budgeted here — only the two things that cost real money or invite abuse:
--   misses         live upstream searches (search provider + distill tokens)
--   registrations  in-band `register` tool issuances (keys minted per IP per day)
--
-- Privacy: the IP never lands in the table — only a salted sha256 prefix
-- (apikeys.hash_ip), consistent with client_hash in lookup_event. Rows are one per
-- (ip_hash, UTC day) and tiny; reserve_anon() opportunistically purges expired days.
CREATE TABLE IF NOT EXISTS anon_usage (
    ip_hash        text NOT NULL,
    day            date NOT NULL,
    misses         integer NOT NULL DEFAULT 0,
    registrations  integer NOT NULL DEFAULT 0,
    PRIMARY KEY (ip_hash, day)
);

-- ----------------------------------------------------------------------------
-- reserve_anon: atomically take one unit of `kind` ('miss' | 'registration')
-- for this ip_hash today, bounded by p_cap. Returns whether the unit was
-- granted. A single INSERT ... ON CONFLICT ... WHERE is atomic under
-- concurrency (no advisory lock needed for a monotone capped counter).
-- ----------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION reserve_anon(p_ip_hash text, p_kind text, p_cap integer)
RETURNS boolean
LANGUAGE plpgsql AS $$
DECLARE
    v_granted boolean;
BEGIN
    IF p_cap <= 0 THEN
        RETURN false;
    END IF;
    -- Opportunistic cleanup: expired days are dead weight, drop them as we go
    -- (cheap: primary-key range scan, and only this ip_hash's rows).
    DELETE FROM anon_usage WHERE ip_hash = p_ip_hash AND day < (now() AT TIME ZONE 'utc')::date;

    IF p_kind = 'miss' THEN
        INSERT INTO anon_usage (ip_hash, day, misses)
        VALUES (p_ip_hash, (now() AT TIME ZONE 'utc')::date, 1)
        ON CONFLICT (ip_hash, day) DO UPDATE SET misses = anon_usage.misses + 1
        WHERE anon_usage.misses < p_cap
        RETURNING true INTO v_granted;
    ELSIF p_kind = 'registration' THEN
        INSERT INTO anon_usage (ip_hash, day, registrations)
        VALUES (p_ip_hash, (now() AT TIME ZONE 'utc')::date, 1)
        ON CONFLICT (ip_hash, day) DO UPDATE SET registrations = anon_usage.registrations + 1
        WHERE anon_usage.registrations < p_cap
        RETURNING true INTO v_granted;
    ELSE
        RAISE EXCEPTION 'unknown anon reservation kind %', p_kind;
    END IF;
    RETURN COALESCE(v_granted, false);
END;
$$;
