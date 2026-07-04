-- 0011_satisfaction.sql — hit-satisfaction signal: explicit rejects + implicit
-- retry (reformulation) detection.
--
-- Citation clicks turned out to be a near-useless usefulness signal for AGENT
-- consumers (agents read the answer text and rarely fetch links; a click is as
-- likely distrust as interest). What we actually want per served hit is "did the
-- caller accept this answer?", measured two ways:
--   * EXPLICIT: the hit response invites the agent to re-search with
--     reject_entry=<id> when the cached answer doesn't match its question. The
--     retry lookup records `rejected_entry`, and the entry's reject_count bumps.
--   * IMPLICIT: the same client re-asking a near-duplicate question (embedding
--     distance below a strict threshold) within a short window means the hit
--     didn't satisfy. Needs per-lookup client identity + query embedding.
--
-- `client_hash` is a truncated sha256 of the caller's client id ("admin" for env
-- admin keys) — enough to sequence one client's lookups, never a direct identity.
-- Only the AGGREGATE satisfaction rate is ever published (dashboard invariant).

ALTER TABLE lookup_event
    ADD COLUMN IF NOT EXISTS client_hash    text,
    ADD COLUMN IF NOT EXISTS embedding      vector(384),
    ADD COLUMN IF NOT EXISTS rejected_entry uuid;

-- Satisfaction scans walk one client's recent lookups.
CREATE INDEX IF NOT EXISTS lookup_event_client_idx ON lookup_event (client_hash, id);

-- Explicit-reject counter per entry (demotion signal; hit_count's counterpart).
ALTER TABLE pool_entry
    ADD COLUMN IF NOT EXISTS reject_count integer NOT NULL DEFAULT 0;
