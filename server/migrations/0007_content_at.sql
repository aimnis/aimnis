-- Content-freshness timestamp: when this entry's ANSWER/sources were produced or
-- last re-distilled. Distinct from updated_at (which _bump_hit moves on every
-- serve, so it tracks last-access, not freshness) and from created_at (which
-- never moves when an entry is refreshed). This is the honest staleness signal we
-- return to the agent. Set on insert and on update_answer (refresh); NEVER bumped
-- by a serve. Backfill existing rows to created_at (conservative — understates
-- freshness for already-refreshed entries rather than overclaiming it).
ALTER TABLE pool_entry ADD COLUMN IF NOT EXISTS content_at timestamptz;
UPDATE pool_entry SET content_at = created_at WHERE content_at IS NULL;
ALTER TABLE pool_entry ALTER COLUMN content_at SET DEFAULT now();
ALTER TABLE pool_entry ALTER COLUMN content_at SET NOT NULL;
