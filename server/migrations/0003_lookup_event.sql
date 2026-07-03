-- 0003_lookup_event.sql — append-only lookup log powering the Gate 1 flywheel metric.
--
-- pool_entry.hit_count is a per-entry popularity counter (a snapshot). It cannot
-- show the metric Gate 1 actually turns on: cache hit RATE plotted against
-- cumulative unique-query count over TIME. That needs one row per lookup with a
-- timestamp and outcome, which is what this table records. It is intentionally
-- cheap (one small insert per search, best-effort, off the hot answer path) and
-- never read during resolution — only by the stats tool / public dashboard.

CREATE TABLE IF NOT EXISTS lookup_event (
    id          bigserial PRIMARY KEY,
    ts          timestamptz NOT NULL DEFAULT now(),
    query_hash  text NOT NULL,          -- normalized-query hash (already scrubbed upstream)
    outcome     text NOT NULL
                  CHECK (outcome IN ('hit_exact','hit_semantic','miss','error')),
    distance    real,                   -- cosine distance for a semantic hit; NULL otherwise
    entry_id    uuid,                   -- pool entry served (hit) or created (miss); NULL on error
    niche       text
);

CREATE INDEX IF NOT EXISTS lookup_event_ts_idx      ON lookup_event (ts);
CREATE INDEX IF NOT EXISTS lookup_event_outcome_idx ON lookup_event (outcome);
