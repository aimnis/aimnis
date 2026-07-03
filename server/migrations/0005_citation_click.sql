-- Citation click log — implicit relevance telemetry on the POOL, not on users.
--
-- When an agent/browser follows a cited source, it goes through the signed
-- /r/<token> redirect, which appends one row here and 302s to the destination.
-- We record only which pool entry + which of its sources was followed, plus the
-- destination host and a timestamp. NO IP, NO user-agent, NO session/user id —
-- this is aggregate ranking signal (which entries earn follow-through, which
-- sources are dead weight), never per-user tracking.

CREATE TABLE IF NOT EXISTS citation_click (
    id           bigserial PRIMARY KEY,
    entry_id     uuid NOT NULL REFERENCES pool_entry(id) ON DELETE CASCADE,
    source_idx   int  NOT NULL,           -- 0-based index into pool_entry.sources
    host         text,                    -- destination host only (e.g. "docs.python.org")
    created_at   timestamptz NOT NULL DEFAULT now()
);

-- Per-entry aggregation (CTR / follow-through) is the hot read path.
CREATE INDEX IF NOT EXISTS citation_click_entry_idx ON citation_click (entry_id);
