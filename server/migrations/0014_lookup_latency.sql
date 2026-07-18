-- 0014_lookup_latency.sql — per-search latency + calling application on the
-- lookup log, so both are tied to the actual search and its outcome.
--
-- latency_ms: end-to-end resolve_search time in milliseconds (embed + cache
-- lookup for a hit; + live search + distill for a miss). Makes p50/p95 latency
-- BY OUTCOME queryable — the real proof of "instant from cache, escalate on miss".
--
-- user_agent: the client application that made the search (stdio/local = NULL),
-- so hit rate and latency can be broken down by app. This is per-search
-- attribution; request_log.user_agent is the coarser per-edge-request view.
ALTER TABLE lookup_event
    ADD COLUMN IF NOT EXISTS latency_ms  int,
    ADD COLUMN IF NOT EXISTS user_agent  text;
