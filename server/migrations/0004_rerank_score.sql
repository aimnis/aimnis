-- 0004_rerank_score.sql — record the cross-encoder rerank score on each lookup.
--
-- Tuning rerank_min_score (the semantic-hit accept floor) needs the score
-- DISTRIBUTION of both accepted hits and rejected near-misses on real traffic.
-- We log the best candidate's rerank score (0..1) alongside its vector distance:
--   hit_semantic → the accepted candidate's score
--   miss         → the best candidate's score when one was found but REJECTED
--                  (NULL when there was no semantic candidate at all)
-- Idempotent; the column is nullable (exact hits / errors / rerank-disabled = NULL).

ALTER TABLE lookup_event ADD COLUMN IF NOT EXISTS rerank_score real;
