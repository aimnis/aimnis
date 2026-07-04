-- 0010_byok.sql — BYOK: per-client upstream credentials, encrypted at rest.
--
-- A client may attach their own OpenRouter key and/or search-provider key. Their
-- cache MISSES then spend THEIR upstream quota instead of the shared service
-- ceiling, in exchange for much higher per-key caps ("your keys, your limits").
--
-- ToS invariants (the anti-"quota pooling" line):
--   * A client's keys are used EXCLUSIVELY for that client's own requests — never
--     to serve other users' misses. Each account is genuinely its owner's; this is
--     Cursor-style BYOK, not multi-accounting.
--   * Pool entries produced under BYOK credentials are provenance-tagged
--     (byok_search / byok_distill in pool_entry.provenance) so any later
--     per-provider ToS finding can filter or purge them — no anonymous taint.
--
-- Keys are encrypted with pgcrypto (pgp_sym_encrypt) under AIMNIS_BYOK_SECRET,
-- which lives only in the deploy environment. No secret configured ⇒ BYOK is
-- disabled fail-closed (nothing stored, nothing decryptable). Values are never
-- returned by any API and never logged.

ALTER TABLE api_client
    ADD COLUMN IF NOT EXISTS openrouter_key_enc bytea,
    ADD COLUMN IF NOT EXISTS search_provider    text,
    ADD COLUMN IF NOT EXISTS search_key_enc     bytea;

ALTER TABLE api_client DROP CONSTRAINT IF EXISTS api_client_search_provider_check;
ALTER TABLE api_client ADD CONSTRAINT api_client_search_provider_check
    CHECK (search_provider IS NULL OR search_provider IN ('brave','tavily','exa'));
