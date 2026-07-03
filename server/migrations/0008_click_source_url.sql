-- Stable click label: the destination URL a click resolved to. source_idx records
-- the position the source sat at when clicked (for position-bias debiasing), but
-- position shifts when an entry is re-selected/re-ordered — so the URL is the
-- durable identity that click-based source ranking trains on. Still no IP/UA/user.
ALTER TABLE citation_click ADD COLUMN IF NOT EXISTS source_url text;
