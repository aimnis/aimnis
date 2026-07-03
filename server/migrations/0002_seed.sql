-- 0002_seed.sql — seed the primary OpenRouter :free key with conservative limits.
-- Nominal OpenRouter limits are 20 req/min and 1,000 req/day (the latter needs a
-- one-time $10 credit purchase). We seed BELOW nominal so a burst never trips a
-- 429 — which would itself consume quota. Adjust via UPDATE once real headroom
-- is observed. Budgets sum to rpd_limit (650 + 200 + 100 = 950).

INSERT INTO quota_key (label, provider, rpm_limit, rpd_limit, day_window_kind, notes)
VALUES ('primary-free', 'openrouter', 18, 950, 'calendar_utc',
        'OpenRouter :free primary key. Conservative margins below nominal 20/min, 1000/day.')
ON CONFLICT (label) DO NOTHING;

INSERT INTO quota_budget (key_id, purpose, daily_limit)
SELECT k.id, b.purpose, b.daily_limit
  FROM quota_key k
  CROSS JOIN (VALUES
        ('background_precompute', 650),
        ('stale_refresh',         200),
        ('interactive_fallback',  100)
  ) AS b(purpose, daily_limit)
 WHERE k.label = 'primary-free'
ON CONFLICT (key_id, purpose) DO NOTHING;
