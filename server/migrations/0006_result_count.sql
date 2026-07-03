-- Number of cited sources returned in each reply (the results/sources list length):
-- the stored entry's sources on a cache hit, the live results on a miss. NULL on
-- error/empty replies (nothing was returned). Lets us report the average grounding
-- richness per reply and watch it move as distillation/search tuning changes.
ALTER TABLE lookup_event ADD COLUMN IF NOT EXISTS result_count int;
