-- Per-message token length under the model's tokenizer, so a chat's cumulative
-- token usage can be summed for display without re-tokenizing. Default 0 covers
-- existing rows (backfilling historical counts would need the model's merges).
ALTER TABLE messages ADD COLUMN IF NOT EXISTS tokens INT NOT NULL DEFAULT 0;
