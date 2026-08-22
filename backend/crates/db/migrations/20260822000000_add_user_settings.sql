-- Chat settings (decoding knobs, system prompt, UI toggles) as one JSONB blob
-- per user, opaque to this side the same way training_presets.config is: the
-- frontend's Settings shape can grow without a migration here. A column, not a
-- side table, because a user has exactly one — a table keyed UNIQUE(id_user)
-- would only ever hold the row the user row already has.
ALTER TABLE users ADD COLUMN settings JSONB NOT NULL DEFAULT '{}'::jsonb;
