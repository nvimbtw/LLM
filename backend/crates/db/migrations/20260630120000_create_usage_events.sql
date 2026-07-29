-- Append-only record of one chat generation turn, for the usage page. Kept
-- separate from `messages` so usage history survives chat/message deletion.
-- id_model is nullable + ON DELETE SET NULL so a deleted model doesn't take its
-- usage rows with it (they're attributed to "unknown" thereafter).
CREATE TABLE usage_events (
    id_usage      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_user       UUID NOT NULL REFERENCES users(id_user),
    id_model      UUID REFERENCES models(id_model) ON DELETE SET NULL,
    input_tokens  INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- The usage page always filters by user over a recent time window.
CREATE INDEX idx_usage_events_user_created ON usage_events (id_user, created_at);
