-- models predates the created_at convention used elsewhere; add it so the
-- registry can be ordered by recency (matching users/chats/messages).
ALTER TABLE models ADD COLUMN created_at TIMESTAMPTZ NOT NULL DEFAULT now();
