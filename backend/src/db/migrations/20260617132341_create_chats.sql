-- Add migration script here
CREATE TABLE chats (
    id_chat    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_user    UUID NOT NULL REFERENCES users(id_user),
    title      TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
