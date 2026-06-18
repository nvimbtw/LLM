-- Add migration script here
CREATE TABLE messages (
    id_message UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_chat    UUID NOT NULL REFERENCES chats(id_chat),
    id_user    UUID REFERENCES users(id_user),
    id_model   UUID REFERENCES models(id_model),
    content    TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
