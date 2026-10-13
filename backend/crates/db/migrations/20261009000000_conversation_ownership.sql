CREATE TYPE conversation_role AS ENUM ('user', 'assistant', 'tool');
CREATE TYPE conversation_feedback AS ENUM ('up', 'down');

CREATE TABLE chats (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id_user) ON DELETE CASCADE,
    title VARCHAR(200) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX chats_owner ON chats(owner_id);

CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chat_id UUID NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    position BIGINT NOT NULL,
    role conversation_role NOT NULL,
    content TEXT NOT NULL,
    tokens INTEGER NOT NULL DEFAULT 0 CHECK (tokens >= 0),
    model_id UUID,
    model_name TEXT,
    tool TEXT,
    argument TEXT,
    feedback conversation_feedback,
    request_id UUID,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(chat_id, position),
    UNIQUE(chat_id, request_id, role)
);
CREATE INDEX messages_chat ON messages(chat_id);

CREATE TABLE attachments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id_user) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    media_type VARCHAR(160) NOT NULL,
    size INTEGER NOT NULL CHECK (size > 0 AND size <= 52428800),
    content BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (octet_length(content) = size)
);
CREATE INDEX attachments_owner ON attachments(owner_id);
CREATE INDEX attachments_message ON attachments(message_id);
