CREATE TABLE projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    owner_id UUID NOT NULL REFERENCES users(id_user) ON DELETE CASCADE,
    name TEXT NOT NULL CHECK (char_length(name) BETWEEN 1 AND 120 AND btrim(name) <> ''),
    description TEXT NOT NULL DEFAULT '' CHECK (char_length(description) <= 2000),
    notes TEXT NOT NULL DEFAULT '' CHECK (char_length(notes) <= 8000),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE(id, owner_id)
);
CREATE INDEX projects_owner_created ON projects(owner_id, created_at DESC, id);

ALTER TABLE chats ADD COLUMN project_id UUID;
ALTER TABLE chats ADD CONSTRAINT chats_owned_project
    FOREIGN KEY (project_id, owner_id) REFERENCES projects(id, owner_id)
    ON DELETE SET NULL (project_id);
CREATE INDEX chats_owner_project_created ON chats(owner_id, project_id, created_at DESC, id);

CREATE FUNCTION project_memory_text(source_role conversation_role, source_content TEXT)
RETURNS TEXT LANGUAGE SQL IMMUTABLE PARALLEL SAFE
RETURN CASE WHEN source_role = 'assistant' THEN
    split_part(regexp_replace(left(source_content, 8192),
        '<\|thinking\|>.*?<\|/thinking\|>', '', 'gs'), '<|thinking|>', 1)
    ELSE left(source_content, 8192) END;

ALTER TABLE messages ADD COLUMN memory_words TSVECTOR
    GENERATED ALWAYS AS (to_tsvector('simple', project_memory_text(role, content))) STORED;
CREATE INDEX messages_memory_words ON messages USING GIN(memory_words)
    WHERE role IN ('user', 'assistant');
