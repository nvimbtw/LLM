-- Add migration script here
CREATE TABLE models (
    id_model    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        TEXT NOT NULL UNIQUE,
    description TEXT,
    parameters  JSONB,
    context     INTEGER,
    family      TEXT
);
