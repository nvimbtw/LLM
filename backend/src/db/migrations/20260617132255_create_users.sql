-- Add migration script here
CREATE TYPE access_level AS ENUM ('user', 'admin');

CREATE TABLE users (
    id_user      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username     TEXT NOT NULL UNIQUE,
    password     TEXT NOT NULL,
    access_level access_level NOT NULL DEFAULT 'user',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);
