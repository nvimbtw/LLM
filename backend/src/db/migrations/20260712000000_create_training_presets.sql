-- Named, reusable training parameter sets, scoped per user. `config` holds a
-- TrainConfig-shaped JSON blob (inspectable, not queried column-wise), mirroring
-- how `models.parameters` stores a config. Presets are personal scratch data, so
-- they cascade-delete with the user. A preset name is unique per user, not
-- globally, which gives the route layer a clean 409 path and an upsert target.
CREATE TABLE training_presets (
    id_preset  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_user    UUID NOT NULL REFERENCES users(id_user) ON DELETE CASCADE,
    name       TEXT NOT NULL,
    config     JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (id_user, name)
);
