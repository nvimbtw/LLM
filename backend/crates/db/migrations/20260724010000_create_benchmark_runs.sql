-- Benchmark history. One row per evaluation of one model, kept forever so scores
-- can be compared across models and across retrains of the same model.

CREATE TYPE benchmark_status AS ENUM ('running', 'done', 'failed');

CREATE TABLE benchmark_runs (
    id_run       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_model     UUID NOT NULL REFERENCES models(id_model) ON DELETE CASCADE,
    -- The model's name *at the time of the run*. Denormalized on purpose: a model
    -- can be renamed, and a history chart that retroactively relabels past runs
    -- makes it impossible to line a score up with the run that produced it.
    model_name   TEXT NOT NULL,
    status       benchmark_status NOT NULL DEFAULT 'running',
    -- The composite 0-100 headline, lifted out of `report` so the history list and
    -- the trend chart can be read without parsing every JSON blob.
    score        DOUBLE PRECISION,
    -- The full eval::Report. Stored opaquely: the suite will grow new metrics, and
    -- a column per metric would mean a migration each time.
    report       JSONB,
    error        TEXT,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- The history view is "this model's runs, newest first".
CREATE INDEX benchmark_runs_model_idx ON benchmark_runs (id_model, created_at DESC);
