-- The GPU queue. Every piece of work that touches the device — a training run, a
-- benchmark, a chat turn — is a row here, and the runner executes them one at a
-- time in `queue_rank` order. Before this table the "one job at a time" rule was
-- enforced by route handlers asking two in-memory registries whether they were
-- busy, which meant it was only true for the paths that remembered to ask, and
-- was forgotten entirely across a restart.

CREATE TYPE job_type AS ENUM ('train', 'benchmark', 'chat');

-- `cancelling` is a real state, not a client-side fiction: a running job can only
-- stop at a step boundary, so there is a genuine interval between the request and
-- the worker noticing it.
CREATE TYPE job_status AS ENUM (
    'queued', 'running', 'cancelling', 'done', 'cancelled', 'failed'
);

CREATE TABLE jobs (
    id_job      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    kind        job_type   NOT NULL,
    status      job_status NOT NULL DEFAULT 'queued',
    -- What to call this job in a list. Denormalized on purpose (same reasoning as
    -- benchmark_runs.model_name): a model renamed after the fact must not
    -- retroactively relabel the job that trained it.
    label       TEXT NOT NULL,
    -- Ordering key for the queued band. A new job goes in at max+1; a drag writes
    -- the midpoint of its two new neighbours, so reordering touches one row rather
    -- than renumbering the tail. Doubles run out of midpoints after ~50 inserts
    -- between the same adjacent pair, which db::job::reorder handles by
    -- renormalizing the band to whole numbers when a gap gets too small.
    queue_rank  DOUBLE PRECISION NOT NULL,
    -- Everything needed to run this job, and to run it *again* from the retry
    -- button: a TrainConfig, a { id_model }, or a { id_chat, id_model, ... }.
    -- Opaque here — each kind owns its own shape, and they change independently.
    config      JSONB NOT NULL,
    -- Coarse progress, durable across a page load. The high-frequency frames (loss
    -- series, per-layer diagnostics, tokens) stay in memory and go out over the
    -- socket; writing those here would flatten the database for no gain, since
    -- nothing in flight survives a restart anyway.
    progress    DOUBLE PRECISION NOT NULL DEFAULT 0,
    stage       TEXT,
    -- The record this job produced: models.id_model, benchmark_runs.id_run, or
    -- messages.id_message. Deliberately not a foreign key — it points into three
    -- different tables, and the job outlives a deleted result.
    ref_id      UUID,
    error       TEXT,
    -- Who queued it. SET NULL rather than CASCADE: a deleted account shouldn't
    -- erase the history of what ran on the machine.
    id_user     UUID REFERENCES users(id_user) ON DELETE SET NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at  TIMESTAMPTZ,
    finished_at TIMESTAMPTZ
);

-- The hot read: "what runs next", and the queue band on /jobs. Partial, because
-- the live band stays small no matter how long the history grows.
CREATE INDEX jobs_queue_idx ON jobs (queue_rank)
    WHERE status IN ('queued', 'running', 'cancelling');

-- The history read: newest first, sometimes filtered by kind.
CREATE INDEX jobs_history_idx ON jobs (created_at DESC);

-- Whether the runner is draining the queue. A single row (the CHECK on the
-- primary key is what makes it a singleton) rather than a config file, because
-- pausing has to survive a restart: the point of the switch is to stop the
-- machine picking up work while you are editing datasets or want the GPU for
-- chat, and a restart in the middle of that shouldn't quietly resume.
CREATE TABLE queue_state (
    only_row   BOOLEAN PRIMARY KEY DEFAULT TRUE CHECK (only_row),
    paused     BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

INSERT INTO queue_state (only_row) VALUES (TRUE);
