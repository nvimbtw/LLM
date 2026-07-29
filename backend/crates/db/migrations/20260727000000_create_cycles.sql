-- Post-training cycles: the layer above the queue.
--
-- **Why this is not a job kind.** A cycle spawns the jobs it is made of and waits
-- for each to finish. If it were itself a job it would hold the single GPU slot
-- while its children queued behind it, waiting for the slot the parent is holding
-- until they complete — a deadlock, not a design smell. So a cycle lives beside the
-- queue: it enqueues work and reacts to completions, and never touches the device.
--
-- The other reason it is a table rather than an in-memory driver: a cycle runs for
-- hours across a dozen jobs, and the whole point is that the path lives here
-- instead of in someone's head. A restart resumes it; a page load can show it.

CREATE TYPE cycle_status AS ENUM (
    'running',
    -- A child job was cancelled, or a person pressed pause. Resumable.
    'paused',
    -- Stopped on a guardrail or a failure. Not resumable without a decision —
    -- `halt_reason` says which.
    'halted',
    -- Reached max_rounds with nothing tripped.
    'done'
);

-- Where a round is. `deciding` holds the GPU for nothing — it is the accept /
-- roll back / halt step, which is pure arithmetic over the round's results.
CREATE TYPE cycle_stage AS ENUM (
    'generating',
    'selftrain',
    'preference',
    'benchmarking',
    'deciding'
);

CREATE TABLE cycles (
    id_cycle    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label       TEXT NOT NULL,
    status      cycle_status NOT NULL DEFAULT 'running',
    stage       cycle_stage  NOT NULL DEFAULT 'generating',
    round       INTEGER NOT NULL DEFAULT 1,
    max_rounds  INTEGER NOT NULL,

    -- Where the cycle started. Kept for the whole run: "how far has this drifted
    -- from where it began" is the question a KL anchor cannot answer across rounds,
    -- because each round re-anchors to the previous one.
    id_model_start   UUID REFERENCES models(id_model) ON DELETE SET NULL,
    -- What the next round will train from.
    id_model_current UUID REFERENCES models(id_model) ON DELETE SET NULL,
    -- The best-scoring model produced so far, and its score. This is the cycle's
    -- actual output: when a round fails to improve, the answer is this, not
    -- whatever the last round happened to produce.
    id_model_best    UUID REFERENCES models(id_model) ON DELETE SET NULL,
    best_score       DOUBLE PRECISION,

    -- The job the current stage is waiting on. Denormalized onto the cycle rather
    -- than found by querying jobs, so advancing is one read: a driver that had to
    -- search for its own child on every tick would be one bad WHERE clause away
    -- from advancing on somebody else's job.
    id_job_current   UUID REFERENCES jobs(id_job) ON DELETE SET NULL,

    -- Every knob the run uses: candidates per prompt, prompt count, temperature,
    -- learning rates, beta, and the minimum score gain that counts as improvement.
    -- Stored rather than read from constants so a cycle records the rules it ran
    -- under — two cycles six weeks apart are otherwise not comparable.
    config      JSONB NOT NULL,
    halt_reason TEXT,

    id_user     UUID REFERENCES users(id_user) ON DELETE SET NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ
);

-- The driver's hot read: which cycles need advancing. Partial, because finished
-- cycles accumulate and are never scanned by the loop.
CREATE INDEX cycles_active_idx ON cycles (created_at)
    WHERE status IN ('running', 'paused');

-- One row per round — the "generation" record: what it trained on, what it
-- produced, and whether it was kept.
CREATE TABLE cycle_rounds (
    id_round   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_cycle   UUID NOT NULL REFERENCES cycles(id_cycle) ON DELETE CASCADE,
    round      INTEGER NOT NULL,

    -- Catalogue size either side of the round's generation stage. The pair is what
    -- makes yield computable — "how much of what this model wrote cleared the bar"
    -- — which is the loop's actual progress metric, and it is a delta, not a total.
    approved_before BIGINT,
    rejected_before BIGINT,
    approved_after  BIGINT,
    rejected_after  BIGINT,

    id_model_out UUID REFERENCES models(id_model) ON DELETE SET NULL,
    score        DOUBLE PRECISION,
    -- accepted / no_improvement / skipped_preference / failed — why the round ended
    -- as it did, in a word, with `note` carrying the sentence.
    verdict      TEXT,
    note         TEXT,

    started_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    finished_at TIMESTAMPTZ,

    -- A round number belongs to exactly one row of its cycle. Without this, a
    -- driver that ticked twice on the same event would quietly record two.
    UNIQUE (id_cycle, round)
);

CREATE INDEX cycle_rounds_cycle_idx ON cycle_rounds (id_cycle, round);

-- Which cycle a job belongs to, and which stage it is. Nullable: jobs queued by
-- hand have no cycle, and that stays the common case.
ALTER TABLE jobs ADD COLUMN id_cycle UUID REFERENCES cycles(id_cycle) ON DELETE SET NULL;
ALTER TABLE jobs ADD COLUMN cycle_role TEXT;

-- **The background band.** A cycle occupies the GPU for hours, and with one strictly
-- ordered queue that means every chat turn sent while it runs waits behind a
-- multi-minute generation pass. Rather than a priority number (which invites
-- arithmetic nobody can predict), jobs sit in one of two bands: foreground work
-- runs first, background work fills the gaps. Ordering everywhere becomes
-- (background, queue_rank), and ranks are assigned within a band, so the two never
-- interleave and dragging a job never moves it across the divide.
--
-- Cycle-spawned jobs are the only background work today. A cycle therefore keeps
-- the machine busy without ever making it unusable.
ALTER TABLE jobs ADD COLUMN background BOOLEAN NOT NULL DEFAULT FALSE;

-- Replaces the queue index: the runner's "what next" read is now ordered by the
-- band first.
DROP INDEX IF EXISTS jobs_queue_idx;
CREATE INDEX jobs_queue_idx ON jobs (background, queue_rank)
    WHERE status IN ('queued', 'running', 'cancelling');
