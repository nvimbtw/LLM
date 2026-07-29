-- Synthetic SFT generation, and the lane that lets it run without idling the GPU.
--
-- **Why a lane rather than another kind in the same line.** Every job so far is
-- FFI into CUDA, so one runner draining one queue *is* the resource constraint —
-- the line and the device are the same thing. Generating SFT data is not: it is a
-- sequence of HTTP calls to somebody else's model, minutes long, using no local
-- compute at all. Put in the existing queue it would hold the single runner while
-- waiting on a socket, so a 200-conversation run means twenty minutes of a
-- completely idle GPU and a chat box that does not answer.
--
-- So a job now declares which runner it belongs to. `device` is the GPU line,
-- unchanged and still strictly one at a time. `network` is its own line with its
-- own runner, and the two do not interact: a generation run and a training run
-- proceed side by side because they contend for nothing.
--
-- This is orthogonal to `background`, which is a *priority band within a lane*
-- (see 20260727000000_create_cycles.sql). Lane says which runner; band says who
-- goes first in front of that runner. Ordering everywhere is
-- (lane, background, queue_rank).

-- ALTER TYPE ... ADD VALUE cannot run inside a transaction block on older
-- PostgreSQL, and sqlx wraps each migration in one. IF NOT EXISTS makes this
-- re-runnable; on PG 12+ the in-transaction restriction is lifted for a type that
-- was not created in the same transaction, which is the case here. The new value
-- is deliberately not *used* anywhere below — that is still forbidden until this
-- transaction commits.
ALTER TYPE job_type ADD VALUE IF NOT EXISTS 'synth';

CREATE TYPE job_lane AS ENUM (
    -- Work that touches the GPU: training, benchmarks, chat turns, generation and
    -- labelling passes. One runner, one at a time.
    'device',
    -- Work that waits on the network and nothing else. Its own runner, so it can
    -- never be the reason a chat turn is waiting.
    'network'
);

ALTER TABLE jobs ADD COLUMN lane job_lane NOT NULL DEFAULT 'device';

-- Replaces the queue index from the cycles migration: the runners' "what next"
-- read now filters by lane before ordering by band and rank.
DROP INDEX IF EXISTS jobs_queue_idx;
CREATE INDEX jobs_queue_idx ON jobs (lane, background, queue_rank)
    WHERE status IN ('queued', 'running', 'cancelling');
