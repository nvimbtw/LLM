-- Removes synthetic SFT generation.
--
-- It called a hosted model over HTTP and never touched the GPU, which is why it
-- had its own `network` lane. Nothing consumes what it produced any more, and
-- when generation returns it will belong to the data module rather than to the
-- queue. Forward-only, matching the cycle removal before it.
--
-- `lane` and `background` are deliberately left in place: with synth gone the
-- network lane has no producer and the background band has had none since the
-- cycle was dropped, but both columns disappear with this table when the engine
-- takes ownership of the queue. Dropping them here would be a migration
-- superseded a few commits later.

-- Compared as text, not as the enum: a database whose `job_type` never carried
-- the value cannot parse the literal, and this has to be a no-op there rather
-- than a migration that refuses to run.
DELETE FROM jobs WHERE kind::text = 'synth';

-- Postgres cannot drop a value from an enum in place, so the type is rebuilt.
ALTER TYPE job_type RENAME TO job_type_old;
CREATE TYPE job_type AS ENUM ('train', 'benchmark', 'chat');

ALTER TABLE jobs
    ALTER COLUMN kind TYPE job_type USING kind::text::job_type;

DROP TYPE job_type_old;
