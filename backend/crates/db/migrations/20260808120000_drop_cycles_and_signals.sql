-- Removes the post-training cycle (autolabel -> RAFT -> KTO), the gathered
-- rating panel that fed it, and the local-GGUF `teach` job.
--
-- Forward-only rather than deleting the migrations that created these: the
-- files are history, and editing them would leave any database that has already
-- run them disagreeing with the schema this build expects.
--
-- `message_feedback` is deliberately NOT dropped. It was the seed for preference
-- tuning, but it is also a plain UI affordance — thumbs on a chat reply — and it
-- keeps working with nothing training on it.

-- `jobs` carries the round a job belonged to. The FK has to go before the table
-- it points at; the columns go with it, since nothing reads them once the cycle
-- driver is gone.
ALTER TABLE jobs DROP COLUMN IF EXISTS id_cycle;
ALTER TABLE jobs DROP COLUMN IF EXISTS cycle_role;

DROP TABLE IF EXISTS cycle_rounds;
DROP TABLE IF EXISTS cycles;
DROP TABLE IF EXISTS signal_samples;

-- The enum values go last: the tables above referenced them.
-- Postgres cannot drop a value from an enum in place, so the type is rebuilt.
ALTER TYPE job_type RENAME TO job_type_old;
CREATE TYPE job_type AS ENUM ('train', 'benchmark', 'chat', 'synth');

DELETE FROM jobs WHERE kind::text IN ('autolabel', 'gather', 'teach');

-- The column is `kind`; `job_type` is the enum type's name.
ALTER TABLE jobs
    ALTER COLUMN kind TYPE job_type USING kind::text::job_type;

DROP TYPE job_type_old;
