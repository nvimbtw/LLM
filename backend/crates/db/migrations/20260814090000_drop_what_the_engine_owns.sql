-- The engine owns models, conversations, runs and scores now; these tables are
-- the copies this side stopped reading. Their rows were copied first — see
-- engine/scripts/import_from_backend.py — and this is the half that cannot be
-- undone, so it is deliberately a separate step from the copy.
--
-- Usage goes with them: it is counted from the turns themselves now, so a
-- ledger recording the same numbers could only ever drift from them.

DROP TABLE IF EXISTS usage_events;
DROP TABLE IF EXISTS benchmark_runs;
DROP TABLE IF EXISTS messages;
DROP TABLE IF EXISTS chats;
DROP TABLE IF EXISTS models;
DROP TABLE IF EXISTS jobs;
DROP TABLE IF EXISTS queue_state;

DROP TYPE IF EXISTS benchmark_status;
DROP TYPE IF EXISTS benchmark_suite;
DROP TYPE IF EXISTS message_feedback;
DROP TYPE IF EXISTS job_type;
DROP TYPE IF EXISTS job_status;
DROP TYPE IF EXISTS job_lane;

-- Left behind by the cycle, which was dropped before the tables that used them.
DROP TYPE IF EXISTS cycle_stage;
DROP TYPE IF EXISTS cycle_status;
