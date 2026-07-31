-- Two benchmarks instead of one.
--
-- The language suite measures whether the model can model English; the markup suite
-- measures whether it uses its own special tokens correctly. They are queued, run and
-- scored independently, because a supervised fine-tune is expected to move them in
-- opposite directions and one blended number cannot say so.
--
-- A discriminator column rather than a second table: a run is a run — it has a model,
-- a status, a score, a report and a duration whichever suite produced it, and every
-- query, the history list, the delete path and `fail_orphaned` would otherwise be
-- written twice.
CREATE TYPE benchmark_suite AS ENUM ('language', 'markup');

-- The default backfills every existing row as 'language', which is what they are.
ALTER TABLE benchmark_runs
    ADD COLUMN suite benchmark_suite NOT NULL DEFAULT 'language';

-- Which version of the suite produced the score. Bumped whenever a scoring constant,
-- the pair set, the prompt set or the corpus split changes — all of which move the
-- number without the model having changed.
--
-- Existing rows are version 1: the pre-rebuild composite, whose corpus and variety
-- terms were pinned at their bounds. Those scores are on a different scale to
-- anything produced from now on and must not be plotted on the same trend line or
-- compared by `jobs::cycle`, which is what this column exists to make detectable.
ALTER TABLE benchmark_runs
    ADD COLUMN suite_version INTEGER NOT NULL DEFAULT 1;

-- The lookup is now per (model, suite): "the latest language score for this model" is
-- what the post-train cycle asks, and it must not see a markup run.
DROP INDEX IF EXISTS benchmark_runs_model_idx;
CREATE INDEX benchmark_runs_model_idx
    ON benchmark_runs (id_model, suite, created_at DESC);
