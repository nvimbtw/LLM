-- How many tokens the benchmarked model was trained on, so the page can plot the
-- scaling curve (score against training tokens) without reading every model's
-- config JSON back out.
--
-- Snapshotted onto the run rather than joined from `models` at read time, for the
-- same reason `model_name` is denormalized here: retraining a model upserts its
-- row in place, so a joined count would retroactively relabel every past run with
-- the corpus size of the *newest* training run and quietly break the curve.
ALTER TABLE benchmark_runs ADD COLUMN train_tokens BIGINT;

-- Backfill from each model's current config. Runs that predate this column can
-- only be attributed the model's present corpus size — right for a model trained
-- once, approximate for one that has since been retrained, and either way better
-- than dropping the run off the chart entirely.
--
-- `corpus_len` is what a pretrain run records; `total_tokens` is the SFT
-- equivalent (see DatasetStats / SftDatasetStats in training.rs).
UPDATE benchmark_runs b
SET train_tokens = COALESCE(
        (m.parameters #>> '{dataset,corpus_len}')::bigint,
        (m.parameters #>> '{dataset,total_tokens}')::bigint
    )
FROM models m
WHERE m.id_model = b.id_model;
