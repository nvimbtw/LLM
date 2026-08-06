-- Tokens the model forwarded to complete the run: the cost of the benchmark
-- itself, in the unit the model works in.
--
-- This is what the score is now plotted against. Training tokens (the column
-- above) described the model, not the run, and said nothing about what the
-- measurement cost — two runs of the same model over different corpora landed on
-- exactly the same x. `eval_tokens` moves with the work actually done, and unlike
-- the wall clock it means the same thing on any machine.
--
-- Lifted out of the report JSON on completion (`llm::eval::Report::tokens_used`,
-- and its markup/assistant twins) for the same reason `score` is: the history list
-- deliberately does not ship the report, and a chart should not have to parse one.
ALTER TABLE benchmark_runs ADD COLUMN eval_tokens BIGINT;

-- No backfill. Runs that predate the counter never measured this, and there is
-- nothing in their stored report to reconstruct it from — the per-stage counts
-- there cover only the tokens each stage chose to report, not the forwards it made.
-- Those rows stay null and the chart leaves them off, which is the same policy it
-- already applies to a missing `train_tokens`: an absent measurement is not a zero.
