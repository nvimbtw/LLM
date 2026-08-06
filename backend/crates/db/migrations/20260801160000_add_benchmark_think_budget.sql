-- Benchmarking a model at a thinking budget, and connecting the runs that sweep it.
--
-- A run may now render every held-out conversation under a forced `<|think:*|>`
-- marker instead of the one each record was written with, which is the same act the
-- chat composer performs when you pick a budget — so the run has to record which
-- marker was in force. Two scores are only comparable at the same budget, exactly as
-- they are only comparable at the same `suite_version`.
--
-- Null means "each conversation kept its own budget": every run before this one, and
-- every run with the switch off. Not a fourth value — the absence of a forced marker,
-- matching how `ThinkBudget` is already stored and sent everywhere else.
ALTER TABLE benchmark_runs ADD COLUMN think_budget TEXT;

-- The sweep a run belongs to. Three runs queued together by the split switch share
-- one id, which is what lets the chart draw them as one connected series rather than
-- three unrelated dots that happen to be the same model.
--
-- Deliberately not derived from (model, suite, timestamp): sweeping the same model
-- twice is two lines, and only the enqueue that created them knows which is which.
-- Null for any run queued on its own, which is then a series of one and draws as a
-- lone dot.
ALTER TABLE benchmark_runs ADD COLUMN id_series UUID;

-- How many held-out conversations the run scored. Lifted out of the report on
-- completion beside `score` and `eval_tokens`, for the same reason: the history list
-- does not ship the report, and the chart needs this to mark a point as provisional.
--
-- It is the count a budget sweep actually varies over, and it is currently *small* —
-- the held-out set runs 15 low, 15 med, 3 high — so a point drawn from it can rest on
-- far less evidence than a neighbouring one and look identical without this.
ALTER TABLE benchmark_runs ADD COLUMN eval_conversations BIGINT;
