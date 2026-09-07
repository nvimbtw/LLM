# Automatic math practice

`RlConfig.curriculum` opts a math run into automatic progression. It is absent
on ordinary RL runs. The gym's catalogue, reached through `/internal/gyms`,
owns task ids, levels and prerequisites. The engine owns mastery and sampling;
it never receives a solution before the episode is graded.

`train/curriculum.py` is the pure state machine; `orchestrate/practice.py`
conducts exams with the existing rollout path and no backward pass. The RL
pipeline invokes placement before training, checks every `check_every`
iterations, and fixed tests before and after the run. Context behavior is
unchanged.

## The gates

Each selected task type must include its prerequisites. Initially the reachable
easy tasks are checked. Two consecutive checks at or above `promote_at` (90%)
mark a rung mastered and unlock its dependents. Two checks below `retain_at`
(75%) return it to practice and invalidate dependent mastery. Each check uses
100 distinct problems per type by default; the API allows 10–100 for small
experiments. A partial or failed exam must never promote a model.

Sampling mass is 60% current practice, 30% mastered review, 10% the next
reachable stretch; empty categories are omitted and the remainder renormalized.
Within a category, types are sampled uniformly. In automatic mode these weights
replace the manually configured task weights. Two checks below 5% mark a skill
`needs_sft`. It remains eligible for practice but the UI recommends demonstrations;
no extra SFT job is started automatically. All-fail binary-reward groups have
zero policy-gradient signal, so harder questions alone cannot unstick them.

## Validation is not the test

Validation changes curriculum decisions. Its seed range advances with a saved
check counter; the gym uses a separate validation stream and wording. The fixed
test uses `holdout_base`, every selected rung and `test_samples` (16 per rung
by default). It is measured at both ends and never read by the mastery policy.
Both use greedy decoding, a single response and no tools. Generation is
micro-batched so a 100-question exam does not allocate 100 KV caches at once.
Cancellation is checked between micro-batches and task types.

An exam measures correctness, invalid final answers and truncated responses.
The test roster stays fixed even as practice gets harder, so its before/after
accuracy is comparable within a run. Small test samples have substantial
uncertainty. Elementary facts may overlap between splits: see
[math tasks](../gyms/math.md). This is not a claim of general reasoning transfer.

## What survives

Finished models and intermediate checkpoints store `result.curriculum` in
`meta.json`; checkpoints also carry it in `train_state.pt` extras. It includes
the catalogue, schedule, mastery counters, validation cycle, training group
cursor and promotion/review events. Starting a new adaptive run from that model
or checkpoint inherits the lessons when the catalogue and settings match.
A mismatch is refused rather than silently interpreting old mastery under new
rules. This continues the curriculum, not the previous optimizer: RL still
starts a fresh optimizer and anchors KL to the chosen base model.

The report publishes per-rung mastery and decisions, with the measured numerator
and denominator behind each promotion. `result.test_before` and `test_after`
hold the fixed scores. Exam batches stay under `kind: rl` on the gym side, so
they do not bury manually requested evaluations. Finished job pages read their
report from the saved result, and live pages enrich it from progress frames.
