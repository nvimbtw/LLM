---
id: ISSUE-0006
type: issue
title: Split run_training_loop and train_epoch — both exceed 60 lines (R1)
scope: backend/llm
priority: P1
status: done
rule: R1
location: backend/src/llm/src/train.rs:194-388
blocked-by: []
blocks: []
---

## Description
`run_training_loop` is 87 lines and `train_epoch` (extracted from the original
214-line `run_training_loop`) is 107 lines — both still over the R1 60-line limit.
Extract the progress callback construction block from `run_training_loop` (lines
344-362) and split `train_epoch` into named phases (e.g. `process_chunk`,
`step_optimizer`, `report_progress`).

## Comments
### Verifier · migrated · bounce
BOUNCED (2nd) — `run_training_loop` unchanged at 87 lines, `train_epoch` at 107
lines also exceeds limit. Neither was addressed since previous bounce. (This issue
has been bounced twice; the first-round bounce text was not preserved by the old
single-note `todo.md` format.)

### Builder · 2026-07-10
Split both functions; each body is now ~40 lines (well under R1's 60).

Root cause of the length: both functions inlined a full `TrainingProgress`
frame-build (~20 lines each), and `train_epoch` additionally carried the
optimizer-step block and eight loose accumulator variables (`total_loss`,
`n_chunks`, `min/max_loss`, `win_total/n/min/max`). Extractions:

- New `LossStats { total, n, min, max }` struct with `record`/`avg` — collapses
  the eight accumulators into two values (`epoch_stats`, `window`). Sentinel
  behaviour preserved: `min/max` stay ±INFINITY and `avg()` is NaN until the
  first `record`, matching the old `0/0` and `f32::INFINITY` seeds exactly.
- `step_optimizer(ws, layers, lr, adam_step)` — the clip + warmup-Adam + zero +
  grad-norm-log block, lifted verbatim.
- `report_chunk_progress(...)` — the mid-epoch progress frame.
- `report_epoch_progress(...)` — the end-of-epoch frame (was inline in
  `run_training_loop`, the block the issue called out at old lines 344-362).
- `should_stop_early(avg_loss, epoch, &mut best_loss, &mut epochs_since_best)`
  — the early-stopping bookkeeping + break decision.

Behaviour is unchanged — the extracted blocks are the originals, and the
`LossStats` arithmetic reproduces the prior values (`avg = total/n`, running
min/max). The two report helpers carry `#[allow(clippy::too_many_arguments)]`
with a justifying comment (consistent with `train_epoch`'s existing allow) so R7
stays clean.

Left out of scope on purpose: the `n_chunks % 10` progress-interval literal
(ISSUE-0029 / ISSUE-0020 own that constant) and the `% 50` grad-log literal
(ISSUE-0020) — kept verbatim to avoid colliding with those issues.

Verified: `cargo build -p llm` clean (zero warnings); `cargo clippy -p llm`
clean. Both function bodies now ~40 lines.

### Verifier · 2026-07-10 · pass
Checked the following:
1. **`train_epoch` body** (lines 241–281): 41 lines — well under R1's ~60 line limit.
2. **`run_training_loop` body** (lines 354–398): 45 lines — well under R1's ~60 line limit.
3. **Extracted helpers confirmed present:** `LossStats` (struct + impl, lines 160–186),
   `step_optimizer` (286–302), `report_chunk_progress` (307–339),
   `report_epoch_progress` (404–434), `should_stop_early` (438–459). All are
   single-purpose, named stages as required by R1.
4. **Build clean:** `cargo build -p llm` completed with zero warnings and zero errors
   (dev profile, finished 0.18s).
5. **Behaviour preserved:** The extracted blocks match the originals (clip, warmup-Adam,
   zero-accum, grad-norm-log in `step_optimizer`; the progress frames in the report
   helpers; early-stopping bookkeeping in `should_stop_early`). The `LossStats` min/max
   sentinels (±INFINITY) and NaN `avg()` match the pre-refactor semantics.
R1 violation resolved. Passing.
