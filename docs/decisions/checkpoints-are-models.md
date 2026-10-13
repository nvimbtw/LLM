# A checkpoint is a model, not a lesser artifact

Update 2026-10-09: ownership and retention portions are superseded by
[Execution and ownership](execution-and-ownership.md). The original rationale
below records the design at its date; current contracts are in the module docs.


- **Date**: 2026-08-29
- **Status**: Implemented

## Context

A run saved `checkpoints/epochNNN.pt` under the model it was training: weights
alone, one file an epoch, named by no row and reachable by nothing. Three things
people wanted from a long run were all impossible against it.

- **Go back.** A run that diverged at epoch 12 had nothing usable from epoch 8 —
  the file existed, but chatting with it, benchmarking it or fine-tuning from it
  all take a model id, and it had none.
- **Carry on.** Weights are not a training state. AdamW's two moments are two
  thirds of what a step depends on, and restarting cold throws away the
  second-moment estimates the run spent thousands of steps building.
- **Survive an interruption.** A cancelled or crashed run left nothing at all,
  because the only registered output was written after the last epoch returned.

The question was not whether to keep more, but **what a kept thing is**.

## Decision

**A checkpoint is a model in every respect a model is defined by**: its own
uuid, its own `artifacts/models/<uuid>/` laid out exactly like a finished
model's — `model.pt`, `tokenizer.json`, `meta.json` — plus a `train_state.pt`,
and its own `models` row.

Everything else follows from that sameness rather than being built:

- Chat, benchmarking, the attention readout and SFT's `base_model` already take
  a model id and a directory. **None of them learned what a checkpoint is**, and
  none of them has a second code path to keep in step with the first.
- `resume_from` does not branch on whether it was handed a finished model or a
  checkpoint. They are the same directory shape; the difference is only whether
  a `train_state.pt` is in it, and a plain weights file read as a state is a
  continuation with cold moments that reports itself as one.
- A run that never finishes still leaves usable models behind, because each
  checkpoint's row is written the moment its frame arrives rather than at the
  end.

Four nullable columns carry it (`run_id`, `parent_id`, `checkpoint_step`,
`checkpoint_index`), and **`checkpoint_index IS NOT NULL` is the sole definition
of "intermediate"** — deliberately not a boolean beside it, which is a second
source of truth that can disagree with the first. The mechanism and the resume
semantics are [`../llm-engine/checkpoints.md`](../llm-engine/checkpoints.md); the
columns are in [`../llm-engine/README.md`](../llm-engine/README.md#schema).

## What this costs

**Disk, honestly.** A `train_state.pt` is the weights again plus AdamW's two
moments per parameter — about **3x a `model.pt`** — and the directory keeps the
plain weights too, so a checkpoint costs roughly **4x a finished model** and the
ten of them a default run leaves cost roughly **40x**. Against a 100 MB
`model.pt` that is 400 MB a checkpoint and ~4 GB for the run, where the same run
used to leave 100 MB. It scales linearly with the model, which is why the number
worth remembering is the multiple.

Three things bound it, and they were part of taking the decision rather than
afterthoughts:

- The count is scaled to the run, not to its step count, so **the bill does not
  grow with run length** — a ten-hour run and a ten-minute one both leave about
  ten.
- `TrainConfig.checkpoint_target` is a disk budget as much as a cadence, and `0`
  turns intermediates off entirely.
- Deleting a model takes its checkpoints with it, rows and directories both.

**Ten extra rows per run in a table the product lists everywhere.** Paid for by
`GET /models` excluding intermediates unless asked, on *both* halves of its
disk/row join — filtering only the rows would have let every checkpoint back in
through the directory listing, reported as a model trained outside the queue.

**Nothing prunes old checkpoints.** They live until their parent is deleted.
That is a real gap and the honest place to revisit this record: a retention rule
(keep the last N, keep the best by eval loss) is a change to the cadence and the
delete path, not to what a checkpoint *is*.

## What was rejected

**Checkpoints as a lesser artifact under the parent's directory**, with rows
keyed `(parent_id, step)` and no uuid, no tokenizer copy and no `meta.json`.
This is the cheaper design on disk — roughly a third less — and it was the
starting point. It fails on the thing the feature exists for: every consumer of
a model would grow "…or a `(parent, step)` pair", in the loader, the benchmark
selection, the SFT base, the attention route and every list the frontend builds
from `/models`. The union of those branches is more code, in more places, than
the copies it saves, and each branch is a place the two kinds of artifact can
drift. It is also structurally wrong about interruption: a checkpoint filed
under a parent directory that a crashed run never created belongs to nothing.

**Foreign keys on `run_id` and `parent_id`.** Rejected for two *different*
reasons, which is why neither column implies anything about the other:

- `run_id` is not a key because **run rows are prunable and the grouping has to
  outlive them** — the same reason `runs.ref_id` is a bare uuid. A cleanup of
  old run history should not cascade into deleting models or null a column
  people group by.
- `parent_id` is not a key because **the parent row does not exist yet when the
  first checkpoint lands**. The constraint would refuse the very insert it is
  meant to describe. Writing an empty parent row up front to satisfy it was
  considered and is worse: it puts a model with no weights and no result in the
  table, so every listing then has to filter a second kind of not-quite-a-model.
  Deferring the constraint does not reach it either — Postgres defers to the end
  of a transaction, and the checkpoint row is committed in a different one,
  potentially hours earlier.

**A fixed step interval** — "every 500 steps", the obvious cadence. Run lengths
here span orders of magnitude, so one number is wrong at both ends: two
checkpoints on a short run and several hundred on a long one, which at ~4x a
model each is hundreds of gigabytes and a model list nobody can read. Scaling
the interval to the run's length bounds both. The floor of **one per epoch**
exists because the behaviour being replaced was per-epoch and an epoch boundary
is what a person reaches back to by name; without it a forty-epoch run keeping
ten would have discarded three quarters of the points anyone asks for. Snapping
the interval **up** to a multiple of `report_every` is what keeps a checkpoint
on a step that is already publishing a frame, so nothing has to invent one and
`Progress.checkpoint_path` keeps meaning what it always meant.

**Keeping weights-only checkpoints, and recording nothing.** The smallest
possible change: leave the per-epoch `.pt` files, add no columns, add no rows.
It buys nothing. Weights cannot be resumed from without the moments, and a file
no row names is unreachable from the product — which is precisely the state this
record exists to leave. The mirror of it, **a state file with no row**, fails
the same way for the same reason: nothing can list what nothing indexes.

**Bumping `CHECKPOINT_VERSION` for the added keys.** They change nothing about
what an existing v1 file means, and a bump would have made every model already
on disk unreadable to the loader that now has to read both shapes.
