---
id: ISSUE-0043
type: issue
title: "Training done/snapshot frame carries the trained model's id + name (enable Continue-to-SFT)"
scope: backend
priority: P2
status: verification
rule: none
location: backend/src/routes/training.rs backend/src/ws.rs backend/src/jobs.rs
blocked-by: []
blocks: [ISSUE-0049]
---

## Description

Part of the `/training` redesign epic (EPIC-0040). Enables the pretrain→SFT
**continuation** flow (child 0049): when a pretrain finishes, the frontend needs the
**id and name of the model that was just registered** so it can seed an SFT run with
that model as its base — without the user hunting for it in a dropdown.

### The gap

On completion the socket emits a `done` frame carrying only `{ type, status,
model_dir }` (see the client contract in `frontend/lib/training-socket.ts:14` and
its handler at `:175`). `model_dir` is an on-disk path — the frontend can't turn it
into a `base_model_id` for a subsequent SFT POST, which keys off the model's UUID.
The training worker *does* register the model via `db::model::upsert` (it has the
`id_model` and `name` — `db/model.rs:27`), so the identifiers exist server-side;
they're just not sent to the client.

### The change

Add the trained model's identifiers to the completion payload:

- **`done` frame** — include `id_model` (UUID) and `model_name` alongside the
  existing `status`/`model_dir`. Only meaningful on a *successful* pretrain/SFT that
  produced a registered model; on `cancelled`/`failed` these are absent/null.
- **`snapshot` frame** — a run that finished while the tab was closed is re-attached
  via the snapshot (`ws.rs`/`jobs.rs` catch-up). Carry the same `id_model`/
  `model_name` there so a reattached, already-`done` run can still offer
  "Continue to SFT". Thread the value through wherever the job's terminal result is
  stored so both the live `done` and the replayed `snapshot` read from one source.

### Where it flows

Trace the completion path: the worker finishes → registers the model
(`db::model::upsert`) → marks the job done → the WS pushes `done` and stores the
terminal state for later `snapshot`s. The id/name should be captured at
registration and stored on the job's terminal record so both frames can read it.
Keep the existing fields; this is purely additive to the wire payload (the epic's
non-goal list forbids changing the progress frames).

### Frontend contract note (for 0049, not this issue)

`TrainingHandlers.onDone` / the `TrainingSnapshot` type in `training-socket.ts` will
grow `id_model`/`model_name`; the type + handler update is part of frontend child
0049. This backend issue only has to *send* the fields.

### Notes for the Builder

- Confirm exactly where the model is registered on a successful run and where the
  job's terminal status is persisted for reattach — those are the two emission
  points. Match the existing serialization style of the `done`/`snapshot` frames.
- Additive + backward-compatible: an older frontend ignores the new fields; the new
  frontend treats their absence (cancel/fail) as "no continuation available".

## Comments
### Auditor · 2026-07-12
Child of EPIC-0040. Small, additive backend change with no algorithm impact —
kept separate from the frontend continuation flow (0049) so it can land and be
verified (inspect the `done` frame over the socket) on its own. Blocks 0049.

### Builder · 2026-07-12
Both emission points are in `jobs.rs`. The live `done` frame (in `finish`) already
had `config.storage_id` + `config.model_name` in scope — added them as `id_model` /
`model_name` alongside the existing `status`/`model_dir` (only built on the
success/`None`-error branch). The `snapshot_frame` now emits the same two fields,
gated on `state.status == JobStatus::Done` (else JSON `null`), reading from
`state.config` — so a reattached, already-finished pretrain carries them too, while
in-flight/cancelled/failed runs send null. Purely additive to the wire payload; the
progress frames are untouched (per the epic non-goal). `cargo build` + `cargo
clippy` clean. Frontend type/handler consumption is deferred to 0049 as planned.
