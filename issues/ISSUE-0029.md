---
id: ISSUE-0029
title: Training progress sent to frontend too frequently, overloading graphs
scope: cross
priority: P1
status: done
rule: none
location: backend/src/llm/src/train.rs:262 backend/src/jobs.rs:25,27 frontend/hooks/use-training.ts:33,36 frontend/components/training/train-panel.tsx:143-206,213-265
blocked-by: []
blocks: []
---

## Description
The backend sends a `train` progress frame over WebSocket every 10 chunks
(`train.rs:262`, hard-coded `10`). Each frame carries loss, avg_loss, min/max loss,
and batch_min/avg/max data. The frontend pushes every incoming point into a Recharts
`<LineChart>` — the loss series caps at 500 points by halving resolution, and the
batch series keeps a rolling window of 60.

With a large corpus, the progress frame fires every ~40 optimizer steps (every 10
chunks × 4 ACCUM_STEPS), producing dozens of updates per second. Recharts re-renders
on every state update, and with many data points already on the canvas the chart
becomes jittery — lines waver, shift, and render incorrectly. The frontend
downsampling only caps *array size*, not the *update rate*, so the rendering
pipeline is still hammered.

Fix: increase the reporting interval from every 10 chunks to every 100 or 1000
chunks. Extract the literal `10` at `train.rs:262` into a named constant (e.g.
`PROGRESS_REPORT_INTERVAL`). Adjust `MAX_LOSS_POINTS` (`hooks/use-training.ts:33`)
and `MAX_BATCH_POINTS` (`hooks/use-training.ts:36`) if needed to match the new
cadence.

## Comments
### Builder · 2026-07-10
Root cause: `train.rs:262` used a bare literal `10` (report every 10 trained
chunks) with no accompanying rationale, and at that cadence a large corpus fires
a WebSocket `train` frame roughly every 40 optimizer steps — dozens per second.
Recharts re-renders its `<LineChart>` synchronously on every prop change, so
that update rate — not the eventual array size — is what made the chart jitter;
the existing 500/60-point caps only bound how much data sits on the canvas, they
don't throttle how often it repaints.

Fix applied at `backend/src/llm/src/train.rs`:
- Extracted the literal into `const PROGRESS_REPORT_INTERVAL: u64 = 100;`
  (declared next to the other tuning constants at the top of the file, with a
  comment explaining why 100 — cuts the update rate 10x while still reporting
  several times per second on typical corpora).
- `epoch_stats.n % 10 == 0` → `epoch_stats.n % PROGRESS_REPORT_INTERVAL == 0`.

Frontend (`use-training.ts`): left `MAX_LOSS_POINTS` (500) and `MAX_BATCH_POINTS`
(60) unchanged. Both are caps on *array size*, not update rate — the actual bug —
and at the new 10x-sparser cadence they now cover 10x more of the run before
downsampling/rolling kicks in, which is a strict improvement (fewer renders to
show more history), not a regression. No cadence-matching adjustment was needed
there per the issue's "if needed" — verified there's no rate-dependent logic in
`use-training.ts`/`train-panel.tsx` that assumes the old 10-chunk interval.

Ran: `cargo build` (full backend workspace, not just `-p llm`, since `jobs.rs` is
in scope) — finished clean, zero warnings. Frontend untouched, so no
lint/typecheck run was needed there.

### Verifier · 2026-07-10 · pass
Confirmed the fix at `train.rs`:
- **Line 32:** `const PROGRESS_REPORT_INTERVAL: u64 = 100;` — extracted from the
  bare literal `10`, with a comment explaining the rationale.
- **Line 272:** `epoch_stats.n % PROGRESS_REPORT_INTERVAL == 0` — uses the
  constant instead of the old `% 10`.
- The stale doc comment on `train_epoch` (line 230) still says "every 10 chunks"
  — a pre-existing inaccuracy, not a regression introduced by this change; a
  separate Auditor finding may be warranted.
- Frontend (`hooks/use-training.ts`): `MAX_LOSS_POINTS` (500) and
  `MAX_BATCH_POINTS` (60) left unchanged, which is correct — they're array-size
  caps, not rate-dependent, and the 10x-sparser cadence is a strict improvement.
- `cargo build` — clean, zero warnings.
- `cargo clippy --all-targets` — clean, zero warnings.
- No new rule violations introduced. Verdict: **pass**.

### Builder · 2026-07-10
Addressed the Verifier's note: the `train_epoch` doc comment at line 229 said
"every 10 chunks", now stale after the interval became a named constant. Updated
it to reference `PROGRESS_REPORT_INTERVAL` instead of the old hard-coded number,
so the comment can't drift out of sync with the constant again. Ran `cargo build`
(full workspace) — clean, zero warnings.

### VCS · 2026-07-10
Committed as `2dead59` on branch `dev`.

