---
id: ISSUE-0029
title: Training progress sent to frontend too frequently, overloading graphs
scope: cross
priority: P1
status: todo
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
_No comments yet._
