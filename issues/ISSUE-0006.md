---
id: ISSUE-0006
title: Split run_training_loop and train_epoch — both exceed 60 lines (R1)
scope: backend/llm
priority: P1
status: todo
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
