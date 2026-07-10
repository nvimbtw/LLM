---
id: ISSUE-0025
title: run_job and finish do multiple things separated by block comments (R13)
scope: backend
priority: P3
status: todo
rule: R13
location: backend/src/jobs.rs:187-250,333-389
blocked-by: []
blocks: []
---

## Description
`run_job` has `// ---- Phase 2: train ----` comment splitting it into tokenize and
train phases — the comment itself signals they should be extracted as
`tokenize_phase()` and `training_phase()`. `finish` does four things (write
meta.json, register model, update status, send frame) and the generic name `finish`
obscures the side effects. Extract each phase into a named function.

## Comments
_No comments yet._
