---
id: ISSUE-0024
type: issue
title: training_session spans ~110 lines (R1)
scope: backend
priority: P3
status: todo
rule: R1
location: backend/src/routes/training.rs:91-211
blocked-by: []
blocks: []
---

## Description
`training_session` body is ~110 lines covering three distinct phases: authenticate
client, attach to job, and stream snapshot+live frames. Each phase is separated by
`// ----` comment blocks — a signal per R13 they should be extracted functions.
Extract `authenticate_client`, `attach_to_job`, and `stream_job_frames` as named
helpers.

## Comments
_No comments yet._
