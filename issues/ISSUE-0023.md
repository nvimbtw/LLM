---
id: ISSUE-0023
title: Dropped Results and unwraps without justification across backend (R3, R11)
scope: backend
priority: P1
status: todo
rule: R3, R11
location: backend/src/chat.rs:153-159,176,183,389,392 backend/src/jobs.rs:88,95,98,124,138,143,150,208,212,263,264,274,282,285,286,312,329,366,370,388,394,398 backend/src/routes/training.rs:105-210
blocked-by: []
blocks: []
---

## Description
Pervasive `let _ = session.text(...)` / `let _ = session.close(...)` across chat.rs
and training.rs drop `Result` without comment. ~20 `.unwrap()` calls on
`Mutex::lock()` in jobs.rs have no poison-justification comment (despite
`catch_unwind` protecting the worker). `let _ = tok_tx.send(...)` and `let _ =
done_tx.send(...)` at chat.rs:389,392 drop send failures without note. Every dropped
`Result` or `unwrap` needs either a comment explaining why failure is
impossible/ignorable, or proper error propagation.

## Comments
_No comments yet._
