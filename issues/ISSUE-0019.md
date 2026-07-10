---
id: ISSUE-0019
title: Duplicated corpus loading and serialization reading logic (R6)
scope: backend
priority: P2
status: todo
rule: R6
location: backend/src/llm/src/train.rs:66-69,124-128 backend/src/llm/src/training/transformer.rs:48-72,1084-1104,1059-1071
blocked-by: []
blocks: []
---

## Description
Three instances of duplicated logic: (1) `train()` and `continue_train()` in
train.rs independently implement identical `fs::read` + `chunks_exact(4)` +
`u32::from_le_bytes` corpus loading; (2) `read_model_header` and `read_header` in
transformer.rs both read magic/version/six i64 fields — `read_model_header` should
delegate to `read_header`; (3) the `f32::from_le_bytes(c.try_into().unwrap())`
byte-conversion pattern from `chunks_exact(4)` repeats 7 times across transformer.rs
(lines 1059, 1071, 1687, 1701, 1716, 1745, 1798) — should be a shared helper.

## Comments
_No comments yet._
