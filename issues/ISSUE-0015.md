---
id: ISSUE-0015
title: cargo build emits 4 warnings (R7)
scope: backend
priority: P1
status: todo
rule: R7
location: backend/src/db/message.rs:7,12 backend/src/extractors.rs:85 backend/src/routes/user.rs:57 backend/src/main.rs:75
blocked-by: []
blocks: []
---

## Description
`cargo build` (and `cargo clippy`) produce 4 warnings: 2 `dead_code` on
`Message.id_chat`, `Message.id_model` (message.rs) and `AdminUser.claims`
(extractors.rs); 2 `clippy::needless_borrow` on `&pool.get_ref()` (user.rs:57) and
`&pool` (main.rs:75). R7 requires a clean build. Either fix or add justified
`#[allow]` with a comment.

## Comments
_No comments yet._
