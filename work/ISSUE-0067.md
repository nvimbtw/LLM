---
id: ISSUE-0067
type: issue
title: "Rename db::message::Message to db::message::MessageRow"
scope: backend
priority: P2
status: todo
rule: R21
location: backend/src/db/message.rs:5
blocked-by: []
blocks: [EPIC-0064]
---

## Description

`db::message::Message` stutters — the module `message` and the struct `Message` repeat. Per **R21**, rename to `MessageRow` so the path reads `db::message::MessageRow`.

### Scope of change
- `backend/src/db/message.rs`: rename struct, update internal references
- `backend/src/routes/chat.rs`: update import and usage
- Any other file referencing `db::message::Message`

### Shape of the fix
1. Rename struct in `message.rs`: `pub struct MessageRow`
2. Update imports and references in consuming files

### Verification
- `cargo build` passes with zero warnings
- No references to `db::message::Message` remain

## Comments
_No comments yet._
