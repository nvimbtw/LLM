---
id: ISSUE-0066
type: issue
title: "Rename db::chat::Chat to db::chat::ChatRow"
scope: backend
priority: P2
status: todo
rule: R21
location: backend/src/db/chat.rs:5
blocked-by: []
blocks: [EPIC-0064]
---

## Description

`db::chat::Chat` stutters — the module `chat` and the struct `Chat` repeat. Per **R21**, rename to `ChatRow` so the path reads `db::chat::ChatRow`.

### Scope of change
- `backend/src/db/chat.rs`: rename struct, update internal references
- `backend/src/routes/chat.rs`: update import and usage
- Any other file referencing `db::chat::Chat`

### Shape of the fix
1. Rename struct in `chat.rs`: `pub struct ChatRow`
2. Update imports and references in consuming files

### Verification
- `cargo build` passes with zero warnings
- No references to `db::chat::Chat` remain

## Comments
_No comments yet._
