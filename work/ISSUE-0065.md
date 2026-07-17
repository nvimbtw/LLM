---
id: ISSUE-0065
type: issue
title: "Rename db::user::User to db::user::UserRow"
scope: backend
priority: P2
status: todo
rule: R21
location: backend/src/db/user.rs:12
blocked-by: []
blocks: [EPIC-0064]
---

## Description

`db::user::User` stutters — the module `user` and the struct `User` repeat. Per **R21**, rename to `UserRow` so the path reads `db::user::UserRow`, making explicit that this is a database row struct (matching the convention in `usage.rs`).

### Scope of change
- `backend/src/db/user.rs`: rename `pub struct User` → `pub struct UserRow`, update all references within the file
- `backend/src/routes/auth.rs`: `use crate::db::user::User;` and any field access on `User` instances
- `backend/src/`: any other file importing or referencing `db::user::User`
- `backend/src/routes/user.rs`: references

### Shape of the fix
1. Rename struct in `user.rs`: `pub struct UserRow`
2. Update all `use` imports from `db::user::User` to `db::user::UserRow`
3. Update any pattern matches or destructuring of `User` structs

### Verification
- `cargo build` passes with zero warnings
- No references to `db::user::User` remain (except type alias or re-export if any)

## Comments
_No comments yet._
