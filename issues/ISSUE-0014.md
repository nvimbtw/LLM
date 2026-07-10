---
id: ISSUE-0014
title: insert_user error response leaks internal error details
scope: backend
priority: P2
status: done
rule: none
location: backend/src/routes/user.rs:50-52
blocked-by: []
blocks: []
---

## Description
When `crypto::password::hash_password` fails, the handler returns the error details
to the client via `format!("failed to hash password: {}", e)` instead of a generic
server-error message. This violates the CLAUDE.md convention: "Never return raw
error internals in the HTTP response body". Log server-side (`eprintln!`) and
return a generic `"failed to create user"` message instead.

## Comments
### Verifier · migrated · pass
Confirmed line 50 now uses `eprintln!("password hashing failed: {}", e);` for
server-side logging, and line 52 returns generic `"failed to create user"` to the
client.
