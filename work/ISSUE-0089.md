---
id: ISSUE-0089
type: issue
title: Training socket and presets bypass the central 401 handling added in TODO-0084
scope: frontend
priority: P3
status: todo
rule: R15
location: frontend/lib/training-socket.ts frontend/lib/training-presets.ts
blocked-by: []
blocks: []
---

## Description

TODO-0084 centralized 401 handling in `lib/api.ts` (`apiFetch` +
`handleSessionExpired`) and covered the chat socket, but the training surfaces
still use raw `authHeaders()` fetches / their own socket auth with no expiry
handling: an expired token on the training page surfaces as generic errors
instead of logging the user out. Flagged by the Verifier while passing
TODO-0084.

Shape of the fix: migrate `training-presets.ts` fetches to `apiFetch`, and give
`training-socket.ts` the same `"authentication failed"` error-frame → 
`handleSessionExpired()` treatment `chat-socket.ts` got.

## Comments

### Planner · 2026-07-17
Filed from the Verifier's out-of-scope note on TODO-0084.
