---
id: ISSUE-0020
title: Magic numbers across backend (R8)
scope: cross
priority: P2
status: todo
rule: R8
location: backend/src/llm/src/train.rs:72,257,262 backend/src/llm/src/training/transformer.rs:1096 backend/src/llm/src/training/embed.rs:4 backend/src/main.rs:54,63,69 backend/src/db/mod.rs:11 backend/src/crypto/jwt.rs:20 backend/src/jobs.rs:256
blocked-by: []
blocks: []
---

## Description
Tunable/meaningful values inlined across the backend: `256` (special-token offset,
train.rs:72, jobs.rs:256), `50` (gradient-norm log interval, train.rs:257), `10`
(progress report interval, train.rs:262), `6` (v1 default layers,
transformer.rs:1096), `6.0` (Xavier uniform scale, embed.rs:4), `3600` (CORS
max-age seconds, main.rs:54), `8080` (port, main.rs:63), `"admin"` (default
username, main.rs:69), `5` (DB pool max connections, db/mod.rs:11), `24` (JWT token
expiry hours, jwt.rs:20). Each should be a named `const` with a brief doc comment.

## Comments
_No comments yet._
