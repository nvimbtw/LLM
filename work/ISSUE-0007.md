---
id: ISSUE-0007
type: issue
title: Split handle_message — 172 lines (R1)
scope: backend
priority: P2
status: done
rule: R1
location: backend/src/routes/chat.rs:190-362
blocked-by: []
blocks: []
---

## Description
`handle_message` spans 172 lines covering parse → resolve chat → persist message →
resolve model → build history → spawn inference → stream tokens → persist reply →
record usage → send done. Each stage should be a named helper.

## Comments
### Verifier · migrated · pass
`handle_message` is now ~35 lines with `parse_frame`, `resolve_chat`,
`persist_user_message`, `resolve_model`, `load_history`, `stream_reply`,
`finalize_turn` extracted as named helpers.
