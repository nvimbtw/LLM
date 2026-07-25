---
id: ISSUE-0059
type: issue
title: "Smaller default vocab (32768 → ~8192) for small corpora"
scope: backend
priority: P3
status: todo
rule: none
location: backend/src/training.rs frontend/hooks/use-training.ts
blocked-by: []
blocks: []
---

## Description
### What
Lower the **default** BPE vocabulary size for the toy/small-corpus regime this
project actually trains on (32768 → ~8192). A large vocab on a small corpus wastes
a big fraction of parameters/compute on the output projection and embedding
(`d*vocab` + `vocab*d`) for tokens that barely occur, which both slows training
and dilutes the learning signal.

### Why this is *separate* from the mixed-precision epic (EPIC-0052)
It's a cheap quality win called out in ISSUE-0050 alongside the LR defaults, but
it neither depends on nor gates the f32-master fix, so it is filed independent of
the epic's `blocked-by` chain. It can land before, during, or after 0052 with no
ordering constraint.

### Shape of the change
- Change the default `max_vocab` where it's defined (the pretrain config default
  in `backend/src/training.rs` and the frontend `DEFAULT_CONFIG.max_vocab` in
  `frontend/hooks/use-training.ts`) to ~8192.
- Keep it a **default only** — the field stays user-editable; explicit overrides
  and existing saved models (which carry their own vocab) are unaffected.
- Confirm the value keeps whatever power-of-two / divisibility constraint the
  existing `validate_config` enforces (8192 is a power of two — verify it passes
  the same checks 32768 did).

### Verification
- Backend + frontend build/typecheck clean; a fresh pretrain with defaults builds
  a ~8192-token tokenizer and trains.
- Existing validation still passes for the new default; explicit larger vocab
  still allowed.

## Comments
### Auditor · 2026-07-13
Filed as the *independent* cheap win from ISSUE-0050 (the LR defaults are the
epic-coupled one — ISSUE-0058 — because they depend on the master-weight fix being
proven; the vocab default does not). P3: pure default tuning, no correctness risk.
