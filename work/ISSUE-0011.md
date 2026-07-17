---
id: ISSUE-0011
type: issue
title: Missing doc comments on public Transformer API surface (R9)
scope: backend/llm
priority: P2
status: done
rule: R9
location: backend/src/llm/src/training/transformer.rs:49-116,118-407,850-1057,1438-1650
blocked-by: []
blocks: []
---

## Description
`pub struct Transformer` and its public methods (`init`, `forward`, `forward_seq`,
`unembed`, `train_step`, `save_to_path`, `save`, `load`, `to_bytes`, `from_bytes`)
lack `///` doc comments. R9 requires every `pub fn`/`pub struct` to document its
contract (invariants, panics, units). E.g. `train_step` does not document that it
mutates `ws`, or that `chunk` must be ≥2. `load`/`from_bytes` don't mention the
binary format versioning or that they panic on `heads=0`.

## Comments
### Verifier · migrated · pass
Confirmed `///` doc comments on `Transformer` struct (lines 99-101), `init`
(122-123), `forward_seq` (173-175), `train_step`, `save_to_path` (855), `save`
(861), `load` (866-867), `to_bytes`, `from_bytes` (1496).
