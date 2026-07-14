---
id: ISSUE-0069
title: "Rename db::preset::Preset to db::preset::PresetRow"
scope: backend
priority: P2
status: todo
rule: R21
location: backend/src/db/preset.rs:10
blocked-by: []
blocks: [ISSUE-0064]
---

## Description

`db::preset::Preset` stutters — the module `preset` and the struct `Preset` repeat. Per **R21**, rename to `PresetRow` so the path reads `db::preset::PresetRow`.

### Scope of change
- `backend/src/db/preset.rs`: rename struct, update internal references
- `backend/src/routes/preset.rs`: update import and usage
- Any other file referencing `db::preset::Preset`

### Shape of the fix
1. Rename struct in `preset.rs`: `pub struct PresetRow`
2. Update imports and references in consuming files

### Verification
- `cargo build` passes with zero warnings
- No references to `db::preset::Preset` remain

## Comments
_No comments yet._
