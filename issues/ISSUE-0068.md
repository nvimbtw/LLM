---
id: ISSUE-0068
title: "Rename db::model::Model to db::model::ModelRow"
scope: backend
priority: P2
status: todo
rule: R21
location: backend/src/db/model.rs:11
blocked-by: []
blocks: [ISSUE-0064]
---

## Description

`db::model::Model` stutters — the module `model` and the struct `Model` repeat. Per **R21**, rename to `ModelRow` so the path reads `db::model::ModelRow`. This also avoids confusion with the `llm::training::transformer::Model` (after ISSUE-0061 renames it) — distinct names for distinct concepts.

### Scope of change
- `backend/src/db/model.rs`: rename struct and `NewModel`, update internal references
- `backend/src/jobs.rs`: `use crate::db::model::NewModel;`
- `backend/src/routes/model.rs`: update import and usage
- Any other file referencing `db::model::Model` or `db::model::NewModel`

### Shape of the fix
1. Rename `pub struct Model` → `pub struct ModelRow` in `model.rs`
2. Also rename `pub struct NewModel` → `pub struct NewModelRow` for consistency (optional but recommended since it's a DB input row)
3. Update all imports and references in consuming files

### Verification
- `cargo build` passes with zero warnings
- No references to `db::model::Model` remain (pre-rename name)

## Comments
_No comments yet._
