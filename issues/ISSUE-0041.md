---
id: ISSUE-0041
title: "training_presets table + db/preset.rs — persist named parameter sets per user"
scope: backend
priority: P2
status: verification
rule: none
location: backend/migrations/ backend/src/db/preset.rs backend/src/db/mod.rs
blocked-by: []
blocks: [ISSUE-0042]
---

## Description

Part of the `/training` redesign epic (ISSUE-0040). Adds the storage layer for
**named, reusable training parameter sets**, per user. The route layer that exposes
these over HTTP is the sibling child ISSUE-0042; this issue is the migration + the
pure-SQL `db/` module only, matching the strict layering in `backend/CLAUDE.md`.

### The table

New forward-only migration under `backend/migrations/` (next sequential number):

```sql
CREATE TABLE training_presets (
    id_preset   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    id_user     UUID NOT NULL REFERENCES users(id_user) ON DELETE CASCADE,
    name        TEXT NOT NULL,
    config      JSONB NOT NULL,          -- a TrainConfig-shaped blob (inspectable, not queried column-wise)
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (id_user, name)               -- a preset name is unique per user, not globally
);
```

- `config` is JSONB, mirroring how `models.parameters` stores the training config as
  an inspectable JSON blob rather than a wide column set (`db/model.rs:6-10`). We do
  not enforce the `TrainConfig` shape at the DB layer — the `/training` POST already
  validates a config when a run actually starts.
- `ON DELETE CASCADE` on `id_user`: presets are personal scratch data, so they die
  with the user (unlike models, which detach references). Confirm the actual PK
  column name on `users` (`id_user` per the db conventions) before writing the FK.
- `UNIQUE (id_user, name)` gives the route layer a clean `409` path for a duplicate
  name and an upsert target for "save over an existing preset".

### The db module (`db/preset.rs`)

Follow the shape of `db/model.rs` — plain structs, free functions returning
`Result<T, sqlx::Error>`, compile-time-checked `query_as!`/`query!`, no HTTP or
serde-to-client concerns:

```rust
pub struct Preset {
    pub id_preset: Uuid,
    pub id_user: Uuid,
    pub name: String,
    pub config: Value,        // serde_json::Value
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

pub async fn list_for_user(pool, id_user) -> Result<Vec<Preset>, sqlx::Error>;   // newest-updated first
pub async fn create(pool, id_user, name, config) -> Result<Preset, sqlx::Error>; // may hit UNIQUE violation
pub async fn update(pool, id_preset, id_user, name: Option<&str>, config: Option<&Value>) -> Result<Option<Preset>, sqlx::Error>; // scoped to owner; COALESCE style; bump updated_at
pub async fn delete(pool, id_preset, id_user) -> Result<bool, sqlx::Error>;      // scoped to owner
```

Every mutating/reading-by-id query must be **scoped to `id_user`** in the `WHERE`
clause so ownership is enforced in SQL, not just in the handler — a user must never
touch another user's preset. Register the module in `db/mod.rs`.

### Notes for the Builder

- Do not serialize the `Preset` db struct straight to a client — that's the route
  layer's response struct (0042), per the layering rule.
- `updated_at` should advance on `update` (`SET updated_at = now()`), so the
  `list_for_user` "newest-updated first" ordering is meaningful after an edit.
- Keep it plain and C-like per `backend/CLAUDE.md` — no generics, no builder.

## Comments
### Auditor · 2026-07-12
Child of ISSUE-0040. Storage-only so it's independently verifiable (migration
applies, `cargo build` compile-checks the queries) without waiting on the route or
frontend work. Blocks 0042.

### Builder · 2026-07-12
Added migration `20260712000000_create_training_presets.sql` (table per the spec —
`UNIQUE(id_user, name)`, `ON DELETE CASCADE`, JSONB `config`) and `db/preset.rs`
with `list_for_user` / `create` / `update` / `delete`, all owner-scoped in the SQL
`WHERE`. Registered in `db/mod.rs`. Applied the migration to the dev DB and built:
`query_as!` compile-time verification passes against the live schema. Dropped the
`id_user` field from the `Preset` struct (never read — ownership is enforced by the
`WHERE`, response omits it) to keep clippy clean. `cargo build` + `cargo clippy`
zero warnings.
