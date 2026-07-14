---
id: ISSUE-0042
title: "Preset REST routes — list/create/update/delete named param sets, per user"
scope: backend
priority: P2
status: verification
rule: none
location: backend/src/routes/preset.rs backend/src/routes/mod.rs
blocked-by: [ISSUE-0041]
blocks: [ISSUE-0048]
---

## Description

Part of the `/training` redesign epic (ISSUE-0040). Exposes the `training_presets`
storage from ISSUE-0041 over HTTP so the frontend params rail can save, list, load,
and delete named parameter sets. HTTP layer only — parse request, call `db::preset`,
map to `HttpResponse` with correct status codes, per `backend/CLAUDE.md`.

### The routes

A new `routes/preset.rs` exposing `pub fn configure(cfg: &mut web::ServiceConfig)`,
mounted under a `web::scope("/training/presets")` from `routes/mod.rs` (sit it next
to the existing training scope). Every handler takes `AuthUser` and scopes to the
caller's user id — a user only ever sees/edits their own presets.

| Method | Path | Handler | Success | Notes |
| --- | --- | --- | --- | --- |
| GET | `/training/presets` | list | `200` `[PresetResponse]` | `db::preset::list_for_user`, newest-updated first |
| POST | `/training/presets` | create | `201` `PresetResponse` | body `{ name, config }`; `409` on `UNIQUE(id_user,name)` violation |
| PUT | `/training/presets/{id}` | update | `200` `PresetResponse` / `404` | body `{ name?, config? }`; owner-scoped; `409` on rename collision |
| DELETE | `/training/presets/{id}` | delete | `204` / `404` | owner-scoped |

### Shape + rules

- A dedicated `#[derive(Serialize)] struct PresetResponse` — never serialize the
  `db::preset::Preset` struct straight to the client (layering rule, even though the
  fields currently coincide — cf. `ModelResponse` in `routes/model.rs`).
- Request bodies deserialize into small `#[derive(Deserialize)]` structs. `config`
  is accepted as an opaque `serde_json::Value` — the preset store doesn't validate
  the `TrainConfig` shape (the `/training` POST validates on use); just require
  `name` non-empty and trimmed.
- **Status-code discipline** (`backend/CLAUDE.md`): `409 Conflict` on a duplicate
  name — detect via `sqlx::Error::Database` + `is_unique_violation()`, the same
  pattern used for duplicate usernames. `404` when an id isn't found *for this user*
  (a preset owned by someone else must read as `404`, never `403`, so ownership
  isn't leaked — mirror `get_model`'s private-model handling). `400` on empty name.
  Never echo raw `sqlx`/error internals in the body — log server-side, return a
  generic message.
- Ownership is enforced in the SQL `WHERE id_user = $…` (from 0041), and the handler
  passes `auth.claims`'s user id — do not trust an id from the body.

### Notes for the Builder

- Resolve the authenticated user's id the same way other authed routes do
  (`AuthUser` extractor / `auth.claims`). Check how `routes/chat.rs` or `usage.rs`
  pull the user id and match it.
- Keep handlers thin: one `db::preset` call each, plus the status mapping. No
  business logic lives here.

## Comments
### Auditor · 2026-07-12
Child of ISSUE-0040, blocked by 0041 (needs the db module + table). Independently
verifiable with `curl`/an HTTP test once the migration is applied. Blocks the
frontend preset UI (0048).

### Builder · 2026-07-12
Added `routes/preset.rs` (list/create/update/delete, all `AuthUser`- and
owner-scoped via `auth.claims.sub`) with a dedicated `PresetResponse` (omits
`id_user`). Mounted `web::scope("/training/presets")` in `routes/mod.rs` **before**
`/training` so the more specific prefix matches first. Status discipline mirrors
`user.rs`/`model.rs`: `409` on `is_unique_violation()`, `404` for not-found-for-user
(never `403`, so ownership isn't leaked), `400` on empty name, generic bodies with
server-side logging.

**Verified end-to-end** against a debug server (temporarily on :8081 to avoid the
running release server on :8080; port reverted after). Full CRUD sequence returned
exactly the intended codes: list-empty `200 []` → create `201` → duplicate `409` →
empty-name `400` → update `200` (`updated_at` advanced) → update-missing `404` →
list `200` (1 item) → unauth `401` → delete `204` → delete-again `404`. `config`
round-trips through JSONB intact. `cargo build` + `cargo clippy` clean.
