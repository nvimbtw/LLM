---
id: ISSUE-0015
title: cargo build emits 4 warnings (R7)
scope: backend
priority: P1
status: done
rule: R7
location: backend/src/db/message.rs:7,12 backend/src/extractors.rs:85 backend/src/routes/user.rs:57 backend/src/main.rs:75
blocked-by: []
blocks: []
---

## Description
`cargo build` (and `cargo clippy`) produce 4 warnings: 2 `dead_code` on
`Message.id_chat`, `Message.id_model` (message.rs) and `AdminUser.claims`
(extractors.rs); 2 `clippy::needless_borrow` on `&pool.get_ref()` (user.rs:57) and
`&pool` (main.rs:75). R7 requires a clean build. Either fix or add justified
`#[allow]` with a comment.

## Comments
### Builder · 2026-07-10
Reproduced all 4 warnings first (`cargo build` → 2 `dead_code`; `cargo clippy` →
those 2 plus 2 `needless_borrow`), then fixed each at its root:

- **`db/message.rs` `id_chat` / `id_model` (dead_code):** these are columns in
  every `messages` `RETURNING` that `query_as!` maps into the struct, so they
  can't be deleted without trimming the queries and making `Message` no longer
  mirror the table. Added field-level `#[allow(dead_code)]` with a comment
  explaining they're DB-mapped but not yet read — the justified-allow path R7
  permits.
- **`extractors.rs` `AdminUser.claims` (dead_code):** mirrors `AuthUser.claims`
  (the auth guard carries the decoded payload for handlers that want the admin's
  identity); the guard's job is proving admin access, so it isn't always read.
  Justified `#[allow(dead_code)]` + comment.
- **`routes/user.rs:57` `&pool.get_ref()` (needless_borrow):** `get_ref()`
  already yields `&PgPool` and `db::user::create` takes `&PgPool`, so the outer
  `&` created a `&&PgPool` the compiler immediately re-derefs. Dropped the `&`.
- **`main.rs:75` `&pool` (needless_borrow):** `pool` is already `&PgPool` here;
  dropped the `&`.

No `#[allow]` was used to paper over a real removable dead field — both allows
are load-bearing DB/auth struct fields. Verified: `cargo build` **0 warnings**
and `cargo clippy` **0 warnings** across the whole `backend` crate.

### Verifier · 2026-07-10 · pass
Confirmed: `cargo build` → 0 warnings, `cargo clippy` → 0 warnings in
`backend/`. Checked all 4 fix sites:
- `db/message.rs` lines 10/16 — both `#[allow(dead_code)]` carry justifying
  comments explaining they're DB-mapped columns.
- `extractors.rs` line 88 — `#[allow(dead_code)]` on `AdminUser.claims` with
  comment explaining it mirrors `AuthUser` for handlers that need it.
- `routes/user.rs:57` — `pool.get_ref()` (no outer `&`).
- `main.rs:75` — `pool` (no outer `&`).
All R7 warnings resolved. Passing.
