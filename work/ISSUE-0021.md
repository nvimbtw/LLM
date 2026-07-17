---
id: ISSUE-0021
type: issue
title: Missing input validation on user-facing route parameters (R10)
scope: backend
priority: P1
status: done
rule: R10
location: backend/src/routes/user.rs:40-73 backend/src/routes/files.rs:22-93 backend/src/routes/chat.rs:191-225,476-493 backend/src/routes/model.rs:91-118 backend/src/routes/auth.rs:14-66
blocked-by: []
blocks: []
---

## Description
Several HTTP route handlers accept user input without validating size/format
boundaries: `insert_user`/`update_user` accept arbitrary-length usernames/passwords;
`upload_files` has no max file size; `handle_message`/`create_chat` have no max
content length; `update_model` has no name/description length limit; `login` has no
username/password length check. Malicious or oversized inputs can exhaust resources
or cause unexpected DB behavior. Validate at the boundary before passing to
DB/crypto layers.

## Comments
### Builder · 2026-07-10
Added `src/routes/validate.rs`: named limit constants (username 1–64, password
8–256, message ≤32768, model name ≤128 / description ≤2048 chars, upload ≤25 MiB
— so no magic numbers, R8) plus `field_len`/`max_len` helpers returning a
client-safe message (field + bound, no internals). Registered `pub mod validate`
in `routes/mod.rs`. Wired the boundary checks (all reject with 400 before the
value reaches db/crypto):

- **user.rs** `insert_user` (username + password) and `update_user` (each, when
  present).
- **auth.rs** `login`: upper-bound only, generic `"invalid request"` 400 — must
  not reveal which field, and caps input before argon2 verify (cheap DoS guard).
- **model.rs** `update_model`: name + description when present.
- **chat.rs** `create_chat` (400) and `handle_message` (WS error frame): message
  content length.
- **files.rs** `upload_files`: per-file byte cap enforced *while streaming*; on
  overflow the partial file is removed and it returns 413.

Optional fields use `as_deref().and_then(|v| …err())` (single non-nested `if`) to
keep clippy's `collapsible_if` happy. Verified: `cargo build` and `cargo clippy`
both **0 warnings**.

### Verifier · 2026-07-10 · pass

**Checked:**
1. **`src/routes/validate.rs`** exists with named constants (`MIN_USERNAME_LEN`,
   `MAX_USERNAME_LEN`, `MIN_PASSWORD_LEN`, `MAX_PASSWORD_LEN`, `MAX_MESSAGE_LEN`,
   `MAX_MODEL_NAME_LEN`, `MAX_MODEL_DESC_LEN`, `MAX_UPLOAD_BYTES`) — no magic
   numbers. `field_len`/`max_len` helpers present and return client-safe messages.
   Registered as `pub mod validate` in `routes/mod.rs`.
2. **Validation wired into all cited handlers** (each rejects oversized input
   **before** reaching db/crypto):
   - `user.rs` `insert_user`: `field_len` on username (1–64) and password (8–256)
     before `hash_password`.
   - `user.rs` `update_user`: `field_len` on username/password when present (via
     `as_deref().and_then()`) before hash/db.
   - `auth.rs` `login`: `max_len` upper-bound only; generic `"invalid request"` 400
     that doesn't reveal which field; checked before `argon2` verify.
   - `model.rs` `update_model`: `field_len`/`max_len` on name/description when
     present, before `db::model::update`.
   - `chat.rs` `create_chat`: `max_len` on message content, returns 400 before
     `db::chat::create`.
   - `chat.rs` `handle_message`: `max_len` on message content, returns WS error
     frame before db operations.
   - `files.rs` `upload_files`: per-file byte cap enforced while streaming; removes
     partial file and returns 413 on overflow.
3. **Build clean:** `cargo build` — 0 warnings; `cargo clippy` — 0 warnings.
