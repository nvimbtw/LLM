# CLAUDE.md

> **Role:** See [`AGENTS.md`](../AGENTS.md) at the project root for the agent
> role system. Unless told otherwise, you are the **Fixer** — the only role
> that edits application code. Adhere to the coding rules in
> [`skills/RULES.md`](../skills/RULES.md).

Guidance for Claude Code when working in this repository.

## Project overview

Backend API for an LLM project (crawler, data pipeline, tokenizer, and model training/inference, fronted by a chat-style web UI). Rust backend, Next.js/TypeScript frontend. This file covers the `backend/` half.

## Tech stack

- **Web framework**: Actix-web. Not Axum — deliberately avoided for its extractor/trait-bound complexity.
- **Database**: PostgreSQL via `sqlx`. Not an ORM (no Diesel, no SeaORM) — raw SQL with compile-time query checking via `query_as!`/`query!`.
- **Auth**: JWT via `jsonwebtoken` (rust_crypto backend), passwords hashed with `argon2`.
- **Migrations**: `sqlx::migrate!`, embedded in the binary, run automatically on startup. Migration files live in `backend/migrations/`.

## Directory structure

backend/
├── Cargo.toml          (also the workspace root)
├── migrations/
├── src/
│   ├── main.rs          thin entrypoint only: HttpServer setup, CORS, app_data registration
│   ├── routes/           one file per resource, HTTP layer only
│   │   ├── mod.rs         composes all route modules
│   │   ├── auth.rs
│   │   ├── users.rs
│   │   └── chats.rs
│   ├── db/                one file per entity, pure SQL queries
│   │   ├── mod.rs
│   │   ├── user.rs
│   │   └── model.rs
│   └── crypto/
│       ├── mod.rs
│       ├── password.rs    argon2 hash/verify
│       └── jwt.rs          token generation/verification
└── crates/                 nested under backend, NOT siblings of backend/
    ├── crawler/
    ├── data_pipeline/
    ├── tokenization/
    └── llm/

`crates/*` are Cargo workspace members declared in `backend/Cargo.toml`. They are subordinate to `backend`, not independent top-level projects — never create a new crate outside `backend/crates/`.

## Layering rules (strict)

- **`routes/*.rs`**: HTTP handlers only. Parses requests, calls `db::` functions, maps results to `HttpResponse` with correct status codes. Each file exposes a `pub fn configure(cfg: &mut web::ServiceConfig)`, registered in `routes/mod.rs` via `web::scope(...)`.
- **`db/*.rs`**: Pure SQL queries via `sqlx`. Functions return `Result<T, sqlx::Error>` or `Result<Option<T>, sqlx::Error>` — never return `HttpResponse` or `impl Responder` from this layer. Db structs (e.g. `User`) may carry sensitive fields (password hash) — never serialize a db struct directly back to a client; use a separate response struct.
- **`crypto/*.rs`**: Self-contained crypto helpers (hashing, JWT). No DB or HTTP awareness.
- Composite operations that span multiple concerns (e.g. login: lookup user + verify password + issue token) live in `routes/`, not `db/`, since they aren't pure SQL.

## Database conventions

- Enums (e.g. `AccessLevel`, `document_status`, `job_type`) are real Postgres `ENUM` types, mapped via `#[derive(sqlx::Type)]` with `#[sqlx(type_name = "...", rename_all = "lowercase")]` on the Rust side. Don't use plain `TEXT` columns for fixed-value fields.
- Every mutable entity should have `created_at` (and `updated_at` where relevant) — required for ordering and for sidebar/recency UI on the frontend.
- IDs are UUIDs (`gen_random_uuid()` default in Postgres, `Uuid` type in Rust via the `uuid` crate).
- Migrations are additive and forward-only once data exists. Only edit an existing migration file directly while the local dev DB is still disposable/empty.

## Data storage (crawler/pipeline)

Actual data (scraped HTML, processed text, tokenized output, model weights) lives on the filesystem under `backend/data/scraped/`, `backend/data/processed/`, `backend/data/tokenized/`, `backend/data/models/` — not in Postgres — needed for manual inspection during crawling/debugging. Postgres only stores tracking metadata:
- `documents` table: one row per crawled page, tracks status, file paths, crawl depth/parent, content hash for dedup.
- `jobs` table: tracks long-running async operations (crawl, tokenize, train) — status, progress, config, for frontend polling and lineage.

## Error handling

Distinguish failure types and map to correct HTTP status codes — don't collapse everything into 500:
- `400 Bad Request` — malformed/invalid client input
- `401 Unauthorized` — auth failure (use an identical generic message for "user not found" and "wrong password" — never reveal which one failed)
- `409 Conflict` — e.g. duplicate username (check `sqlx::Error::Database` + `is_unique_violation()`)
- `500 Internal Server Error` — genuine server-side faults (DB connection issues, hashing failure, etc.)

Never return raw error internals (`format!("{}", e)`) in the HTTP response body to the client — log server-side (`eprintln!`/`log::error!`/`tracing::error!`) and return a generic message instead.

## Code style

- Favor plain structs and free functions over trait-heavy abstractions. Avoid generics unless there's a concrete, current need for them (not "might be reusable later").
- Avoid Rust idioms that exist mainly to be clever — no unnecessary lifetime gymnastics, no premature trait objects, no builder patterns for structs with 2–3 fields.
- This is a deliberate reaction to past frustration with idiomatic-but-opaque Rust. When in doubt, pick the more direct, C-like option over the more "proper" Rust one.

## Working in this repo (Claude Code specifically)

- Implement small, well-scoped, obvious changes freely (a new route following existing patterns, a new migration, a straightforward db query).
- Before any architectural change — new layering pattern, new crate, restructuring existing modules, swapping a library — explain the reasoning and proposed shape first, and wait for confirmation before editing.
- When fixing a compiler error or bug, explain why it happened, not just the fix. The person working on this project is intentionally trying to deepen their own understanding of Rust, not offload it.
- Match the existing patterns in the codebase (route file shape, db file shape, error-matching style) rather than introducing a new convention for similar problems.
