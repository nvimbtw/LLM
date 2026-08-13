# CLAUDE.md

The Rust half: HTTP, auth, and the seam to the Python engine. Training and
inference live in `engine/` — nothing here computes anything about a model.

## Tech stack

- **Web framework**: Actix-web. Not Axum — deliberately avoided for its
  extractor/trait-bound complexity.
- **Database**: PostgreSQL via `sqlx`. Raw SQL with compile-time checking
  through `query_as!`/`query!`, not an ORM.
- **Auth**: JWT via `jsonwebtoken`, passwords hashed with `argon2`.
- **Migrations**: `sqlx::migrate!`, embedded at compile time, run on startup
  from `db::create_pool`. The macro resolves its argument against
  `CARGO_MANIFEST_DIR`, so the directory and the `"./migrations"` literal must
  move together — mismatching them still compiles and fails at runtime.

## Crates

```
backend/
├── Cargo.toml         workspace root AND the binary package
├── src/main.rs        HttpServer setup, CORS, app_data
└── crates/
    ├── crypto/        argon2 hashing, AES token cipher
    ├── db/            one module per entity, sqlx + migrations/   → crypto
    ├── engine/        the engine subprocess, and typed calls into it
    ├── training/      the run config the API validates             → engine
    ├── jobs/          queue + train_worker + chain (owns the pool) → db, training
    └── api/           routes + extractors + ws + jwt               → jobs, db, crypto
```

Dependency order is strictly one-directional and Cargo enforces it — a cycle
simply fails to compile. `crates/*` are workspace members subordinate to
`backend`; never create a crate outside `crates/`.

`api` is named for what it is rather than `web`, which collides with
`actix_web::web` in every route file.

## Layering rules

- **`api/routes/*.rs`** — HTTP only. Parse, call `db::`, map to `HttpResponse`.
  Each file exposes `pub fn configure(cfg: &mut web::ServiceConfig)`.
- **`db/*.rs`** — SQL only. Returns `Result<T, sqlx::Error>`, never an
  `HttpResponse`. A `db` struct is never serialized to a client:
  `db::user::User` carries the password hash.
- **`crypto/*.rs`** — no DB or HTTP awareness, enforced by its manifest naming
  neither `sqlx` nor `actix-web`.
- Composite operations spanning concerns (login: look up, verify, issue a
  token) live in `api/routes/`, not `db/`.

`jwt` lives in `api` rather than `crypto` because its `Claims` carry a
`db::user::AccessLevel`, and `db` depends on `crypto` — the other direction is
a cycle. `api` is the only layer that issues or verifies a token anyway.

## Database conventions

- Fixed-value fields are real Postgres `ENUM`s mapped with `#[derive(sqlx::Type)]`,
  not `TEXT`.
- IDs are UUIDs (`gen_random_uuid()`, `Uuid` in Rust).
- Every mutable entity carries `created_at`, and `updated_at` where relevant.
- Migrations are additive and forward-only once data exists.

## The Python engine

Training and inference run in `engine/` (PyTorch). `crates/engine` is the only
crate that knows that: it spawns the engine as a child of the Actix server,
health-checks it before the server reports ready, restarts it if it dies, and
kills it on shutdown. Everything above calls typed functions.

Configured by environment, because these are properties of the machine:

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_DIR` | `../engine` | Working directory; `main:app` resolves against it |
| `ENGINE_PYTHON` | `$ENGINE_DIR/.venv/bin/python` | Interpreter; started as `-m uvicorn` |
| `ENGINE_HOST` | `127.0.0.1` | |
| `ENGINE_PORT` | `8001` | 8000 is commonly taken by a local Apache |
| `ENGINE_AUTOSTART` | `true` | Off = "an engine is already running, do not spawn" |
| `ENGINE_STARTUP_TIMEOUT_SECS` | `60` | Importing torch is slow on a cold cache |

Two things that are load-bearing and non-obvious:

- **`ENGINE_PYTHON` is made absolute but never `canonicalize`d.** A virtualenv's
  `bin/python` is a symlink to the system interpreter, and Python derives
  `sys.prefix` from the path it was invoked by — resolving the link hands over
  an interpreter with no idea it is in a venv, and `-m uvicorn` fails against a
  venv where uvicorn is plainly installed.
- **Absolute at all, because `Command` resolves the program path *after*
  chdir-ing into `current_dir`.** A relative `ENGINE_PYTHON` would be looked up
  under `$ENGINE_DIR/`, failing with a bare ENOENT.

Autostart off is the normal development setup: run `uvicorn main:app --reload`
in another terminal and the server attaches to it.

## Error handling

Routes map failures to status codes and log the cause; they never return a
`sqlx::Error` to a client. A `500` body says what failed, not why.

## Code style

- A comment that restates the code is deleted.
- A comment longer than its function means the function needs a better name, a
  smaller body, or to be split.
- The *why* that would otherwise be re-broken stays, and is short.
- Long-form findings live in [`docs/decisions/`](../docs/decisions/), not here.
