# CLAUDE.md

The Rust half: people, sessions, and the seam to the services behind it. It owns
users and presets and nothing else — models, conversations, runs and scores live
in `engine/`, and this side forwards to them.

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
    ├── auth/          argon2 hashing, the token cipher, JWT issue/verify
    ├── db/            users and presets — sqlx + migrations/      → auth
    ├── engine/        the engine subprocess, and the proxy to it  → auth
    └── api/           routes + extractors + sockets               → engine, db, auth
```

The database holds three tables: `users`, `training_presets`, and sqlx's own
migration ledger. Everything else moved to the engine.

Dependency order is strictly one-directional and Cargo enforces it — a cycle
simply fails to compile. `crates/*` are workspace members subordinate to
`backend`; never create a crate outside `crates/`.

`api` is named for what it is rather than `web`, which collides with
`actix_web::web` in every route file.

## Layering rules

- **`api/routes/*.rs`** — HTTP only. Parse, call `db::` or the proxy, map to a
  response. Each file exposes `pub fn configure(cfg: &mut web::ServiceConfig)`.
- **`db/*.rs`** — SQL only. Returns `Result<T, sqlx::Error>`, never an
  `HttpResponse`. A `db` struct is never serialized to a client:
  `db::user::User` carries the password hash.
- **`auth/*.rs`** — no DB or HTTP awareness, enforced by its manifest naming
  neither `sqlx` nor `actix-web`.
- Composite operations spanning concerns (login: look up, verify, issue a
  token) live in `api/routes/`, not `db/`.

## Proxy routes

A route for something a service owns forwards rather than re-models it. The
payload is `serde_json::Value` on purpose — a field the engine adds needs no
change here, and `upstream::Upstream` carries the service's own status back, so
a 404 stays a 404 instead of becoming a generic 502.

The one thing this side decides is **who is asking**: `upstream::caller` turns
the verified user token into the `act` claim of a short-lived service token, and
the service scopes by that. A user token is never forwarded.

## Adding a service

`Proxy` names no service. A second one is:

1. a crate beside `engine/` owning its process and its `AUDIENCE`,
2. a `Proxy` built with that audience in `main.rs`, registered as `app_data`,
3. route files above it, mounted in `routes/mod.rs`.

Nothing in `api` or `auth` changes.

## Database conventions

- Fixed-value fields are real Postgres `ENUM`s mapped with `#[derive(sqlx::Type)]`,
  not `TEXT`.
- IDs are UUIDs (`gen_random_uuid()`, `Uuid` in Rust).
- Every mutable entity carries `created_at`, and `updated_at` where relevant.
- Migrations are additive and forward-only once data exists.

## The Python engine

`crates/engine` is the only crate that knows the engine is a subprocess: it
spawns it as a child of the Actix server, health-checks it before the server
reports ready, restarts it if it dies, and kills it on shutdown.

Configured by environment, because these are properties of the machine:

| Variable | Default | Meaning |
|---|---|---|
| `ENGINE_DIR` | `../engine` | Working directory; `main:app` resolves against it |
| `ENGINE_PYTHON` | `$ENGINE_DIR/.venv/bin/python` | Interpreter; started as `-m uvicorn` |
| `ENGINE_HOST` | `127.0.0.1` | |
| `ENGINE_PORT` | `8001` | 8000 is commonly taken by a local Apache |
| `ENGINE_AUTOSTART` | `true` | Off = "an engine is already running, do not spawn" |
| `ENGINE_STARTUP_TIMEOUT_SECS` | `60` | Importing torch is slow on a cold cache |
| `ENGINE_SERVICE_SECRET` | unset | Shared with the engine; unset leaves both open |

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

Naming carries the meaning; comments carry only what naming cannot.

- **A comment that restates its code is deleted.** If a function is well named
  and well scoped, saying so again in prose is noise that buries the comments
  that matter.
- **A comment longer than its function is a design signal**, not documentation:
  the function needs a better name, a smaller body, or to be split.
- **What stays** is the *why* behind something well named but not readable at a
  glance — a trap that was walked into once (`canonicalize` and the venv), a
  status chosen deliberately (404 rather than 403 so existence is not leaked),
  an ordering that is load-bearing (`/me` registered before `/{id}`).
- Long-form findings live in [`docs/decisions/`](../docs/decisions/), not here.
