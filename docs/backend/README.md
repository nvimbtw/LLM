# Backend — Rust API

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
    ├── tools/         what a model can reach for, and the answer  → (nothing)
    └── api/           routes + extractors + sockets        → engine, tools, db, auth
```

`tools` is a leaf like `auth`: it knows about the outside world and about the
marker vocabulary, and nothing about this project's users, sessions or the engine
process.

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

## Tool calls

The models here are too small to stop at a tool call: a large model emits one and
halts, ours emits one and carries on. So a turn is not generate-pause-resume, it
is **let the reply finish, run what it asked for while it does, then ask again
with the answers in the conversation** — see
[`../decisions/tool-calls.md`](../decisions/tool-calls.md) for what that forces.

`api/turn.rs` drives it. Per round: relay the engine's frames to the browser,
spawn a tool the moment its `tool_call` frame arrives, keep relaying, join at
`done`, then POST the next round with a `tool_result`. Bounded at
`MAX_TOOL_ROUNDS` — without it a model that searches in every reply never
terminates. Spawning on the call rather than at the end is what makes a search
overlap the tokens after it instead of stalling them.

The engine detects, this side executes. Detection needs token *ids*, which only
the generation loop has; the API key, the outbound HTTP and the round loop are
all properties of the machine, which is this side. Each `done` is stamped
`final` so the client can tell a round's end from the turn's.

Adding a tool is an `impl Tool` and a line in `Registry::from_env`, plus the two
reserved slots the engine needs to spell it (`engine/llm/chat/tools.py`). A tool
whose credentials are missing is **not registered** rather than registered and
failing — chat has to work on a checkout with no API keys, and `Registry::answer`
turns the resulting `Unknown` into a `<|warn|>` the model reads.

A tool's answer is text a stranger wrote going into a prompt, so every tool runs
it through `tools::markup::neutralize` — the same transformation
`scripts/corpus_shards.py` applies to scraped corpus text. Without it a page
containing `<|/assistant|>` ends the turn.

| Variable | Default | Meaning |
|---|---|---|
| `SEARCH_URL` | unset | A self-hosted SearXNG, e.g. `http://localhost:8080` |
| `SEARCH_API_KEY` | unset | Brave Search, used only when no `SEARCH_URL` is set |
| `SEARCH_RESULTS` | `5` | How many results reach the model |
| `TOOL_TIMEOUT_SECS` | `10` | Per call; the reply is already generated by then |

Neither set means `web-search` is not registered. Both set is SearXNG: naming an
instance is a deliberate act, where a key can be left over in an environment
from something else. The startup line says which — `tools ready: web-search
(SearXNG at http://localhost:8080)`.

### Web search

`crates/tools/src/search/` is one tool over two providers. Everything the model
sees — query validation, the result cap, the rendering into markers — lives in
`mod.rs`, so swapping providers cannot change the shape of what a reply is
handed; a `Provider` only turns a query into `Hit`s.

**SearXNG (default, self-hosted).** No key, no quota, no third party:

```sh
cd deploy/searxng
SEARXNG_SECRET=$(openssl rand -hex 32) docker compose up -d
curl 'http://localhost:8080/search?q=test&format=json'   # must not be 403
```

Then `SEARCH_URL=http://localhost:8080` in `backend/.env`.

Two settings in `deploy/searxng/settings.yml` are load-bearing and both fail the
same way — a bare `403` that says nothing about its cause. `search.formats` must
include `json`, which is **not** a default; and `server.limiter` must be off,
because it works by profiling requests as browser-like and an API client is
exactly what it then blocks. `searxng.rs::refusal` names both in the message, so
a 403 in a chat reply tells you which file to open.

The compose file binds to `127.0.0.1` only. An instance with its limiter off is
one that will happily be scraped by anyone who can reach it.

**Brave (hosted).** `SEARCH_API_KEY`. $5/month of free credits at $5 per 1,000
requests, and signing up requires a card even for those.

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
- Long-form findings live in [`docs/decisions/`](../decisions/), not here.
