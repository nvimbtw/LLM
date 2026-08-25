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

`query_as!`/`query!` check against a live schema at **compile** time, so a build
needs either `DATABASE_URL` pointing at a migrated database or the checked-in
`.sqlx/` cache and `SQLX_OFFLINE=true`. The container images use the second —
a build stage has no route to the `postgres` service. **Regenerate the cache in
the same change as any query or migration edit:**

```sh
cargo sqlx prepare --workspace
```

Cargo fingerprints `.rs` files and not `migrations/`, so a migration edit alone
does not force the crate whose queries it invalidates to rebuild. A stale
`.sqlx/` shows up as a container build failing while `cargo run` on the host
still succeeds.

## Crates

```
backend/
├── Cargo.toml         workspace root AND the binary package
├── src/main.rs        HttpServer setup, CORS, app_data
└── crates/
    ├── auth/          argon2 hashing, the token cipher, JWT issue/verify
    ├── db/            users and presets — sqlx + migrations/      → auth
    ├── service/       spawning a Python service, and the hops both ways → auth
    ├── engine/        names the LLM engine: audience, port, DATA_DIR → service
    ├── data-engine/   names the data engine                          → service
    ├── tools/         what a model can reach for, and the answer  → (nothing)
    └── api/           routes + extractors + sockets   → service, engine, data-engine, tools, db, auth
```

`tools` is a leaf like `auth`: it knows about the outside world and about the
marker vocabulary, and nothing about this project's users, sessions or the engine
process.

`service` names no service. `engine` and `data-engine` are ~40 lines each — an
audience, a set of defaults, and a `Proxy` newtype — and do **not** depend on
each other: the two Python processes are siblings that never speak.

The database holds three tables: `users`, `training_presets`, and sqlx's own
migration ledger. Everything else moved to the engine.

A user's chat settings (decoding knobs, system prompt, UI toggles) live in
`users.settings`, one opaque JSONB blob rather than a table of their own — a
user has exactly one, so a table keyed `UNIQUE(id_user)` would only ever hold
the row the user row already has. `GET`/`PUT /user/me/settings`
(`api/routes/user.rs`) read and replace it whole; the backend never parses the
blob, so the frontend's `Settings` shape (`lib/settings.ts`) can grow without a
migration here, the same way `training_presets.config` stays opaque to a run
shape it doesn't need to know either.

`users.composer_choice` is a second, identically-shaped blob for what the
composer is set to send with (model, thinking budget), behind its own
`GET`/`PUT /user/me/composer-choice`. Not merged into `settings` — the two
write at different rhythms (a pick fires immediately, a settings change
debounces), and one column would mean the later of two independent writes
clobbering the other's field. See
[`../frontend/README.md`](../frontend/README.md) for the client side of both.

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

**One pair of routes forwards something that is not JSON.** `/attachments` takes
a multipart upload from the browser and forwards it as multipart to the engine
(`Proxy::upload`), and serves the bytes back with the content type and
disposition the engine set (`Proxy::fetch`) — this side stores nothing about a
file and would otherwise be guessing at a name it never saw. Both are buffered
rather than streamed: the engine caps what it will store, and a streaming body
cannot carry an upstream failure through a response already begun. The size
ceiling is repeated here so an oversized body is refused at the edge rather than
read into memory on its way to being refused; keep it in step with `MAX_BYTES`
in `engine/api/routes/attachments.py`. What the engine does with them —
and, more to the point, what it does not — is
[`../decisions/attachments-are-stored-not-read.md`](../decisions/attachments-are-stored-not-read.md).

**Two routes are spelled almost alike and are not the same thing.**
`POST /runs/tokenize` queues a vocabulary *build* — minutes of CPU, admin work,
a job with a page. `POST /tokenize/count` measures a string under a model's
existing vocabulary, loads no weights, and takes `AuthUser`, because the
composer's context meter asks it on every pause in typing
([`../frontend/README.md`](../frontend/README.md)). Gating the second one behind
`AdminUser` by analogy with the first would leave every ordinary session's meter
collecting 403s.

The one thing this side decides is **who is asking**: `upstream::caller` turns
the verified user token into the `act` claim of a short-lived service token, and
the service scopes by that. A user token is never forwarded.

## The hop that runs the other way

`api/routes/internal.rs`, behind the `ServiceCaller` extractor, is the one place
a **service** is the caller. The LLM engine reads documents there that the data
engine owns — it cannot ask the data engine itself, because the two Python
services never speak, and it cannot be handed them in advance, because a queued
run needs them at a moment nothing on this side knows ahead of time.

Three things make that safe to have, each enforced somewhere different:

- **The audience claim, in the new direction.** An inbound token says
  `aud: backend`, so a token this server minted *for* a service cannot be
  replayed at it. `contracts/backend-token.json` is the shared vector — minted by
  PyJWT, verified by `jsonwebtoken`, the only token in this project that Python
  signs and Rust checks.
- **The issuer list, named at the call site.** `InboundSecret::caller` takes the
  issuers it will accept, and the extractor passes `engine::AUDIENCE` alone. The
  data engine holds a valid secret and still cannot read through the backend,
  because it has no business asking.
- **`act` survives the hop.** The caller the engine names is re-minted into the
  token going out to the data engine, so a run reads exactly the datasets the
  person who queued it could have read. `crates/service` names no service, so
  the crate that knows both is `api` and only `api`.

The routes are reads only. A run consumes datasets; creating, cleaning and
deleting them stay a person's, under `/data`.

`BACKEND_SERVICE_SECRET` unset leaves `/internal` open and the server says so at
startup — the same bargain each Python service strikes with its own secret unset,
and the same trust boundary, since this server binds loopback by default.

**That default is what makes the bargain safe, so a container cannot strike it.**
`BIND_ADDR` overrides the binding and `docker-compose.yml` sets it to `0.0.0.0`,
because a published port cannot reach a loopback binding. Which is why the same
file *requires* `BACKEND_SERVICE_SECRET` and both service secrets rather than
letting them default to unset — see
[`../decisions/one-command-development.md`](../decisions/one-command-development.md).

## Adding a service

`service` names no service, and `crates/data-engine` is the worked example. A
third one is:

1. a crate beside `engine/` and `data-engine/` naming its `AUDIENCE`, its
   `Defaults` (env prefix, directory, port), and a `Proxy` newtype,
2. a `Supervisor::start` and a `Proxy` in `main.rs`, registered as `app_data`,
3. route files above it, mounted in `routes/mod.rs`.

Nothing in `service`, `api` or `auth` changes.

**The newtype is not ceremony.** Actix keys `app_data` by type, so two bare
`service::Proxy` values silently overwrite each other and every route reaches
whichever was registered last — a failure that compiles and passes most tests.
`crates/api/tests/two_services.rs` is what notices. See
[`../decisions/a-crate-per-service.md`](../decisions/a-crate-per-service.md).

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

Adding a tool is an `impl Tool` and a line in `Registry::with_config`, plus a row
in the engine's table (`engine/llm/chat/tools.py`). **No reserved slots**: every
tool is called through one `<|tool-call|>` marker naming it in JSON, so the tool
table is a property of the build rather than of the vocabulary — see
[`../decisions/generic-tool-calls.md`](../decisions/generic-tool-calls.md). A
tool whose configuration is missing is **not registered** rather than registered
and failing — chat has to work on a checkout with no API keys and no sandbox,
and `Registry::answer` turns the resulting `Unknown` into a `<|warn|>` the model
reads.

`Tool::run` takes the argument *and* a `session` key, which is the chat id.
Most tools ignore it — a search is the same search whoever asked — but the
sandboxed shell keeps a filesystem under it, so a reply that writes a file in
one round finds it there in the next. It is a key and not a chat: `tools` never
learns what it stands for.

A tool's answer is text a stranger wrote going into a prompt, so every tool runs
it through `tools::markup::neutralize` — the same transformation
`scripts/corpus_shards.py` applies to scraped corpus text. Without it a page
containing `<|/assistant|>` ends the turn.

| Variable | Default | Meaning |
|---|---|---|
| `SEARCH_URL` | unset | A self-hosted SearXNG, e.g. `http://localhost:8888` |
| `SEARCH_API_KEY` | unset | Brave Search, used only when no `SEARCH_URL` is set |
| `SEARCH_RESULTS` | `3` | Ceiling on how many results reach the model |
| `SEARCH_BUDGET_CHARS` | `320` | Ceiling on the whole rendered result. **Raise it with the model's context window** |
| `BASH_URL` | unset | The just-bash sandbox, e.g. `http://localhost:8899`. Unset, no `bash` tool |
| `BASH_BUDGET_CHARS` | `320` | Ceiling on the rendered command output. **Raise it with the window** |
| `TOOL_TIMEOUT_SECS` | `10` | Per call; the reply is already generated by then |
| `SEARCH_CACHE_MAX_AGE_DAYS` | `7` | How old a cached result may be before `web-search` re-fetches it live — see "The search cache" below |

Neither search setting means `web-search` is not registered. Both set is
SearXNG: naming an instance is a deliberate act, where a key can be left over in
an environment from something else. The startup line says which — `tools ready:
web-search (SearXNG at http://localhost:8888), bash (just-bash at
http://localhost:8899)`.

### Web search

`crates/tools/src/search/` is one tool over two providers. Everything the model
sees — query validation, the result cap, the rendering into markers — lives in
`mod.rs`, so swapping providers cannot change the shape of what a reply is
handed; a `Provider` only turns a query into `Hit`s.

**SearXNG (default, self-hosted).** No key, no quota, no third party:

```sh
cd deploy/searxng
./run.sh                                          # or: docker compose up -d
curl 'http://localhost:8888/search?q=test&format=json'   # must not be 403
```

`run.sh` is the same container as one `docker run`, for a machine without the
compose plugin — it is a separate package on most distributions. It waits for
the JSON API to answer and says what to check if it does not. Prepend `sudo` if
your user is not in the `docker` group; being in that group is root-equivalent,
so which of the two you pick is a decision about the machine.

Then `SEARCH_URL=http://localhost:8888` in `backend/.env`.

Two settings in `deploy/searxng/config/settings.yml` are load-bearing and both
fail the same way — a bare `403` that says nothing about its cause.
`search.formats` must include `json`, which is **not** a default; and
`server.limiter` must be off, because it works by profiling requests as
browser-like and an API client is exactly what it then blocks.
`searxng.rs::refusal` names both in the message, so a 403 in a chat reply tells
you which file to open.

Two details the image dictates. The **whole config directory** is mounted
writable, because the container runs as its own user and takes ownership of it
on start (`FORCE_OWNERSHIP`) — mounting `settings.yml` alone and read-only is
the obvious thing to try and fails on exactly that. And the port binds to
`127.0.0.1` only: an instance with its limiter off will be scraped by anyone who
can reach it.

**Brave (hosted).** `SEARCH_API_KEY`. $5/month of free credits at $5 per 1,000
requests, and signing up requires a card even for those.

#### Sizing the result

`SEARCH_BUDGET_CHARS` is the one worth thinking about, because the models here
are small enough that a search result competes with the conversation for room.
The engine's default `ctx_window` is **256 tokens**, half of which
`orchestrate/chat.py` reserves for the reply — so the whole prompt, system block
and history included, is about 380 characters. Five untrimmed results measured
~1,200. Nothing errors on that; `build_prompt` simply evicts the question to fit
the answer to it.

So the default is deliberately mean: 320 characters, which is two results. Raise
it with the window — a model at 1024 tokens can afford four or five times this
and will answer better for it.

To see what your own instance actually returns, and what it costs:

```sh
cargo run -p tools --example search_once -- "how tall is elbrus"
```

It reads the same environment the server does and prints the rendered result
with its size in tokens.

### The search cache

`web-search` checks a cache before paying for a live fetch, and writes what it
finds back — see
[`../decisions/search-result-cache.md`](../decisions/search-result-cache.md).
The cache lives in the **data engine**, as a dataset (`web-search-cache`,
`origin = search`, `owner_id = NULL`) rather than a table of its own: a search
result is data on the same terms anything else pulled from the Hub is, and
storing it there is what makes it selectable for training later with no
engine-side work at all.

`tools::SearchCache` is the trait `WebSearch` calls through — `get`/`put`
keyed by a normalized query — and it stays a trait rather than a concrete type
because `tools` is the leaf documented above: it must not learn the shape of
the data engine. The implementation, `api::search_cache::DataEngineCache`, is
wired into `Registry::from_env` in `main.rs`, the one place that already holds
both the registry and `data_engine::Proxy`. Every call it makes is
`auth::Caller::System` — a search result is a fact about the world, not about
whoever happened to ask first, so nothing here is scoped to a person.

Freshness is decided here, not by the data engine: a lookup returns the
newest match and how long ago it was written, and `WebSearch::run` compares
that against `SEARCH_CACHE_MAX_AGE_DAYS` itself. A live fetch that then fails
falls back to a stale entry rather than erroring — the same principle
`search::mod::fit` already applies to a budget too small for a whole result:
something beats nothing.

### The sandboxed shell

`crates/tools/src/bash.rs` is one tool over a shell that cannot touch this
machine. It POSTs the command to `deploy/just-bash/`, a small Node service
wrapping [just-bash](https://github.com/vercel-labs/just-bash) — a bash
*simulation* over a virtual filesystem, with `cat`, `grep`, `sed`, `jq`, `awk`,
`sqlite3` and a hundred others implemented in TypeScript and no `execve`
underneath any of them.

That is the whole reason the tool is allowed to exist. Handing a language model
a real shell on the box holding the database is not a thing to do; handing it a
simulated one costs nothing when the model writes `rm -rf /`, which it will.

```sh
cd deploy/just-bash
./run.sh                                  # installs on first run, then starts
curl -s localhost:8899/health             # {"ok":true,"sessions":0}
```

Then `BASH_URL=http://localhost:8899` in `backend/.env`. No container, unlike
SearXNG: the library *is* the sandbox, so there is nothing for one to isolate.
It needs Node ≥ 20.18.1 and binds loopback only — an unauthenticated command
endpoint has no business on a public interface even when its filesystem is
imaginary.

**One filesystem per chat.** The service keeps a sandbox per `session` key, the
chat id the backend sends, so `echo x > f.txt` in one round is readable in the
next. Sessions expire an hour after their last command and the oldest is
evicted past 64 of them (`BASH_SESSIONS`, `BASH_SESSION_TTL_SECS`); eviction is
silent, and the next command simply gets a fresh filesystem the way a rebooted
machine would.

**Everything optional is off.** No network, no Python, no JavaScript runtime,
in-memory filesystem only. Each is a capability, and a capability this project
has not decided to hand a language model is one it does not have — turning any
of them on is an edit in `server.mjs`, deliberately.

What the model is handed back mirrors the search tool's discipline: the output
in a `<|codeblock|>`, a non-zero exit named in a `<|warn|>` before it, silence
said rather than shown as an empty block, and the whole thing held to
`BASH_BUDGET_CHARS`. A command that overruns is cut at a line boundary and told
so, because a model that needs the end can say `| tail` — which is the habit
worth teaching anyway.

```sh
cargo run -p tools --example bash_once -- 'echo hello > f.txt; ls -l'
```

## Database conventions

- Fixed-value fields are real Postgres `ENUM`s mapped with `#[derive(sqlx::Type)]`,
  not `TEXT`.
- IDs are UUIDs (`gen_random_uuid()`, `Uuid` in Rust).
- Every mutable entity carries `created_at`, and `updated_at` where relevant.
- Migrations are additive and forward-only once data exists.

## The Python engine

`crates/service` is the only crate that knows a service is a subprocess: it
spawns one as a child of the Actix server, health-checks it before the server
reports ready, restarts it if it dies, and kills it on shutdown. Both Python
services are started this way.

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

The data engine takes the same set under `DATA_ENGINE_*`, read by the same
`service::Config::from_env`:

| Variable | Default | Meaning |
|---|---|---|
| `DATA_ENGINE_DIR` | `../data-engine` | |
| `DATA_ENGINE_PYTHON` | `$DATA_ENGINE_DIR/.venv/bin/python` | |
| `DATA_ENGINE_HOST` | `127.0.0.1` | |
| `DATA_ENGINE_PORT` | `8002` | 8001 is the LLM engine's |
| `DATA_ENGINE_AUTOSTART` | `true` | |
| `DATA_ENGINE_STARTUP_TIMEOUT_SECS` | `60` | |
| `DATA_ENGINE_SERVICE_SECRET` | unset | |

One variable belongs to neither service, because it guards the direction that
runs toward this server:

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_SERVICE_SECRET` | unset | Shared with the engine, which signs `aud: backend` tokens with it. Unset leaves `/internal` open |
| `BIND_ADDR` | `127.0.0.1` | What the server listens on. `0.0.0.0` in a container, where a published port cannot reach a loopback binding |

It has no `DATA_DIR`: its datasets are rows in `slate_data`, not a tree. That
variable is the engine's and is injected by `crates/engine` alone — and it now
points at artifacts only, since the text a run trains on comes back the other
way through `/internal`.

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
