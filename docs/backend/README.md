# Backend — Rust API

The Rust half: people, sessions, and the seam to the services behind it. It owns
users, presets, private projects, conversations, messages, uploads and usage. Models/training and
queue scheduling belong to the LLM engine; benchmark scoring/history belongs to
bench-engine. See [conversations.md](conversations.md) and
[ownership-cutover.md](ownership-cutover.md).

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
cargo sqlx prepare --workspace -- --workspace --all-targets
```

Cargo fingerprints `.rs` files and not `migrations/`, so a migration edit alone
does not force the crate whose queries it invalidates to rebuild. A stale
`.sqlx/` shows up as a container build failing while `cargo run` on the host
still succeeds.

## Crates

```
backend/
├── Cargo.toml         workspace root AND the binary package
├── src/main.rs        server lifetime and binding
├── src/startup.rs     supervisor ownership, typed proxy wiring, CORS
└── crates/
    ├── auth/          argon2 hashing, the token cipher, JWT issue/verify
    ├── db/            users, presets, conversations — sqlx + migrations/      → auth
    ├── service/       spawning a Python service, and the hops both ways → auth
    ├── llm-engine/    names the LLM engine: audience, port, DATA_DIR → service
    ├── data-engine/   names the data engine                          → service
    ├── bench-engine/  names the benchmark service                   → service
    ├── gyms-engine/   names the gyms engine                          → service
    ├── tools/         what a model can reach for, and the answer  → (nothing)
    └── api/           routes + extractors + sockets   → service, llm-engine, data-engine, gyms-engine, tools, db, auth
```

`tools` is a leaf like `auth`: it knows about the outside world and about the
marker vocabulary, and nothing about this project's users, sessions or the LLM engine
process.

`service` names no service. `llm-engine`, `data-engine`, `gyms-engine` and `bench-engine` are ~40 lines
each — an audience, a set of defaults, and a `Proxy` newtype — and do **not**
depend on each other: the four Python processes are siblings that never speak.

The database holds `users`, `training_presets`, `projects`, `chats`, `messages`, `attachments`
and the migration ledger. Conversation ownership/cascades and stream persistence
are described in [conversations.md](conversations.md).

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

## Database conventions

- Fixed-value fields are real Postgres `ENUM`s mapped with `#[derive(sqlx::Type)]`,
  not `TEXT`.
- IDs are UUIDs (`gen_random_uuid()`, `Uuid` in Rust).
- Every mutable entity carries `created_at`, and `updated_at` where relevant.
- Migrations are additive and forward-only once data exists.
- Applied migration files are immutable, including comments and whitespace:
  sqlx checksums the entire file and refuses startup with `VersionMismatch` if
  it changes. Historical names and paths in them must stay as written. Restore
  accidentally edited files to their applied versions and recompile the `db`
  crate to refresh the embedded migrations; do not rewrite the database's
  migration checksums to accept an edit.

## Public surface and focused topics

| Topic | Contract |
| --- | --- |
| [HTTP and proxy routes](api.md) | Authorization, service-owned payloads, upload limits and route distinctions. |
| [Conversations and files](conversations.md) | Owner isolation, atomic input persistence, commit-before-acknowledgement, retry identity and disconnect recovery. |
| [Private projects and memories](projects.md) | Owner-scoped project metadata, chat membership, indexed word matching and trusted engine context. |
| [Internal service routes](internal.md) | Issuer-scoped reverse calls, actor propagation and shared gym observations. |
| [Chat tools](tools.md) | Bounded tool rounds, concurrent execution, interjections, search cache and sandbox policy. |
| [Gym sockets](gyms.md) | Sequential episode commands, terminal grading and abandonment. |
| [Service lifecycle](services.md) | Native configuration, health checks, service registration and SSE decoding. |
| [Ownership cutover](ownership-cutover.md) | Existing-installation transfer and rollback procedure. |

## Proxy routes

Service-owned JSON remains opaque and service statuses pass through. See
[HTTP contracts](api.md); named service proxy types prevent Actix registration
collisions. A user JWT is never forwarded to a service.

## The hop that runs the other way

`ServiceCaller` admits LLM dataset/gym requests; `BenchCaller` admits benchmark
measurement requests. Both preserve the actor identity through Rust. See
[reverse-hop authentication](internal.md).

### The gym family

The browser driver and RL reverse route share `gym::play` and
`tools::observation`. See [internal gym routes](internal.md#the-gym-family).

## Adding a service

Add a thin service crate, supervisor/proxy wiring and routes. See
[registration and the type-identity trap](services.md#adding-a-service).

## Tool calls

`turn.rs` coordinates a whole locked conversation turn; its sibling modules
own follow-up policy, tool task lifetimes and streaming persistence. Tool-call
fields are interpreted once into a typed value with the existing empty-string
fallback for malformed legacy fields. See [tools](tools.md).

### Interjecting mid-turn

[Steering and its current acknowledgement limitations](tools.md#interjecting-mid-turn).

### Web search

[Provider selection and deployment](tools.md#web-search).

#### Sizing the result

[Observation sizing for the model context](tools.md#sizing-the-result).

### The search cache

[Data-engine storage and freshness policy](tools.md#the-search-cache).

### The sandboxed shell

[Virtual filesystem capability and session policy](tools.md#the-sandboxed-shell).

## The gym socket

`gym.rs` coordinates runs while focused modules own episode generation, request
policy and observation rendering. Chat and gym keep distinct loops because their
persistence and tool execution contracts differ. See [gym runs](gyms.md).

## The Python services

`src/startup.rs::Services` owns supervisors for the server lifetime; workers get
concrete proxy types through Actix. See [configuration and lifecycle](services.md).

## Error handling

Routes map failures to status codes and log internal causes. A 500 body says
what failed, not the database or authentication details behind it. Stored
conversation responses and opaque upstream responses remain distinct adapters.

## Code style

Follow the complete [C1–C12 conventions](../conventions.md). Brief `///` contracts
belong on named functions, including private helpers; bodies contain no comments.
The API orchestration layer separates pure request/frame policy from I/O without
moving SQL or service-owned domain behavior into Rust.

## Verification and remaining limits

From `backend/`, run `SQLX_OFFLINE=true cargo build`,
`SQLX_OFFLINE=true cargo clippy --workspace --all-targets -- -D warnings` and
`SQLX_OFFLINE=true cargo test --workspace` with `DATABASE_URL` set to a disposable
PostgreSQL instance. SQLx integration tests create their own databases; do not
point verification at a production database. Run formatter checks for changed
files; existing unrelated workspace formatting debt is tracked by the audit.

The per-chat lease protects one backend process only. Interjection queues,
stream buffers and turn-lock admission waits still need explicit budgets.
Browser socket disconnect is detected through existing read/write behavior;
reader cleanup alone does not cancel durable jobs or an upstream generation.
The gym driver retains its existing best-effort status writes and error behavior,
including possible grading after a truncated stream. Those are separate behavior
changes requiring dedicated regression coverage.
