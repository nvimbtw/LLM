# HTTP and proxy contracts

A route for something a service owns forwards rather than re-models it. The
payload is `serde_json::Value` on purpose — a field the LLM engine adds needs no
change here, and `upstream::Upstream` carries the service's own status back, so
a 404 stays a 404 instead of becoming a generic 502.

`/attachments` stores uploads in the backend database with conversation ownership.
The multipart field and database both enforce 50 MiB; downloads return the saved
bytes, media type and filename. Files never enter model prompts. See
[conversations.md](conversations.md) for binding and authorization rules.

**One proxy route is `AuthUser` where its neighbours on the same resource are
`AdminUser`.** `POST /models/{id}/attention` forwards to the LLM engine's attention
endpoint, and it is not admin work: it is one forward pass over a model anyone
signed in can already hold a conversation with, and reading where one of its
heads looks is strictly less than that. Editing or deleting a model next door
stays admin, because those change what everyone else sees. What keeps a
quadratic response off an open route is the LLM engine's own bound on the sequence
(see [`../llm-engine/README.md`](../llm-engine/README.md)) — a long text comes back as a
400, so nothing here has to restate a ceiling it would then have to keep in step.

`GET /models/{id}/attention-points` also takes `AuthUser` and preserves the verified
actor, derived public/administration scope and engine status through bounded JSON.
It discovers actual declared occurrence views or saved legacy dimensions, without
exposing authoring source/graphs. Discovery may load the engine's numerical model
and is an explicit user action. The opaque attention POST also accepts point/head
selection; matrix calculations and runtime evidence remain engine-owned.

**Two routes are spelled almost alike and are not the same thing.**
`POST /runs/tokenize` queues a vocabulary *build* — minutes of CPU, admin work,
a job with a page. `POST /tokenize/count` measures a string under a model's
existing vocabulary, loads no weights, and takes `AuthUser`, because the
composer's context meter asks it on every pause in typing
([`../frontend/README.md`](../frontend/README.md)). Gating the second one behind
`AdminUser` by analogy with the first would leave every ordinary session's meter
collecting 403s.

**One family of proxy routes has a driver behind it.** `/gym/*`
(`routes/gym.rs`, admin only) forwards to the gyms engine like any other, but a
gym *run* is not a request — it is a loop, and `api/src/gym.rs` is where that
loop lives. See [the gym socket](gyms.md) below.

**And one gym loop has no socket at all.** `POST /runs/rl` queues a run in the
LLM engine that drives its own episodes through [the gym family](internal.md#the-gym-family) of
`/internal`, with nobody watching. It is beside `/runs/train` and `/runs/sft`
because it is the same thing to this side — a row in the queue that produces a
model — and the only reason it is not `/runs/train` with a flag is that what it
*reports* is a pass rate rather than a loss.

The one thing this side decides is **who is asking**: `upstream::caller` turns
the verified user token into the `act` claim of a short-lived service token, and
the service scopes by that. A user token is never forwarded.

`POST /runs/train`, `/runs/sft`, `/runs/rl`, `/runs/chain`, `/runs/{id}/retry` and `/runs/{id}/recover`
also sign the verified administrator's `administration` scope, forward query
values such as `request_id` unchanged, and preserve the engine's exact status
and bounded JSON body. Recipe parameters, accepted identities and idempotency
remain engine-owned. Rust does not interpret a selected architecture. The
engine's recipe admission is partial: actual supervised training/save/recovery
integration remains open, as documented in its [API contract](../llm-engine/api.md).

## Vocabulary display metadata

`GET /tokenizers/{id}` includes the vocabulary's facts and recorded model
references. `PATCH /tokenizers/{id}` is admin-only and forwards the engine-owned
payload as the system caller. It accepts `display_name` (trimmed, nonblank, at
most 120 characters) and an optional nullable `description` (at most 2,000).
Duplicate names return 409 without partially saving the description; unknown
identities return 404. UUIDs, files, fingerprints and training references remain
unchanged. The backend owns no tokenizer metadata table.

## Processor source

`GET /data/processors/{name}/source` requires `AuthUser` and forwards to the
data engine's source reader. `PUT` at the same path requires `AdminUser`; browser
access-level display alone cannot authorize a write. Source JSON stays service
owned and the hop replaces browser tokens with audience-scoped service tokens.

Only known custom files are writable. Builtins are read-only. Saves persist
invalid code and return saved/active hashes plus a reload error while preview
continues with the last working callback. See the
[data-engine source contract](../data-engine/processors.md).

`POST /data/processors` also requires `AdminUser`. It forwards a full source file
for creation; the data engine validates the first definition, derives its stable
name, prevents collisions and owns its durable file. Ordinary-user POSTs and PUTs
are rejected in Rust before reaching the service.

## Modular model authoring and private validation

The schema response includes prospective installed runtime defaults, with no
validation claim. `POST /architectures/{id}/declaration` forwards exact saved-draft
identity under administrator/actor scope; the engine parses YAML metadata without
custom imports or model construction. Its output never authorizes acceptance.
Cancel, reorder and delete for `/runs/{id}` also preserve administrative scope and
actor identity so private validation jobs remain controllable. These hops share
the bounded status-preserving administration transport with authoring/submission.

`/blocks` and `/architectures` are administrator-only engine catalog adapters.
Each supports GET/POST collection, GET/PATCH resource metadata, GET/PUT draft,
GET/POST revisions, GET a pinned revision and POST validations. UUIDs are parsed
at the Rust boundary. `/architectures/schema` is registered before the resource
UUID matcher. These routes keep source/YAML, parameters and revision validation
opaque; Rust owns no corresponding storage or graph calculation.

`GET /blocks/{id}/revisions/{revision_id}/copy-source` returns eligible custom
wrapper source for a pinned installed builtin, without creating a resource.
`GET /tokenizers/{id}/architecture-binding` returns authoritative full-hash
tokenizer facts and forwards bounded repeated `reserved` marker-name queries.
Both require `AdminUser` and retain the actor-preserving administration hop.

`GET /validations/{id}`, `/validations/{id}/graph`, `/models/{id}/definition`
and `/models/{id}/graph` also require `AdminUser`. Frozen text reads at
`/models/{id}/sources/{revision_id}` and `/validations/{id}/sources/{revision_id}`
use the same actor-preserving administrative transport and UUID validation.
Engine JSON bodies and all
successful/error statuses pass through unchanged, including 201 acceptance and
202 enqueue. Catalog requests admit at most 1 MiB of encoded JSON; the engine
separately enforces its 256 KiB source/YAML ceiling. The JSON transport caps
responses at 8 MiB and expires each request after 30 seconds. Saving a draft or
receiving an enqueue response never transfers worker ownership to the HTTP
waiter.

Rust derives the service JWT `scope` array from verified browser claims:
`["administration"]` for administrators and `["public"]` for other users. The
existing audience and `act` identity remain intact. Catalog/graph routes always
use administration. Generic model/run reads and `/ws/runs/{id}` event requests
carry the same derived scope; browser query fields cannot grant it. Python treats
missing or unrecognized scopes as public, filters validation jobs before history
pagination/counting, and denies private detail/events before replay. Public
model/run data and telemetry are redacted by the engine rather than implementing
a second model schema in Rust. Socket admission errors are generic for public
observers; administrators retain actionable service errors. Disconnect only
ends observation; explicit cancellation remains a separate operation.

`architecture_proxy.rs` tests the Rust route, signed identity, status, request
budget and real WebSocket seam against an upstream fixture. These tests do not
certify the full live authoring/validation/package APIs: ISS-56, ISS-58 and
ISS-61 remain required FEAT-4 integration gates.

Run the opt-in live privacy seam from the repository root with:

```sh
SQLX_OFFLINE=true cargo test --manifest-path backend/Cargo.toml -p api --test architecture_proxy live_python -- --ignored --nocapture
```

It starts the real engine auth/run/model routers in a supervised fixture process
with unique SQLite storage, synthetic durable rows and no model execution or
main application lifespan. It checks Rust role encryption/service tokens,
pre-pagination totals, source redaction, private SSE denial and real browser
WebSocket replay. `SLATE_ENGINE_PYTHON` may select an installed engine interpreter;
the default is `engine/.venv/bin/python`. The fixture process is stopped/reaped
and its scratch directory removed when the test finishes. This privacy contract
check does not replace the remaining full engine feature gates above.

## Public published-model browsing

`GET /models`, `/models/{id}` and `/models/{id}/checkpoints` accept a missing
browser Authorization header. Rust signs these anonymous reads as the system
actor with explicit public scope. A supplied invalid or malformed bearer still
returns 401; it never falls back to anonymous access. Verified ordinary users
retain their actor with public scope; administrators receive administration scope.

The engine authorizes publication before projecting a detail or checkpoints.
Private and nonexistent IDs both return 404 to public readers. Public lists
exclude private rows and unindexed artifact directories, regardless of query
flags. Public checkpoint lists contain only published checkpoints of a published
parent. Opaque configuration, training sources, run/parent identities, recovery
flags, artifact paths and saved source/graphs remain private. Model detail adds
benchmark records only for administrators. Signed-in attention and benchmark
reads first check the same model visibility. Chat and attention computation
remain login-gated; model mutations and saved-source/graph reads remain admin-only.
