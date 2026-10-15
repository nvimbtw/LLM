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

## Vocabulary display metadata

`GET /tokenizers/{id}` includes the vocabulary's facts and recorded model
references. `PATCH /tokenizers/{id}` is admin-only and forwards the engine-owned
payload as the system caller. It accepts `display_name` (trimmed, nonblank, at
most 120 characters) and an optional nullable `description` (at most 2,000).
Duplicate names return 409 without partially saving the description; unknown
identities return 404. UUIDs, files, fingerprints and training references remain
unchanged. The backend owns no tokenizer metadata table.
