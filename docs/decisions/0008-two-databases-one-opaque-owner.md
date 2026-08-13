# Two databases, one opaque owner

Each module owns its data outright. The Rust backend owns people, sessions and
presets; the Python engine owns models, conversations, runs, scores and the
queue. Nothing reads across the line — the backend forwards.

## Why the queue moved

It was implemented twice. `crates/jobs::queue` enforced one-job-at-a-time in
Rust, and the engine's own registry enforced it again in Python. Two answers to
one question, and only one of them was adjacent to the GPU: a run started
against the engine directly bypassed the Rust queue entirely, and the Rust queue
could not see a device the engine was already holding.

The engine's copy is the one that survives, because that is where the device is.

## The owner is an id and nothing else

`chats.owner_id` is a `users.id` from the *other* database, with no foreign key
and no validation. The engine cannot resolve it and never tries.

That is what keeps the two independent. The consequences are deliberate:

- A chat whose owner is gone is **invisible, not broken** — no query returns it,
  nothing fails.
- Deleting an account is two writes in two places. The engine's `DELETE
  /owners/{id}` runs **first** and is **idempotent**, so a half-failed deletion
  is fixed by retrying it. The other order would leave chats owned by an id
  nothing can resolve.
- Every query that scopes by owner takes it as an argument
  (`store/queries.py`), so forgetting to pass one is a type error rather than a
  silent full-table read.

## The backend never forwards a user token

A request arrives with the user's encrypted JWT (`aud: "web"`). The backend
verifies it, then mints a **new** short-lived token (`aud: "engine"`, 60s)
naming who it acts for in `act`. Two audiences, two secrets, and a token for one
never opens the other — `contracts/service-token.json` is the shared vector both
sides verify against.

A route that acts on nobody's behalf — deleting a model, cancelling a run — signs
as `system`, and the engine refuses to aim an owner-scoped route with that.

## What the backend does not do

It does not re-model anything the engine owns. A proxy route forwards a
`serde_json::Value` and answers with what came back, **status included**: an
engine 404 flattened into a 502 tells a client nothing about what went wrong.
The one thing this side decides is who is asking.
