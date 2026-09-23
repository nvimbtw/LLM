# One generic `service` crate, one thin crate per service

**2026-08-19**

`backend/crates/llm-engine` was split. The part that knows how to spawn a Python
process and forward signed requests to one moved to `crates/service`; what
remains in `crates/llm-engine` — and what `crates/data-engine` now is — names a
service and nothing else: its audience, its environment prefix, its port.

## What forced it

`crates/llm-engine/src/proxy.rs` already said *"Nothing here names the LLM engine. A
second service is a second `Proxy` with its own audience and its own routers
above it; this is the whole seam."* That was true of `proxy.rs`. It was not true
of the crate: `config.rs` read `LLM_ENGINE_DIR`, `LLM_ENGINE_PORT`, `LLM_ENGINE_*`;
`supervisor.rs` injected `DATA_DIR` and signed its health probe with a
crate-level `AUDIENCE` constant.

And a trap that only shows up at the second service: **Actix keys `app_data` by
type.** Registering two `service::Proxy` values means the second silently
replaces the first, and every route reaches whichever service was registered
last. That failure compiles, passes most tests, and shows up as a `/data` route
answering with something the LLM engine said.

## What was weighed

**A new crate depending on `llm-engine` for `Proxy`.** Least churn — `upstream.rs`
would not have moved at all. Rejected: `data-engine → llm-engine` is a dishonest
dependency edge. The two services are siblings that never speak to each other,
and a crate graph saying otherwise is the kind of thing a later reader believes.

**Copying `config.rs` and `supervisor.rs` into a standalone crate.** Fully
independent, zero risk to the working LLM-engine path. Rejected on R6: it duplicates
about five hundred lines including every trap worth keeping — the
absolute-but-never-`canonicalize`d interpreter path, the child-watching health
probe, the backoff that only resets once a replacement is serving. Two copies of
those must change together, and one day would not.

**Generics — `Proxy<S: Service>` with a marker type per service.** Slightly less
code than newtypes, and distinct types fall out for free. Rejected on taste this
repository has already committed to: Actix was chosen over Axum specifically to
avoid trait-bound complexity in route signatures, and `web::Data<Proxy<Engine>>`
in a dozen files is that complexity arriving through the back door.

## What was done

`crates/service` holds `Proxy`, `Frame`, `Error` and the SSE parser unchanged,
plus a `Config` carrying its own `audience`, `secret_var` and `extra_env`, built
by `Config::from_env(&Defaults)` from a prefix. `crates/llm-engine` and
`crates/data-engine` are about forty lines each: the audience, the defaults, and

```rust
pub struct Proxy(service::Proxy);
```

with a `Deref`. The newtype is the whole point — it is what makes the two
distinct types in `app_data`.

`crates/api` changed in one place: `upstream.rs` now imports the error from
`service` rather than through `llm-engine`. **The twelve route files that use
`web::Data<Proxy>` were not touched**, because `llm_engine::Proxy` is still a type
called `Proxy` exported from `llm-engine`.

`crates/api/tests/two_services.rs` is the guard: it stands up both proxies
behind one app with stubs that name themselves, and asserts `/data/datasets`
reaches the data engine while `/models` reaches the LLM engine. Deleting either
newtype makes it fail.
