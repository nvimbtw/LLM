# Service processes and configuration

`crates/service` is the only crate that knows a service is a subprocess: it
can spawn one as a child of the Actix server, restarts it if it dies, and kills
it on shutdown. All four Python services can instead be managed externally.
The API starts while service connections are still being established, so an
unavailable engine can be reported on the administrator dashboard.

Configured by environment, because these are properties of the machine:

| Variable | Default | Meaning |
|---|---|---|
| `LLM_ENGINE_DIR` | `../llm-engine` | Working directory; `main:app` resolves against it |
| `LLM_ENGINE_PYTHON` | `$LLM_ENGINE_DIR/.venv/bin/python` | Interpreter; started as `-m uvicorn` |
| `LLM_ENGINE_HOST` | `127.0.0.1` | |
| `LLM_ENGINE_PORT` | `8001` | 8000 is commonly taken by a local Apache |
| `LLM_ENGINE_AUTOSTART` | `true` | Off = "an LLM engine is already running, do not spawn" |
| `LLM_ENGINE_STARTUP_TIMEOUT_SECS` | `60` | An owned process that survives this long resets the restart backoff |
| `LLM_ENGINE_SERVICE_SECRET` | unset | Shared with the LLM engine; unset leaves both open |

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

The gyms engine follows the same contract under `GYMS_ENGINE_*`:

| Variable | Default | Meaning |
|---|---|---|
| `GYMS_ENGINE_DIR` | `../gyms-engine` | |
| `GYMS_ENGINE_PYTHON` | `$GYMS_ENGINE_DIR/.venv/bin/python` | |
| `GYMS_ENGINE_HOST` | `127.0.0.1` | |
| `GYMS_ENGINE_PORT` | `8003` | |
| `GYMS_ENGINE_AUTOSTART` | `true` | |
| `GYMS_ENGINE_STARTUP_TIMEOUT_SECS` | `60` | |
| `GYMS_ENGINE_SERVICE_SECRET` | unset | |

Bench-engine follows the same contract under `BENCH_ENGINE_*`, with directory
`../bench-engine`, port `8004`, and audience `bench-engine`. See
[its module doc](../bench-engine/README.md) for database and corpus configuration.

One variable guards the direction that
runs toward this server:

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_SERVICE_SECRET` | unset | Shared with the LLM and bench engines, which sign issuer-scoped `aud: backend` tokens with it. Unset leaves `/internal` open |
| `BIND_ADDR` | `127.0.0.1` | What the server listens on. `0.0.0.0` in a container, where a published port cannot reach a loopback binding |

It has no `DATA_DIR`: its datasets are rows in `slate_data`, not a tree. That
variable is the LLM engine's and is injected by `crates/llm-engine` alone — and it now
points at artifacts only, since the text a run trains on comes back the other
way through `/internal`.

Two things that are load-bearing and non-obvious:

- **`LLM_ENGINE_PYTHON` is made absolute but never `canonicalize`d.** A virtualenv's
  `bin/python` is a symlink to the system interpreter, and Python derives
  `sys.prefix` from the path it was invoked by — resolving the link hands over
  an interpreter with no idea it is in a venv, and `-m uvicorn` fails against a
  venv where uvicorn is plainly installed.
- **Absolute at all, because `Command` resolves the program path *after*
  chdir-ing into `current_dir`.** A relative `LLM_ENGINE_PYTHON` would be looked up
  under `$LLM_ENGINE_DIR/`, failing with a bare ENOENT.

Autostart off is the normal development setup: run `uvicorn main:app --reload`
in another terminal and the server attaches to it.

## Availability monitoring

`service::health::Monitor` owns one connection per engine and one HTTP probe per
configured backend tool. It sends `ping` text frames to each authenticated
`/health/socket` and expects `pong` within 15 seconds, every 5 seconds. A lost
connection or unsuccessful probe is `reconnecting`; after 60 seconds since the
last success it is `offline`. Retries continue, and a successful heartbeat or
probe immediately returns the service to `online`. Timeout and retry scheduling
can delay an outage transition. These states describe reachability from the
backend, not process identity or dependency readiness. In particular, an
unrelated process listening on a configured port may answer a probe; operators
should resolve port conflicts before starting an autostart service.

The monitor checks just-bash's `/health` JSON and SearXNG's HTTP root. Every
60 seconds it also checks SearXNG's JSON search API; its failure is shown as
detail while HTTP availability remains online. This test may contact external
search engines. Optional `BASH_URL` and `SEARCH_URL` absent means `not_configured`;
if Brave is selected, the SearXNG row names that provider without probing it.
The just-bash result covers the backend's configured endpoint only, not the
gyms engine's independently configured `BASH_URL`.

`/ws/services` authenticates the browser's first token frame and requires an
administrator. It publishes complete snapshots on change and every 15 seconds;
the browser discards stale snapshots and reconnects after a disconnect. The
monitor stops with the server. The common Python socket authentication contract
is exercised by `contracts/service-health/test_engine_socket.py` from each
engine working directory, with `HEALTH_TEST_AUDIENCE` set to that engine's name.

## Adding a service

`service` names no service, and `crates/data-engine` is the worked example. Another one is:

1. a crate beside `llm-engine/` and `data-engine/` naming its `AUDIENCE`, its
   `Defaults` (env prefix, directory, port), and a `Proxy` newtype,
2. a supervisor in `src/startup.rs::Services`, and a distinct proxy registered by
   `Application::configure` as `app_data`,
3. route files above it, mounted in `routes/mod.rs`.

The generic `service` and `auth` crates do not change; the composition root and
API route map add the new sibling explicitly.

**The newtype is not ceremony.** Actix keys `app_data` by type, so two bare
`service::Proxy` values silently overwrite each other and every route reaches
whichever was registered last — a failure that compiles and passes most tests.
`crates/api/tests/two_services.rs` is what notices. See
[`../decisions/a-crate-per-service.md`](../decisions/a-crate-per-service.md).

## Composition and event decoding

`src/main.rs` owns the `Services` value until the HTTP server drains. Its four
supervisors own their child lifetimes; worker construction only clones typed
proxy handles. Explicit graceful shutdown requests stop all siblings, and
supervisor `Drop` remains the fallback on startup/server failure.

`service::Proxy` shares signed request construction and status handling across
JSON, uploads, binary downloads and event streams. Service-owned JSON stays
opaque. `proxy/sse.rs` owns partial frame bytes; it decodes UTF-8 only once a
complete frame exists, preserving characters split across network reads.

The SSE adapter retains the current service subset: LF blank-line delimiters,
`event: ` names and concatenated `data: ` JSON. Unknown event names pass through;
invalid JSON becomes null; transport errors are visible stream items; unfinished
trailing events are withheld. CRLF framing is not supported. The partial-frame
buffer and response bodies still lack local size bounds; service limits are not
a substitute for a future transport budget.
