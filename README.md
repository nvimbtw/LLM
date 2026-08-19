# Slate

An LLM trained, served and inspected end to end, on one machine: a PyTorch
engine that owns the weights, a Rust API in front of it, and a Next.js UI in
front of that. You build a vocabulary, queue a training run, watch it on the
job board, benchmark the model it produces, and then talk to it — without
leaving the app or handing anything to a third party.

Nothing here is a wrapper around somebody else's API. Every token the app shows
you was predicted by a model trained on this machine, on a corpus you chose.

```
frontend/     chat, the job board, model, data and benchmark views — Next.js, Bun
backend/      people, sessions, and the seam to every module — Rust, Actix, sqlx
engine/       training, inference, evaluation — PyTorch, FastAPI
data-engine/  acquiring datasets from Hugging Face and cleaning them — FastAPI
```

## Getting it running

**You will need** Rust (stable), Python 3.11+, [Bun](https://bun.sh), and
PostgreSQL. A CUDA GPU if you intend to train anything; inference on a small
model is bearable without one.

**Three databases.** Each module owns one and none reads another's — see
[`docs/decisions/`](docs/decisions/) for why. Create all three:

```sh
createdb llm              # the backend's: users, presets
createdb slate_engine     # the engine's: models, conversations, runs, scores
createdb slate_data       # the data engine's: datasets and their documents
```

Those are the names the checked-in defaults expect — `DATABASE_URL` in
`backend/.env`, `ENGINE_DATABASE_URL` (default
`postgresql+psycopg://localhost/slate_engine`) and `DATA_ENGINE_DATABASE_URL`
(default `postgresql+psycopg://localhost/slate_data`).

**The engine**, whose virtualenv the backend expects to find:

```sh
cd engine
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/alembic upgrade head
```

Torch's CUDA wheels are resolved by pip for the local platform, which is why
the `nvidia-*` packages are deliberately absent from `requirements.txt`.

**The data engine**, the same shape and its own virtualenv. It has no torch and
no tokenizers — it does not train, so its install is a fraction of the size:

```sh
cd data-engine
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/alembic upgrade head
```

**The backend** reads `backend/.env`, and `DATABASE_URL` must be set and the
database migrated *before the first build* — sqlx's query macros check every
query against the live schema at compile time, and the app's own migrations run
too late for that.

There is no `.env.example` in the tree; the keys are these:

```sh
cd backend
cargo run
```

| Key | Meaning |
| --- | --- |
| `DATABASE_URL` | The backend's Postgres. Read at build time *and* runtime. |
| `JWT_SECRET` | Signs user sessions. |
| `ENGINE_DIR`, `ENGINE_PYTHON`, `ENGINE_PORT` | Where the engine lives and how to start it. |
| `ENGINE_SERVICE_SECRET` | Authenticates the backend→engine hop. Unset leaves the engine open, and it says so at startup. |
| `ENGINE_AUTOSTART=false` | Don't spawn the engine; attach to one already running. |
| `DATA_DIR` | The engine's artifacts: packed corpora, vocabularies, weights. Not the text they were built from — that is the data engine's. |
| `BACKEND_SERVICE_SECRET` | Authenticates the engine→backend hop, the one that runs toward the backend. Unset leaves `/internal` open, and it says so at startup. |
| `DATA_ENGINE_*` | The same set as `ENGINE_*`, for the data engine. Port defaults to 8002. |
| `SEARCH_URL` | A SearXNG instance for the `web-search` tool. Optional — unset, no tool is registered and chat works exactly as before. |

**The frontend:**

```sh
cd frontend
bun install
bun run dev
```

That is the whole sequence — `cargo run` spawns both Python services as child
processes and health-checks each before reporting ready, so you do not normally
start them yourself. The UI is on **:3000**, the backend on **:8080**, the
engine on **:8001**, the data engine on **:8002**, and the optional search
instance below on **:8888**.

To run either standalone instead, set its `*_AUTOSTART=false` and:

```sh
cd engine      && DATA_DIR=./data .venv/bin/uvicorn main:app --port 8001 --reload
cd data-engine && .venv/bin/uvicorn main:app --port 8002 --reload
```

**Web search (optional).** Replies can call `<|web-search|>`, which the backend
executes. Self-hosted, so it needs no key and no account:

```sh
cd deploy/searxng && ./run.sh        # or: docker compose up -d
```

Then `SEARCH_URL=http://localhost:8888` in `backend/.env`. Leave it unset and
the tool simply is not registered — see
[`docs/backend/README.md`](docs/backend/README.md) for the two SearXNG settings
that must be right, and
[`docs/decisions/tool-calls.md`](docs/decisions/tool-calls.md) for why a turn
that calls one runs twice.

### Tests

```sh
cd engine   && .venv/bin/python -m pytest tests/    # incl. the dataflow trace
cd backend  && cargo test --workspace               # needs DATABASE_URL
cd frontend && bun run typecheck && bun run lint
```

`engine/tests/test_dataflow_trace.py` is the contract that keeps the engine
honest: one fixed question, recorded at five layer boundaries under a pinned
seed. Re-record it only when a behaviour change is intended, and commit the diff
as the evidence of what moved.

## Architecture: a Rust host, and modules behind it

The backend is not a proxy with some auth bolted on. It is the **host**: it owns
people and sessions, it is the only thing the frontend ever talks to, and every
capability the product has arrives as a **module** sitting behind it.

Today there is one module — the engine — but the seam it plugs into is the
architecture, not an accident of having a Python process to talk to.

```
frontend ──► backend ──► module          the only shape a request ever takes
                    └──► module
```

**What a module is.** A service that owns one domain completely, including its
own database, reached only through the backend. The engine owns models,
conversations, runs and scores; the backend owns users and presets and models
none of what the engine owns. It stores an id and asks.

**What the seam looks like**, concretely, in
[`backend/crates/engine/`](backend/crates/engine/):

- **`config`** — where the process lives, what port it wants, how to start it.
- **`supervisor`** — spawns the process, keeps it alive, health-checks it before
  the backend reports ready, and shuts it down with the app.
- **`proxy`** — forwards requests, signing each one with a short-lived service
  token that names the acting user. The token's shape is pinned for both
  languages by [`contracts/service-token.json`](contracts/service-token.json),
  so the Rust side and the Python side cannot drift apart silently.

A module is therefore roughly: a crate that configures, supervises and proxies;
a service that accepts the signed token and enforces it; a database it alone
migrates; and routes on the backend that forward rather than reimplement.

**The rules that keep this honest**, and the reason the split is worth its cost:

1. **One owner per fact.** If a module owns a table, nothing else reads it — not
   with a join, not "just for a count". Cross-module questions are asked over
   the wire, and the answer is whatever the owner says.
2. **The backend models nothing a module owns.** The moment it holds a mirror of
   a module's schema, there are two truths and a sync bug waiting.
3. **The frontend never talks to a module.** One origin, one session, one place
   where authorization is decided.
4. **The hop is authenticated.** A module trusts the backend's token, not the
   network.

### Modules, present and planned

| Module | Owns | Status |
| --- | --- | --- |
| **Engine** | Models, conversations, training runs, benchmark scores, the weights | **Built** — `engine/`, behind `backend/crates/engine/` |
| **Data** | Datasets from the Hub, the documents in them, and the cleaning that produces new ones | **Built** — `data-engine/`, behind `backend/crates/data-engine/` |
| **Benchmark** | Suites, scoring, held-out sets, comparison across models | **Planned** |

Benchmarking is not a stub — there is no separate code for it yet; it is engine
routes plus the `/benchmark` views. Pulling it out is the near-term direction of
the work, and the seam above is what it will be pulled out *into*.

The data engine is what training reads: a run selects datasets by id and the
engine fetches their documents a page at a time through the backend, keeping no
copy. The scripts in [`scripts/`](scripts/) that prepared folders of `.txt` are
superseded by pulling from the Hub in the Data view — see
[`docs/decisions/datasets-out-of-the-engine.md`](docs/decisions/datasets-out-of-the-engine.md).

When you build one, the engine is the worked example: read
[`backend/crates/engine/`](backend/crates/engine/) end to end first. It is small
on purpose.

## Where to look next

- **[`AGENTS.md`](AGENTS.md)** — the entry point for anyone, human or agent,
  about to change something here. How to work, and how to document what you
  changed.
- **[`docs/`](docs/)** — one directory per module: what it owns, its stack,
  layout, schema, routes, conventions and gotchas. The frontend renders it at
  `/docs`.
- **[`docs/conventions.md`](docs/conventions.md)** — the coding rules every
  change is judged against.
- **[`docs/decisions/`](docs/decisions/)** — what was weighed and rejected,
  dated, once. Read this before re-litigating a boundary.
