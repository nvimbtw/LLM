# Slate

An LLM trained, served and inspected end to end, on one machine: a PyTorch LLM
engine that owns the weights, a Rust API in front of it, and a Next.js UI in
front of that. You build a vocabulary, queue a training run, watch it on the
job board, benchmark the model it produces, and then talk to it — without
leaving the app or handing anything to a third party.

Nothing here is a wrapper around somebody else's API. Every token the app shows
you was predicted by a model trained on this machine, on a corpus you chose.

```
frontend/     chat, the job board, model, data and benchmark views — Next.js, Bun
backend/      people, sessions, and the seam to every module — Rust, Actix, sqlx
llm-engine/   training, inference, evaluation — PyTorch, FastAPI
data-engine/  acquiring datasets from Hugging Face and cleaning them — FastAPI
gyms-engine/  generated tasks, sandboxed execution and scoring — FastAPI
```

## Getting it running

```sh
cp .env.example .env    # then fill in the secrets it names
docker compose up
```

That is the whole thing: Postgres, three Python services, the Rust backend, the
UI and a SearXNG instance, on the ports the map below names. Every service
hot-reloads from the source tree, so an edit lands the same way it does natively
— and one can be restarted without disturbing the others:

```sh
docker compose restart backend
docker compose logs -f llm-engine
```

The only prerequisites are Docker and the Compose plugin. It comes up CPU-only:
**to train, uncomment the `deploy:` block on the `llm-engine` service** in
`docker-compose.yml` and install
[nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host. That block is commented out rather than the other way around
because the reservation is not a preference — Compose refuses to start the
service at all on a machine with no such device.

Three things worth knowing before the first run:

- **The LLM engine image is several GB** — torch resolves to a `+cuXXX` build and
  pulls the CUDA wheels with it, whether or not the machine has a GPU. It is
  built once. Everything else is small.
- **Budget ~10 GB free wherever Docker stores images**, which is not always
  where you think. With the containerd image store — `docker info` saying
  `Storage Driver: overlayfs` — layers live under `/var/lib/containerd`, and
  Docker's own `data-root` setting does not move them.
- **Postgres publishes on 5433**, not 5432, because a development machine
  usually already has a server there. Inside the compose network it is 5432.

Why several containers rather than one running everything:
[`docs/decisions/one-command-development.md`](docs/decisions/one-command-development.md).

### Running it natively

Still supported, and still the better loop if you are working on the LLM engine
itself.

**You will need** Rust (stable), Python 3.13+, [Bun](https://bun.sh), and
PostgreSQL. A CUDA GPU if you intend to train anything; inference on a small
model is bearable without one.

**Four databases.** Each module owns one and none reads another's — see
[`docs/decisions/`](docs/decisions/) for why. Create all four:

```sh
createdb llm              # the backend's: users, presets
createdb slate_engine     # the LLM engine's: models, conversations, runs, scores
createdb slate_data       # the data engine's: datasets and their documents
createdb slate_gyms       # the gyms engine's: runs, episodes, steps and scores
```

Those are the names the checked-in defaults expect — `DATABASE_URL` in
`backend/.env`, `LLM_ENGINE_DATABASE_URL` (default
`postgresql+psycopg://localhost/slate_engine`), `DATA_ENGINE_DATABASE_URL`
(default `postgresql+psycopg://localhost/slate_data`) and
`GYMS_ENGINE_DATABASE_URL` (default `postgresql+psycopg://localhost/slate_gyms`).

**The LLM engine**, whose virtualenv the backend expects to find:

```sh
cd llm-engine
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

**The gyms engine** owns generated tasks, sandboxed episodes, and their scores:

```sh
cd gyms-engine
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
| `LLM_ENGINE_DIR`, `LLM_ENGINE_PYTHON`, `LLM_ENGINE_PORT` | Where the LLM engine lives and how to start it. |
| `LLM_ENGINE_SERVICE_SECRET` | Authenticates the backend→LLM-engine hop. Unset leaves the LLM engine open, and it says so at startup. |
| `LLM_ENGINE_AUTOSTART=false` | Don't spawn the LLM engine; attach to one already running. |
| `DATA_DIR` | The LLM engine's artifacts: packed corpora, vocabularies, weights. Not the text they were built from — that is the data engine's. |
| `BACKEND_SERVICE_SECRET` | Authenticates the LLM engine→backend hop, the one that runs toward the backend. Unset leaves `/internal` open, and it says so at startup. |
| `DATA_ENGINE_*` | The same set as `LLM_ENGINE_*`, for the data engine. Port defaults to 8002. |
| `GYMS_ENGINE_*` | The same set as `LLM_ENGINE_*`, for the gyms engine. Port defaults to 8003. |
| `SEARCH_URL` | A SearXNG instance for the `web-search` tool. Optional — unset, no tool is registered and chat works exactly as before. |
| `BASH_URL` | A just-bash sandbox for the `bash` tool. Optional, and the same bargain: unset, the tool is not registered. |

**The frontend:**

```sh
cd frontend
bun install
bun run dev
```

That is the whole sequence — `cargo run` spawns all three Python services as child
processes and health-checks each before reporting ready, so you do not normally
start them yourself. The UI is on **:3000**, the backend on **:8080**, the
LLM engine on **:8001**, the data engine on **:8002**, the gyms engine on
**:8003**, and the two optional tool
services below on **:8888** (search) and **:8899** (the shell sandbox).

To run a Python service standalone instead, set its `*_AUTOSTART=false` and:

```sh
cd llm-engine  && DATA_DIR=./data .venv/bin/uvicorn main:app --port 8001 --reload
cd data-engine && .venv/bin/uvicorn main:app --port 8002 --reload
cd gyms-engine && .venv/bin/uvicorn main:app --port 8003 --reload
```

**Tools (optional).** A reply can call one, as
`<|tool-call|>{"tool": …, "argument": …}<|/tool-call|>`, and the backend runs
it. Both are self-hosted — no key, no account, nothing metered. `docker
compose up` starts both, and points the backend at them itself. To run either
beside a natively run backend instead:

```sh
cd deploy/searxng   && ./run.sh      # web search  (or: docker compose up -d searxng)
cd deploy/just-bash && ./run.sh      # a sandboxed shell (or: … up -d just-bash)
```

Then `SEARCH_URL=http://localhost:8888` and `BASH_URL=http://localhost:8899` in
`backend/.env` — the host addresses, which is what a native backend needs. A
containerised one cannot use them (its `localhost` is itself), so compose sets
both to service names and those win. Leave either unset and that tool simply is
not registered. The
shell is a *simulation* — a virtual filesystem in a Node process, no `execve`
anywhere under it — which is what makes it safe to point a language model at.

See [`docs/backend/README.md`](docs/backend/README.md) for the two SearXNG
settings that must be right and what the sandbox can and cannot run,
[`docs/decisions/tool-calls.md`](docs/decisions/tool-calls.md) for why a turn
that calls one runs twice, and
[`docs/decisions/generic-tool-calls.md`](docs/decisions/generic-tool-calls.md)
for why every tool shares one marker.

### Tests

```sh
cd llm-engine   && .venv/bin/python -m pytest tests/    # incl. the dataflow trace
cd data-engine  && .venv/bin/python -m pytest tests/
cd gyms-engine  && .venv/bin/python -m pytest tests/
cd backend  && cargo test --workspace               # needs DATABASE_URL
cd frontend && bun run typecheck && bun run lint
```

`llm-engine/tests/test_dataflow_trace.py` is the contract that keeps the LLM engine
honest: one fixed question, recorded at five layer boundaries under a pinned
seed. Re-record it only when a behaviour change is intended, and commit the diff
as the evidence of what moved.

## Architecture: a Rust host, and modules behind it

The backend is not a proxy with some auth bolted on. It is the **host**: it owns
people and sessions, it is the only thing the frontend ever talks to, and every
capability the product has arrives as a **module** sitting behind it.

Today there are three modules — the LLM engine, data engine, and gyms engine —
and each plugs into the same seam.

```
frontend ──► backend ──► module          the only shape a request ever takes
                    └──► module
```

**What a module is.** A service that owns one domain completely, including its
own database, reached only through the backend. The LLM engine owns models,
conversations, runs and scores; the backend owns users and presets and models
none of what the LLM engine owns. It stores an id and asks.

**What the seam looks like**, concretely, in
[`backend/crates/llm-engine/`](backend/crates/llm-engine/):

- **`config`** — where the process lives, what port it wants, how to start it.
- **`supervisor`** — spawns the process, keeps it alive, health-checks it before
  the backend reports ready, and shuts it down with the app.
- **`proxy`** — forwards requests, signing each one with a short-lived service
  token that names the acting user. The token's shape is pinned for both
  languages by [`contracts/service-token.json`](contracts/service-token.json),
  so the Rust side and the Python side cannot drift apart silently.

`contracts/` is where anything two languages both implement gets pinned, and the
token is not the only one. [`contracts/shell-observation.json`](contracts/shell-observation.json)
pins what a model reads back from a shell: the backend renders it live, and the
data engine renders it again in Python when it converts a recorded agent corpus
into training data. That one drifts worse than the token does — a token
mismatch is a 401, where a layout mismatch is a fine-tuned model quietly reading
a block it was never trained on.

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
| **LLM engine** | Models, conversations, training runs, benchmark scores, the weights | **Built** — `llm-engine/`, behind `backend/crates/llm-engine/` |
| **Data engine** | Datasets from the Hub, the documents in them, and the cleaning that produces new ones | **Built** — `data-engine/`, behind `backend/crates/data-engine/` |
| **Gyms engine** | Generated tasks, sandboxed episodes, trajectories and scores | **Built** — `gyms-engine/`, behind `backend/crates/gyms-engine/` |
| **Benchmark** | Suites, scoring, held-out sets, comparison across models | **Planned** |

Benchmarking is not a stub — there is no separate code for it yet; the LLM
engine currently serves its routes alongside the `/benchmark` views. Pulling it
out is the near-term direction of the work, and the seam above is what it will
be pulled out *into*.

The data engine is what training reads: a run selects datasets by id and the
LLM engine fetches their documents a page at a time through the backend, keeping
no copy. The scripts in [`scripts/`](scripts/) that prepared folders of `.txt` are
superseded by pulling from the Hub in the Data view — see
[`docs/decisions/datasets-out-of-the-llm-engine.md`](docs/decisions/datasets-out-of-the-llm-engine.md).

When you build one, the LLM engine is the worked example: read
[`backend/crates/llm-engine/`](backend/crates/llm-engine/) end to end first. It is small
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
