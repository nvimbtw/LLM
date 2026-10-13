# AGENTS.md

The entry point for every agent working in this repository. Read it, then read
the doc for the module you are about to touch. Nothing else is required
reading — this file exists so you never have to guess where something lives.

An LLM trained, served and inspected end to end: a PyTorch LLM engine, a Rust API
in front of it, a Next.js UI in front of that.

```
llm-engine/   training, inference, evaluation — PyTorch, FastAPI.
data-engine/  acquiring datasets and cleaning them — FastAPI, huggingface_hub.
bench-engine/ English benchmark scoring and history — FastAPI.
gyms-engine/  tasks, a sandboxed shell, and what a model scored — FastAPI.
backend/      people, sessions, and the seam to the services — Rust, Actix, sqlx.
frontend/     chat, the job board, model, data, benchmark and gym views — Next.js.
```

Each module owns a database. The LLM engine owns models, training and execution scheduling; bench-engine
owns benchmark history and scoring; the data engine owns datasets and the documents in them — including the
text every model is trained on; the gyms engine owns gym runs and the
trajectories in them; the backend owns users, presets, conversations, messages and attachments. A request flows
frontend → backend → a service and back the same way.

**The Python services never speak to each other.** Everything either knows
about the other arrives through the Rust backend. That is why the LLM engine, which
trains on data it does not own, reads it *through* the backend. Bench-engine likewise requests model measurements
through Rust. See
[`docs/decisions/datasets-out-of-the-llm-engine.md`](docs/decisions/datasets-out-of-the-llm-engine.md).

## Where everything is

| | |
| --- | --- |
| [`docs/llm-engine/`](docs/llm-engine/) · [`docs/bench-engine/`](docs/bench-engine/) · [`docs/data-engine/`](docs/data-engine/) · [`docs/gyms-engine/`](docs/gyms-engine/) · [`docs/backend/`](docs/backend/) · [`docs/frontend/`](docs/frontend/) | What each module is: ownership, stack, layout, schema, routes, conventions, gotchas. |
| [`docs/conventions.md`](docs/conventions.md) | The coding rules (R1–R21) every change is judged against. |
| [`docs/decisions/`](docs/decisions/) | Decision records — what was weighed and rejected, dated, once. |
| [`docs/README.md`](docs/README.md) | The spec for the docs themselves: shape of a module directory, what belongs in one. |

# How to work

1. **Read the module doc first.** Before editing `backend/`, read
   [`docs/backend/README.md`](docs/backend/README.md); same for the others; all
   of them if the change spans modules. Not skim — read. Each doc is kept short
   enough that this is cheap, and the things it tells you (who owns which
   table, which crate may depend on which) are exactly what you cannot infer
   from the file you happen to have open.
2. **Do the thing asked, at the size asked.** Fix the described problem. Note
   unrelated problems in your summary instead of fixing them; a change that
   quietly grew is a change nobody can review.
3. **Follow the conventions.** [`docs/conventions.md`](docs/conventions.md)
   plus the module's own conventions section. Load the full text of a rule when
   you need it — don't work from the index alone.
4. **Leave the docs true.** See below. This is part of finishing, not a
   follow-up.
5. **Verify before you claim.** `cargo build` clean with zero warnings (R7) and
   the relevant tests, `bun run lint`/`typecheck`, `pytest` for the LLM engine. If
   something fails and you cannot fix it, say so plainly with the output — do
   not report a change as done on a red build.
6. **Committing is the user's call.** Do not commit or push unless asked.

# How to document

**Every module is documented in [`docs/`](docs/), and the documentation is part
of the code change — not a chore that follows it.**

1. **A change is not done until the docs are true.** If you altered a schema, a
   route, a module's public surface, its layering, or what it owns, update
   `docs/<module>/` **in the same change**. Not later, not in a filed task.
2. **Fix lies where you find them.** A doc that contradicts the code gets
   corrected by whoever noticed, in the change they are already making. A doc
   nobody trusts is worse than no doc: it gets read once, found wrong, and
   skipped forever after.
3. **A new module means a new `docs/<module>/README.md`**, written in the change
   that creates the module.
4. **Markdown only.** No HTML source, no per-file ceremony. Docs are read far
   more often by agents than by people, and markdown greps cleanly, diffs
   reviewably, and costs the fewest tokens to load. The frontend renders the
   tree at **`/docs`** — you never touch the frontend to add a doc, but note
   that a link to a file that doesn't exist **fails the frontend build**. That
   is intentional.
5. **`README.md` is the entry point and the complete map.** Everything about
   the module goes in it, until it passes ~250 lines — then split the biggest
   topic into a sibling file (`schema.md`, `api.md`, …) and leave a one-line
   summary and a link behind. The README never stops being the map.
6. **Write the boundary, the invariant, and the why.** Never restate what a
   file listing or a type signature already says. Prose that duplicates code is
   prose that will contradict it within a month.
7. **Long-form rationale goes to [`docs/decisions/`](docs/decisions/)**, not
   into the module doc. A module doc says what *is*; a decision record says
   what was weighed and rejected, with a date.

The required shape of a module directory, and the section-by-section contents
of a README, are specified in [`docs/README.md`](docs/README.md). Follow it.

Why this is a rule and not a suggestion: without it, the only complete picture
of this system is spread across three codebases and nobody's head, and every
agent that arrives re-derives it badly and gets it slightly wrong. The docs are
the shared map. Keeping them true is cheaper than everyone guessing.

# Running it

```sh
cp .env.example .env
for k in JWT_SECRET LLM_ENGINE_SERVICE_SECRET DATA_ENGINE_SERVICE_SECRET \
         GYMS_ENGINE_SERVICE_SECRET BENCH_ENGINE_SERVICE_SECRET BACKEND_SERVICE_SECRET SEARXNG_SECRET; do
    printf '%s=%s\n' "$k" "$(openssl rand -hex 32)"
done >> .env
docker compose up    # everything, hot-reloading
```

`docker compose restart <service>` restarts one without disturbing the rest.
The images build with `SQLX_OFFLINE=true` against the checked-in `.sqlx/`, so
**a change to a query or a migration must be followed by
`cargo sqlx prepare --workspace` in the same change** — otherwise the container
build fails while `cargo run` on the host still succeeds. Why several containers
rather than one:
[`docs/decisions/one-command-development.md`](docs/decisions/one-command-development.md).

Natively, which is still the better loop for LLM engine work:

```sh
cd llm-engine      && .venv/bin/uvicorn main:app --port 8001   # or let the backend spawn it
cd data-engine && .venv/bin/uvicorn main:app --port 8002   # or let the backend spawn it
cd gyms-engine        && .venv/bin/uvicorn main:app --port 8003   # or let the backend spawn it
cd bench-engine && .venv/bin/uvicorn main:app --port 8004
cd backend     && cargo run                                 # needs DATABASE_URL, JWT_SECRET
cd frontend    && bun run dev
```

The backend spawns the Python services as children unless `LLM_ENGINE_AUTOSTART` /
`DATA_ENGINE_AUTOSTART` / `GYMS_ENGINE_AUTOSTART` / `BENCH_ENGINE_AUTOSTART` is false, and health-checks each before reporting ready.
Set each service’s `*_SERVICE_SECRET` on both sides to
authenticate the hops; unset leaves a service open and it says so at startup. A
token is addressed to one service by name and is refused by the other.
`BACKEND_SERVICE_SECRET` guards reverse calls from the LLM and bench engines,
scoped by issuer, and behaves the
same way when unset.

Each Python service needs its own database, migrated with `alembic upgrade head`:
`LLM_ENGINE_DATABASE_URL` (default `postgresql+psycopg://localhost/slate_engine`),
`DATA_ENGINE_DATABASE_URL` (default `postgresql+psycopg://localhost/slate_data`)
`GYMS_ENGINE_DATABASE_URL` (default `postgresql+psycopg://localhost/slate_gyms`),
and `BENCH_ENGINE_DATABASE_URL` (default `postgresql+psycopg://localhost/slate_bench`).

# Tests

```sh
cd llm-engine      && .venv/bin/python -m pytest tests/   # incl. the dataflow trace
cd data-engine && .venv/bin/python -m pytest tests/   # incl. the migration drift check
cd gyms-engine        && .venv/bin/python -m pytest tests/   # no Docker, no Postgres
cd bench-engine && .venv/bin/python -m pytest tests/
cd backend     && cargo test --workspace              # needs DATABASE_URL
cd frontend    && bun run typecheck && bun run lint
```

`llm-engine/tests/test_dataflow_trace.py` is the contract that keeps the LLM engine
honest: one fixed question, recorded at five layer boundaries under a pinned
seed. Re-record with `python -m tests.fixtures.record` only when a behaviour
change is intended, and commit the diff as the evidence of what moved.
