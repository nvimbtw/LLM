# CLAUDE.md

An LLM trained, served and inspected end to end: a PyTorch engine, a Rust API
in front of it, a Next.js UI in front of that.

```
engine/     training, inference, evaluation, and the data they own — PyTorch, FastAPI.
backend/    people, sessions, and the seam to the engine — Rust, Actix, sqlx.
frontend/   chat, the job board, model and benchmark views — Next.js.
```

Each module owns a database. The engine owns models, conversations, runs and
scores; the backend owns users and presets; a request flows frontend → backend →
engine and back the same way. See
[`docs/decisions/0008`](docs/decisions/0008-two-databases-one-opaque-owner.md).

Conventions live beside the code they govern: [`backend/CLAUDE.md`](backend/CLAUDE.md),
[`frontend/CLAUDE.md`](frontend/CLAUDE.md), and `engine/llm/__init__.py` for the
engine's layering.

## Where knowledge lives

- **Decision records** — [`docs/decisions/`](docs/decisions/). Findings that cost
  real time and would otherwise be re-derived or re-broken.
- **Source comments** — only the non-obvious *why*, only where it is short.

## Running it

```sh
cd engine  && .venv/bin/uvicorn main:app --port 8001    # or let the backend spawn it
cd backend && cargo run                                  # needs DATABASE_URL, JWT_SECRET
cd frontend && bun run dev
```

The backend spawns the engine as a child unless `ENGINE_AUTOSTART=false`, and
health-checks it before reporting ready. Set `ENGINE_SERVICE_SECRET` on both
sides to authenticate the hop; unset leaves the engine open and it says so at
startup.

The engine needs its own database (`ENGINE_DATABASE_URL`, default
`postgresql+psycopg://localhost/slate_engine`) migrated with `alembic upgrade
head`.

## Tests

```sh
cd engine   && .venv/bin/python -m pytest tests/    # incl. the dataflow trace
cd backend  && cargo test --workspace               # needs DATABASE_URL
cd frontend && bun run typecheck && bun run lint
```

`engine/tests/test_dataflow_trace.py` is the contract that keeps the engine
honest: one fixed question, recorded at five layer boundaries under a pinned
seed. Re-record with `python -m tests.fixtures.record` only when a behaviour
change is intended, and commit the diff as the evidence of what moved.
