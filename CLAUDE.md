# CLAUDE.md

An LLM trained, served and inspected end to end: a PyTorch engine, a Rust API
in front of it, a Next.js UI in front of that.

```
engine/     training, inference, evaluation — PyTorch. Owns the models and the data.
backend/    HTTP, auth, and the seam to the engine — Rust, Actix, sqlx.
frontend/   chat, the job board, model and benchmark views — Next.js.
```

Conventions live beside the code they govern: [`backend/CLAUDE.md`](backend/CLAUDE.md),
[`frontend/CLAUDE.md`](frontend/CLAUDE.md), and `engine/llm/__init__.py` for the
engine's layering.

## Where knowledge lives

- **Decision records** — [`docs/decisions/`](docs/decisions/). Findings that cost
  real time and would otherwise be re-derived or re-broken.
- **Source comments** — only the non-obvious *why*, only where it is short.

## Tests

```sh
cd engine   && .venv/bin/python -m pytest tests/    # the engine, incl. the dataflow trace
cd backend  && cargo test --workspace               # needs DATABASE_URL
cd frontend && bun run typecheck && bun run lint
```

`engine/tests/test_dataflow_trace.py` is the contract that keeps the engine
honest: one fixed question, recorded at five layer boundaries under a pinned
seed. Re-record with `python -m tests.fixtures.record` only when a behaviour
change is intended, and commit the diff as the evidence of what moved.
