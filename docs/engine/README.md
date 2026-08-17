# Engine — PyTorch training, inference, evaluation

> **Status: skeleton.** Sections below are the required shape (see
> [`../README.md`](../README.md)); the content is not written yet. Fill a
> section the first time you work in the area it covers — do not leave a
> heading empty behind you.

## What it owns

Models, conversations, runs and scores. The backend owns users and presets and
nothing else; requests flow frontend → backend → engine and back the same way.
See [`../decisions/`](../decisions/) for the two-database split.

<!-- TODO: the ownership boundary in detail — what the engine will never ask
     the backend for, and vice versa. -->

## Tech stack

<!-- TODO: PyTorch, FastAPI, SQLAlchemy/alembic, psycopg — and the why where a
     choice was contested. -->

## Layout

<!-- TODO: package layout and the layering currently described in
     `engine/llm/__init__.py`. That docstring is the source; move it here and
     leave a pointer, or state here that it is authoritative. -->

## Schema

<!-- TODO: tables, who writes each, invariants not expressed in the types.
     Database: ENGINE_DATABASE_URL, default postgresql+psycopg://localhost/slate_engine,
     migrated with `alembic upgrade head`. -->

## Routes

<!-- TODO: the HTTP surface the backend proxies to, and the service-secret
     hop (ENGINE_SERVICE_SECRET on both sides; unset leaves the engine open). -->

## Conventions

<!-- TODO: patterns code in this module must follow. -->

## Gotchas

- `tests/test_dataflow_trace.py` is the contract that keeps the engine honest:
  one fixed question, recorded at five layer boundaries under a pinned seed.
  Re-record with `python -m tests.fixtures.record` only when a behaviour change
  is intended, and commit the diff as the evidence of what moved.

<!-- TODO: the rest. -->
