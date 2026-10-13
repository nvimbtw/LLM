# Bench engine — English language measurements

## Ownership

Bench-engine owns benchmark records, reference English prose, scoring policy and
reports. Only the `language` suite can be newly executed. Migrated `markup` and
`assistant` reports remain readable with their original suite/version labels.
The earlier `language_v3` service’s history has its own legacy tab and is not
compared with the current language suite.
Model weights, tokenizers, training loss and model computation belong to the LLM
engine. Every service request goes through Rust; Python services never call one
another.

## Stack and layout

FastAPI, SQLAlchemy, Alembic and PostgreSQL. This service has no PyTorch dependency
and never mounts model weights. `bench/` owns the corpus split, byte baseline,
fixed prompts, surface measurements and composite. `store.py` owns report rows;
`main.py` dispatches one measurement worker and exposes history. The reverse hop
in `bench/backend.py` addresses Rust as issuer `bench-engine`.

## Schema

`benchmarks` preserves UUID, logical model UUID and name snapshot, suite/version,
measurement status, nullable measurements, full report, errors and timestamps.
Model rename/deletion does not relabel or cascade-delete history. New measurement
IDs equal the scheduling job UUID; migrated report IDs stay unchanged.
`config` records the resolved measurement request. Measurement status describes
the report; authoritative queue status, ordering and dependencies live exclusively
in the LLM engine's `execution_jobs`.

## API

- `GET /health`: supervisor readiness.
- `POST /execute/{id}`: idempotently start English measurement; repeated IDs
  return the existing record. A retry is a new scheduling job/measurement ID.
- `POST /execute/{id}/cancel`: signal the worker and the model measurement through
  Rust. Likelihood work notices between windows, generation between prompts.
- `GET /benchmarks`: history, optional `suite`, `model_id`, bounded `limit`.
- `GET /benchmarks/suite`: active suite and installed reference-file count.
- `GET /benchmarks/{id}`: report; `DELETE` removes terminal history only.

All routes require a backend service token addressed to `bench-engine` when
`BENCH_ENGINE_SERVICE_SECRET` is set. Unset is announced development mode.
Rust's internal measurement routes accept the bench issuer only; dataset/gym
and dispatch routes accept the LLM issuer only.

## Running and conventions

Database: `BENCH_ENGINE_DATABASE_URL`, default
`postgresql+psycopg://localhost/slate_bench`; migrate with `alembic upgrade head`.
Install `requirements.txt`, then `uvicorn main:app --port 8004`. Backend supervision
uses `BENCH_ENGINE_*`, matching the other service adapters. Compose supplies its
own container and database. Existing Postgres volumes need `slate_bench` created;
initialization SQL runs only on an empty volume.

Place reference `.txt` files in `bench-engine/data/corpus`, or set
`BENCH_CORPUS_DIR`. The 90/10 split, prompt set, sampler seeds and version-2 scoring
are preserved. An unavailable measurement remains null; it is never silently
reported as zero. Reference prose must remain outside training datasets.

## Device and migration gotchas

The shared queue waits for this worker, while each model measurement acquires the
LLM device gate. The dispatch call itself holds no gate: holding it would deadlock
the returning measurement request. Interactive inference receives busy while a
measurement holds the device. Interrupted workers become failed at restart;
completed history remains readable without a live model.

See [ownership cutover](../backend/ownership-cutover.md) before upgrading populated
databases. The transfer compares every report before archiving source tables;
no application reads the archive. Never start both old and new writers during
cutover.

Existing databases at `003_suite_to_varchar` upgrade through the restored original
migration history and `004_sprint_ownership`. Reports are altered in place without
changing IDs, scores, versions or JSON values. Terminal legacy scheduling rows
are retained in `migrated_execution_jobs` and copied to the LLM queue by
`scripts/migrate_legacy_bench_jobs.py`; unfinished legacy work blocks the upgrade.
