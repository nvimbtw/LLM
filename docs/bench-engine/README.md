# Bench Engine — `language-v3` and beyond

Owns suites, their held-out data, and every score. The engine owns weights; this owns what the weights are measured against.

## What it owns

`slate_bench` (`BENCH_ENGINE_DATABASE_URL`, default `postgresql+psycopg://localhost/slate_bench`):

- `benchmarks` — one row per model×suite, like `engine.benchmarks` but moved here. `suite_version` gates comparability.
- `runs` — its own queue (one device, one line, same `Runner`/`Registry` shape as the other Python services). Benchmark runs do not contend with training.
- `queue_state` — durable pause.

Held-out text lives under `eval_data/language/corpus/*.txt` and `eval_data/language/grammar/minimal_pairs` — not in `engine/data/eval` any more, but that tree is kept read-only for v1/v2.

What it does **not** own: weights, vocabularies, training corpora. It reads them.

## Stack

FastAPI, SQLAlchemy 2.0 sync, `psycopg`, `torch`, `tokenizers`, `alembic`. Same lifespan/supervisor pattern as `engine` and `data-engine`.

## Routes

All behind `BENCH_ENGINE_SERVICE_SECRET` (`AUDIENCE bench-engine`).

| | |
|---|---|
| `POST /eval` | Queue `language-v3` against `model_name` |
| `GET /benchmarks?suite=` | History, newest first |
| `GET /benchmarks/suite` | Suite info |
| `GET /benchmarks/{id}` | One report |
| `DELETE /benchmarks/{id}` | |
| `GET /runs`, `GET /runs/{id}`, `GET /runs/{id}/events` (SSE), `POST /runs/{id}/cancel`, `PATCH /runs/{id}`, `POST /runs/pause` | Queue |

Backend proxies `GET/DELETE /benchmarks*` and `POST /runs/benchmark` here; all other `/runs/*` stay on the engine.

## `language-v3`

32 prompts: 24 continuations (12 original + 12 narrative/expository) + 8 expository direct. `SamplingConfig(max_tokens=128, temp 0.8, top_k 40, top_p 0.95, seed 1234+i)`. Unconstrained generation (no markup mask).

- **Corpus** — `bits_per_byte` vs `ByteNgram(5)` baseline, strided windows.
- **Surface** — `real_word_rate` vs held-out lexicon, `distinct_{1,2,3}`, `repeat_3`, `self_bleu_2` (cross-sample 2-gram overlap), `per_sample_distinct_3`.
- **Grammar** — 30 minimal pairs, forced-choice logprob accuracy.
- **Coherence** — `gen_bits_per_byte` over concatenated samples (second pass).

Score v3: `W_L 0.35 + W_G 0.25 + W_W 0.15 + W_D 0.15 + W_C 0.10`, redistributed over finite, `* variety_gate(0.30–0.70 on distinct_3)`, 0–100.

Report under `model/surface/grammar/coherence/samples/score` and `tokens_used`.

## Adding a suite

Add `suites/<name>/runner.py` exporting `run_<name>(model, tokenizer, device, on_progress, cancel)`, a `score.py` with `SCORE_VERSION`, extend `store/schema.Suite`, add handler case, add migration if needed, expose in `GET /suite`.

## Generation backend

`api/handlers._load_model` tries `engine/data/artifacts/models/<id>/model.pt` via shared `DATA_DIR`/`/app/engine-data` mount, else mock for tests. To call the engine over HTTP instead, replace `_load_model` with a client to `ENGINE_URL`.

## Gotchas

- `BENCH_ENGINE_DATABASE_URL` must be `slate_bench`, not `slate_engine` — old benchmarks stay read-only in the engine.
- `eval_data` is mounted read-only into the container; edits need rebuild.
- The queue is separate: pausing training does not pause benchmarking.
