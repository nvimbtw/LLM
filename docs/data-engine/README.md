# Data engine — datasets, before they are training data

The second Python service. It acquires datasets from the Hugging Face Hub,
stores them, and cleans them; the LLM engine trains on data and knows nothing
about where it came from.

## What it owns

Datasets, the documents in them, and the runs that write them — in its own
database, `slate_data`. That now includes **the text every model is trained
on**: the LLM engine holds no corpora of its own and reads these by page through the
backend, so a dataset here is not a staging area for one over there.

What it does **not** own, and must never reach for:

- **The LLM engine's artifacts.** `llm-engine/data/` holds packed corpora, vocabularies
  and weights — all of them derived. Nothing here reads or writes that tree, and
  nothing here knows a dataset was ever trained on.
- **People.** `owner_id` is an opaque id minted by the backend and never
  validated, the same arrangement the LLM engine uses. There is no foreign key,
  because the row lives in another database.
- **Anything about training.** No torch, no tokenizers, no notion of a model.

**The Python services never speak to each other.** Everything either knows
about the other arrives through the Rust backend. That is what lets each one be
restarted, moved or replaced without the other noticing, and it is why this
module has no client for the LLM engine and the LLM engine has none for it — the LLM engine
reads documents from here without either side holding an address for the other.

## Tech stack

FastAPI, SQLAlchemy 2.0 ORM over psycopg 3, alembic — the LLM engine's stack minus
everything that computes. `huggingface_hub` and `pyarrow` do the fetching;
`datasets` is deliberately *not* a dependency, because parquet shards read
through `hf_hub_download` + pyarrow are lazy and a budget can stop them
mid-repo.

Sessions are sync, not async, for the reason the LLM engine's are: a run's worker
thread writes its own progress, and reaching an async session from a thread
means bouncing every write through the event loop for no gain.

## Layout

```
main.py          the app, its lifespan, the routers it mounts
api/
  auth.py        the service-token hop — AUDIENCE = "data-engine"
  runs.py        live run state: cancel flag, frame history, subscribers
  runner.py      the queue: claims a row, runs a handler, records the outcome
  handlers.py    what the queue runs — pull and process
  sse.py         frame framing, as the backend's proxy parses it
  routes/        one router per resource; full paths, no prefixes
hub/             the Hub: `search` for what exists, `sample` for a look inside,
                 `fetch` for the rows
processors/      the curated cleaning steps, and the pipeline that runs them
store/           schema, queries, and the session factory
migrations/      alembic; the URL comes from `store.url()`, never alembic.ini
```

Dependencies run one way: `routes → handlers → {hub, processors, store}`.
Nothing in `store/` or `processors/` imports from `api/`.

## Schema

Three tables in `slate_data`, `public`. Its **own database**, not a schema
inside the LLM engine's — one database per module is the boundary this repo already
draws, and it is what lets the tests run against in-memory SQLite instead of
demanding a server.

| Table | Holds |
| --- | --- |
| `datasets` | One named body of documents: pulled from the Hub, derived by cleaning one that was, or written by the backend (the search cache, a gym's demonstrations). |
| `documents` | The content itself — `content` for prose, `messages` for conversations. |
| `runs` | A pull or a processing pass, with its config, progress and outcome. |

Invariants the types do not express:

- **`documents` carries content XOR messages**, enforced by a CHECK. A row with
  both, or neither, is a bug in whatever wrote it.
- **`ordinal` orders a dataset, not `created_at`.** It is what makes a preview
  honest: "the first twenty documents" must mean the same twenty every time a
  parameter is nudged, or the before/after being compared is two samples.
- **`datasets.parent_id` is `SET NULL`, not cascade.** Deleting a source must
  not silently take its cleaned descendants with it.
- **`datasets.steps` is the provenance record** — the pipeline that produced a
  derived dataset, with resolved parameters and submitted/executed source hashes.
  Queued jobs use current code and record any drift; historical source snapshots
  are not retained. See [processor provenance](processors.md#version-records).
- **The counters on `datasets` are recomputed at the end of a run**, never
  accumulated by the writer, for a dataset a `Run` fills — a cancelled pull
  must still report what it kept. The exceptions are the datasets the backend
  writes directly (below), whose rows land outside the `Run` path entirely; they
  bump their counters incrementally instead, under a lock on the dataset row.
- **`documents.created_at` is real on every row**, not only a cached
  search's. Nothing here carried a per-document timestamp before this table
  needed one to answer "how old is this."
- **`documents.cache_key`** is null on anything pulled from the Hub or
  produced by a pipeline — those are found by dataset and ordinal, never by
  content. Non-null only on a row the search cache wrote.

### Datasets the backend writes

Two datasets here are not pulled from the Hub and not derived from anything
stored: the **search cache** (`origin = search`), where a `web-search` tool call
lands a result instead of re-fetching it, and **gym demonstrations**
(`origin = gym`), the expert trajectories a gym generates to be fine-tuned on.
Both are written a row at a time by the backend, outside the `Run` path, `ready`
the moment anything is in them. Both are therefore trainable with no work here.
See [`backend-written.md`](backend-written.md).

### Content in Postgres

Documents are rows, not files. The LLM engine's `data/` tree grew large enough to
slow every tool that walks the repo, and a corpus that is only ever read by
page, sampled by ordinal and filtered by predicate is a table's job anyway.

That choice has a price, and it is paid deliberately: **every pull is capped**,
inserts are batched, reads are paged, and the export streams. Nothing in
`store/queries.py` returns a whole dataset. See
[`../decisions/dataset-content-in-postgres.md`](../decisions/dataset-content-in-postgres.md).

## Routes

Everything is behind the app-level service-token dependency, so a new endpoint
is authenticated because it exists. `GET /` reports liveness; the authenticated
`/health/socket` answers backend heartbeats, as described in
[service availability](../backend/services.md#availability-monitoring).

| | |
| --- | --- |
| `GET /hub/datasets?q=` | Search the Hub. |
| `GET /hub/datasets/detail?repo=` | Configs, columns and shards of one repo. |
| `GET /hub/datasets/sample?repo=&config=&split=&rows=` | The first few rows, with the kind and column they suggest. |
| `GET /datasets` · `GET /datasets/{id}` | What is stored. |
| `GET /datasets/{id}/documents?offset=&limit=` | A page of content. |
| `GET /datasets/{id}/export` | The whole dataset as JSON lines, streamed. |
| `POST /datasets/pull` | Create the row, queue the run that fills it. |
| `POST /datasets/{id}/process` | Queue a pipeline into a **new** dataset. |
| `DELETE /datasets/{id}` | |
| `GET /processors?kind=` | Active catalogue, parameters, source file and SHA256. |
| `POST /processors` | Create a loadable first custom source version through Rust's admin gate. |
| `GET` · `PUT /processors/{name}/source` | Saved source and active identity; custom-file saves through Rust's admin gate. See [processors](processors.md). |
| `POST /preview` | A pipeline's before/after over the first *n* documents. |
| `GET /runs` · `GET /runs/{id}` · `POST /runs/{id}/cancel` | The queue. |
| `GET /runs/{id}/events` | SSE, replayed from the start. |
| `POST /search-cache/lookup` | The newest result under a key, or an empty hit. See [`backend-written.md`](backend-written.md). |
| `POST /search-cache` | Record a result under a key, creating the cache dataset on first use. |
| `POST /datasets/demonstrations` | Append up to 5,000 gym demonstrations to a named `sft` set, creating it on first use. See [`backend-written.md`](backend-written.md). |

**`/datasets/{id}/documents` is the seam the LLM engine reads**, through the
backend's `/internal` routes — it stopped scanning a directory and now trains on
what is stored here. It pages rather than streaming for two reasons: the
backend buffers a forwarded response, so an export would have to become a
byte-stream passthrough the proxy does not have; and a connection dropped forty
minutes into a pack would cost the whole pack, where a page costs one retry.

`/datasets/{id}/export` is therefore still consumed by nobody. It stays because
it is the honest way to hand a whole dataset to something outside this system,
and it is built and tested.

**`MAX_PAGE` is now load-bearing for a second caller.** A browser asking for a
million rows is a mistake; a packing run asking for more than 200 is not, but it
takes the ceiling and pages around it. Raising it would let a run pull larger
pages — measure before assuming that is faster, since over loopback the round
trips are not what a pack spends its time on.

## Conventions

- **Processors are curated code or trusted operator files.** The
  [authoring and reload contract](processors.md) covers the decorator, custom
  import policy, admin source edits and submitted/executed hashes. Python
  executes in process; this interface requires operator trust.
- **A processor returns the document or `None` to drop it.** Cross-document
  state (deduplication) lives in the per-run `State` dict it is handed, never
  in a module global — two runs must not contaminate each other, and a preview
  must always start clean.
- **Block structure is `markup.blocks`; inline formatting is
  `markdown-emphasis`.** A source writes Markdown and the model must emit the
  reserved markers, so the conversion is split by what each side can see:
  `blocks` has the line structure and renders headings, lists, quotes, rules,
  paragraphs and fences while a turn is being converted; `markdown-emphasis`
  runs afterwards over the runs of text and does emphasis and code spans. What
  is converted is fixed by `llm-engine/llm/corpus/special.py`, not by Markdown —
  `#`/`##`/`###+` are `<|title|>`/`<|heading|>`/`<|subheading|>`, and a
  construct with no marker (a table) stays prose rather than inventing one. The
  result must also satisfy the grammar in `llm-engine/llm/chat/markup.py`: an
  `<|item|>` only inside a `<|list|>`, and only items directly inside one.
- **Out-of-range parameters are clamped, not rejected.** They arrive from a
  field the user is still typing in, and a preview that errors mid-keystroke is
  worse than one showing the nearest legal answer.
- **Terminal SSE frames are `done`, `error`, `cancelled`.** The backend's
  `Frame::is_final` switches on those three names; they are a wire contract.
- One query function per question, `db: Session` first, owner scoping an
  explicit argument so omitting it is a type error rather than a silent
  full-table read.

## Gotchas

- **`Json` sets `none_as_null=True`.** Without it SQLAlchemy stores a Python
  `None` as the JSON value `null`, which is not SQL `NULL`, so `messages IS
  NULL` never holds and the content/messages CHECK rejects **every** row. The
  LLM engine's `Json` alias does not set it and has never noticed, because it has
  no such constraint.
- **The preview and the pull read different sources.** `hub/sample.py` asks the
  Hub's **datasets-server** for rows it has already extracted — one small HTTP
  call, no token, public datasets only — while `hub/fetch.py` reads the parquet
  shards. So a repo with no preview (gated, private, unindexed) still pulls
  fine, and the kind/column the preview suggests is a default for the form, not
  a decision: the pull resolves its own column against a real row.
- **Most repos ship no parquet, and are read off `refs/convert/parquet`.** A
  dataset published as CSV, JSON or a loading script has nothing to read on
  `main`; the Hub converts it and keeps the result on that branch.
  `search.parquet_shards` prefers the repo's own shards and falls back to the
  conversion, and it is what both the detail route and the pull list with — so
  the prefixes offered in the form exist on the revision the pull will download
  from. A shard path and its revision therefore travel together, as
  `fetch.Shards`.
- **A pull that cannot resolve its content column fails loudly.** Silence looks
  exactly like a dataset that is genuinely empty. Exhausting the rows without
  ever finding one is the same failure as running out of patience, and both
  raise.
- **`process` reads its source a page at a time, not as a stream.** The writer
  commits while the reader iterates; holding a read transaction open across
  those writes deadlocks on a single connection, which is what the tests run on.
- Roles arrive from the Hub as the source wrote them (`human`, `gpt`). Mapping
  them is the `normalise-roles` processor's job — doing it during the fetch
  would hide from the user that it happened.
  [Project and memory training context](project-memory.md) stays structured JSON.
- **`convert-hermes` renders a tool result the way the live shell does, and that
  mirror is pinned rather than promised.** `processors/hermes.py::_result_body`
  reimplements `backend/crates/tools/src/bash.rs::render` because this service
  cannot call into Rust, and a conversation stored here carries the finished
  block — the LLM engine passes it through untouched, since there is nothing
  left to render. The failure mode has no symptom on this side: the corpus looks
  fine, the fine-tune converges, and the model then misreads its own tool results
  at serving. `contracts/shell-observation.json` is generated from the Rust side
  and read by both suites, so a change to either renderer fails a test. Regenerate
  it after a deliberate change; editing it to make a test pass defeats the point.
- **NUL bytes are stripped during the fetch, not by a processor.** This is the
  one cleanup `hub/fetch.py` does, and it is the exception to the rule above.
  PostgreSQL cannot represent `0x00` in `text` or in a `jsonb` string at all, so
  a single stray one — Gutenberg dumps are full of them — fails the insert for
  its whole `WRITE_BATCH`, leaving a half-written dataset marked `failed`.
  `normalise-unicode` would remove them, but a processor only ever runs over a
  dataset that already landed, and this is precisely a corpus that cannot land.
  Content, turns and metadata are all scrubbed, before `byte_size` is measured
  so the recorded size is of what was actually stored, and the pull's result
  carries `nul_stripped` — the count of documents it rewrote.

## Running it

```sh
createdb slate_data
cd data-engine && .venv/bin/alembic upgrade head
cd data-engine && .venv/bin/uvicorn main:app --port 8002    # or let the backend spawn it
cd data-engine && .venv/bin/python -m pytest tests/
```

`DATA_ENGINE_DATABASE_URL` defaults to
`postgresql+psycopg://localhost/slate_data`. `DATA_ENGINE_SERVICE_SECRET`
unset leaves the service open and says so at startup — the development default,
not a fallback.
