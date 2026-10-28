# LLM engine — PyTorch training and model execution

> **Status: partial.** Sections below are the required shape (see
> [`../README.md`](../README.md)); the ones still marked TODO are not written
> yet. Fill a section the first time you work in the area it covers — do not
> leave a heading empty behind you.

## What it owns

Models, tokenizers, training records, durable execution scheduling and stateless
model computation. Rust owns conversations/files/usage; bench-engine owns
benchmark scoring and history. All service communication passes through Rust.
See [execution.md](execution.md) and the [bench-engine map](../bench-engine/README.md).

The modular architecture [capture](architecture-capture.md) and
[supervised runtime](modular-runtime-feasibility.md) records describe the dated
ISS-48 CPU feasibility prototypes and their remaining production/GPU gates.
The version-one [architecture contracts](architecture-contracts.md) map shared
SDK/YAML/runtime/API schemas and fixtures consumed by the production catalog,
validation, training and serving paths.
[Catalog persistence](architecture-storage.md) maps immutable revisions, evidence
retention and transaction APIs. The [delivery ledger](modular-architecture-evidence.md)
tracks implementation evidence and deferred hardware checks.
The [acceptance map](modular-architecture-acceptance.md) links formal requirements
to tests, the grouped verification command and remaining connected scenarios.
[Saved-definition SFT](modular-sft.md) maps private data/tokenizer authority,
capability warm-start, reference diagnostics and exact batch-cursor recovery.
[Saved-definition serving](modular-serving.md) maps reusable worker ownership,
chat/scoring, cancellation and immutable definition/graph reads.
[Builtin blocks and recipes](architecture-builtins.md) documents the implemented
standard, independent-stack, shared-loop and custom examples. The production
[worker handoff](supervised-runtime.md) maps admission, process cleanup and consumer
responsibilities.

What it does **not** own, and this is the boundary that moved most recently:
**the text it trains on**. Corpora are the data engine's, and a run reads them a
page at a time through the backend — see
[`../decisions/datasets-out-of-the-llm-engine.md`](../decisions/datasets-out-of-the-llm-engine.md).
What is left under `llm-engine/data/` is derived (packed corpora, vocabularies,
weights) plus `eval/`, which is held out from training and is therefore not
training data. A run keeps no copy of what it read.

It owns the **vocabulary** as well, which is the ownership boundary that catches
people out: the marker table, the chat template, and the grammar the model
decodes under are all here, and the other two modules read them rather than
restating them. A tool's *name* is decided here (it is a marker stem); a tool's
*behaviour* is the backend's.

<!-- TODO: the rest of the ownership boundary — what the LLM engine will never ask
     the backend for, beyond datasets, and vice versa. -->

## Tech stack

<!-- TODO: PyTorch, FastAPI, SQLAlchemy/alembic, psycopg — and the why where a
     choice was contested. -->

## Layout

The layering is stated in `llm-engine/llm/__init__.py` and that docstring is
authoritative:

```
storage · corpus  ->  model  ->  train  ->  orchestrate  ->  main.py
```

`llm/chat/` sits beside `corpus`: the template, the marker grammar, and the tool
table. It is imported by both the training path and the serving path, which is
the point — the format the model is trained on and the format it is prompted
with cannot drift apart if there is only one of them.

Project facts and retrieved source messages use input-only context shared by
serving and SFT; [project-memory.md](project-memory.md) documents their payloads,
token limits and private adoption of legacy tokenizer slots.

`llm/storage.py` is the authoritative map of the data tree; its module docstring
is kept true and nothing else states a path.

### Where the text comes from, and what a pack is made of

The LLM engine owns no corpora: `llm/corpus/remote.py` fetches pages through the
backend, a build directory holds the tokens, the document table and a sidecar
recording what it was made from, and phase 1 reports what the tokenization
actually produced. See [`corpus.md`](corpus.md).

`llm/backend.py` is the hop itself — the token, the base URL, the failure —
and corpus reads, gym rollouts and benchmark dispatch are its consumers.
There is one token minter because there is one seam. What each does with it
differs in exactly one way and it is load-bearing: **a read may be retried and
a write may not.** A dropped page costs a page to ask again; a re-sent gym step
runs a command twice in a real filesystem, so `Hop` makes one attempt and the
retry lives in the corpus client.

### Practising against a gym

`llm/orchestrate/rl.py` and the `llm/train/rollout.py` + `policy.py` pair — the
rollout batch, the group-relative advantage, the two optional `Transformer.
forward` arguments a ragged batch needs, and why the LLM engine drives an episode
without ever rendering one. See [`rl.md`](rl.md).
Automatic math placement, mastery gates, fixed tests and saved lesson state are
documented in [math-practice.md](math-practice.md).

### What a training run reports

`llm/train/progress.py` and `llm/train/diagnostics.py` — the frames a run
publishes, the sampling budget they live inside, and the fine-tuning telemetry
that is present on an SFT run and absent on a pretrain one. Field names are
identical from the loop to the chart, and anything unmeasured is **absent, not
zero**. See [`telemetry.md`](telemetry.md).

## Checkpoints

**A checkpoint is a model, not a lesser artifact.** It gets its own uuid, its
own `artifacts/models/<uuid>/` laid out exactly like a finished model's —
weights, a copy of the tokenizer, `meta.json` — plus the `train_state.pt` a
later run picks it up from, and its own `models` row. That sameness is the whole
design: chatting with a checkpoint, benchmarking one and fine-tuning from one
fall out of the code that already loads a model, and **no consumer of a model
knows what a checkpoint is**. What was weighed against it, and what it costs, is
[`../decisions/checkpoints-are-models.md`](../decisions/checkpoints-are-models.md).

A checkpoint row is `published=False`, so an intermediate is never offered in
chat until somebody publishes it, and it is named `<the run's name> #n` — a hash
and not the ` (n)` that `_free_model_name` spells a name collision with, because
the two mean opposite things and a reader who cannot tell them apart is worse
off than one who has never seen either.

**Cadence** (`llm/train/progress.py::checkpoint_every`): roughly
`TARGET_CHECKPOINTS` saves per run, and never fewer than one per epoch. The
interval is snapped **up** to a multiple of `report_every`, so a checkpoint
always lands on a step that is already publishing a frame. The end of every
epoch and the last step of the run also take one, deduplicated against the
interval so the same weights never land under two ids.
`TrainConfig.checkpoint_target` overrides the count; `0` means none at all, and
the epoch and final-step rules go quiet with it because they are what the
cadence is *made of*, not exceptions to it. The run's own finished model is saved regardless —
that has never been the cadence's business.

Cadence controls save frequency; retention keeps one latest automatic recovery
point plus published, explicitly kept or referenced generations. Weights appear
once per generation, with optimizer/RNG state in a separate file. SFT stores an
exact batch cursor; RL resumes at completed iteration boundaries. See
[checkpoints.md](checkpoints.md).

## Schema

Database: `LLM_ENGINE_DATABASE_URL`, default `postgresql+psycopg://localhost/slate_engine`,
migrated with `alembic upgrade head`.

`models` is the index; the weights are on disk under `artifacts/models/<id>/`,
which is why the id and not the name is stable. `parameters` is `meta.json`
verbatim — `{"config", "result"}`. **`config` is what was asked for and
`result.architecture` is what was built**, and only the second is always
answerable: a from-scratch pretrain's `TrainConfig` states dimensions, heads,
layers, ff_multiplier and ctx_window itself, but `SftConfig` states none of
that on purpose (see its docstring) and a *resumed* pretrain states none
either — saying nothing is how a resume asks for the checkpoint's shape. So
both pipelines record the resolved five under `result.architecture`, which is
where `architectureOf` and the model row's `context` read it from, and where a
resumed run's real shape is the only place it can be found. A checkpoint row
answers there too, off its own frame — see [`checkpoints.md`](checkpoints.md). `result.base_model`
carries the same fine-tune's ancestry, mirroring `config.base_model`. Four
nullable columns carry a run's intermediates, and every pre-existing row keeps
them null — which is exactly what being a finished model means:

- **`checkpoint_index`** — the 1-based ordinal within the run, and the `#n` the
  name ends with. **`checkpoint_index IS NOT NULL` is the whole definition of
  "this row is an intermediate"**, and there is deliberately no boolean beside
  it that could disagree. Every listing filters on it.
- **`checkpoint_step`** — the global step it was saved at, and the order
  siblings are read back in; `created_at` ties between two checkpoints of a fast
  run.
- **`run_id`** — the run that produced it, carried by the intermediates *and* by
  the finished model, so a run's whole output groups by one value. The finished
  model is credited after the fact (`handlers.attach_run`), because the pipeline
  is handed a config and never a run id.
- **`parent_id`** — the finished model an intermediate belongs to; null on a
  finished model, which is the parent.

**Neither uuid is a foreign key, and for two different reasons.** `run_id` is
not one because run rows are prunable and the grouping has to outlive them — the
need to preserve historical identity after queue pruning. `parent_id` is not one because the
parent row does not exist yet when a run's first checkpoint lands: the
constraint would refuse the very insert it exists to describe. Both are indexed;
`checkpoint_index` is not, because it is only ever tested for null on a column
that is null for almost every row.

Two consequences that are not in the types. Deleting a model deletes its
checkpoint rows in `queries.delete_model` rather than by cascade, and their
**directories** are the caller's to remove — `store/` has never heard of the
disk. And **which checkpoint a resumed run started from gets no column**: it
lives in `parameters.config.resume_from`, one string nothing joins on.

Conversation and attachment tables have moved to Rust. Benchmark rows have moved
to bench-engine. [Execution schema](execution.md) describes `execution_jobs`,
`training_runs`, `tokenizer_builds` and `benchmark_dispatches`, including the
migration preserving historical IDs.

## Routes

The full surface — auth, the SSE chat protocol, the model listing and its
checkpoint filter, context extension, the prune floor and the attention readout
— is in [`api.md`](api.md).

## Tools

A tool call is a leaf block inside an assistant turn. Every tool is called
through the same marker pair, with the tool named in the JSON body:

```
<|tool-call|>{"tool": "web-search", "argument": "tallest mountain in europe"}<|/tool-call|>
<|tool-call|>{"tool": "bash", "argument": "grep -rn cat . | head -3"}<|/tool-call|>
```

Its answer comes back as a block of its own, between that assistant turn and the
next — the same *outside every turn* position the system block occupies at the
front of a conversation. The result repeats the call it answers, which is what
tells the model which of its calls it is being handed:

```
<|tool-result|><|tool-call|>{"tool": "web-search", "argument": "tallest mountain in europe"}<|/tool-call|>
               <|list|><|item|>Mount Elbrus — 5,642 m<|/item|><|/list|><|/tool-result|>
```

**One marker for every tool, not a pair each.** Adding a tool is a row in
`llm/chat/tools.py` and no vocabulary change at all, which is what lets a model
trained in March call a tool invented in August. It costs the grammar's ability
to refuse a name — that check moved to `tools.parse`, after the fact — and it
asks the model to keep a small JSON object balanced. Both costs, and why they
were taken, are in
[`../decisions/generic-tool-calls.md`](../decisions/generic-tool-calls.md);
[`../decisions/tool-calls.md`](../decisions/tool-calls.md) is still the record
of why a turn that calls one runs twice.

**There is exactly one renderer of that block, and callers holding their own
conversation use it.** A `ChatTurn` on `/chat` and `/chat/stream` may name the
`tool` and `argument` its text answers, and `orchestrate/chat.py::_history`
renders it through `tools.render_result`; leaving them out means the text is
already rendered, which is how a tool message read back from a stored chat
arrives. The backend's gym driver takes the first path — a second copy of this
layout in Rust would be free to drift into a shape the model was never trained
on. See [`../decisions/gyms-are-driven-by-the-backend.md`](../decisions/gyms-are-driven-by-the-backend.md).

**A continuation can carry a person's own words alongside the tool's.**
`Ask.interject` on `POST /chat/prepare` rides beside
`tool_result` only — refused if sent with `text`, since a continuation is not a
question — and becomes a second stored message, `Role.user`, right after the
tool result; both land in `ChatConfig.turns` in that order (tool result, then
the interjection) before the model is asked to continue. It carries no
`attachment_ids` of its own, the same rule a bare `tool_result` already follows:
an interjection is text steering that arrived while the reply was still
generating, not a new question with its own uploads. See
`docs/backend/README.md`'s "Interjecting mid-turn" for where the backend
buffers a person's message and folds it into the continuation request.

**A body that names no tool is not a call.** `tools.parse` reads the argument
under any of `argument`/`query`/`command`/`input`/`text`, nested in `args` or
not, because that is what a small model writes — but a mangled object, or one
naming a tool this build does not have, makes `_ToolCalls.observe` return None.
No frame is emitted, nothing runs, and the reply stands as written. That is the
one failure mode the grammar can no longer prevent, so it is made cheap.

Three invariants, each enforced somewhere different:

- **`<|tool-result|>` is absent from `markup.PAIRS`**, so it lands in
  `Rules.forbidden` and the model can never emit one — it cannot claim to have
  been told something it was not. The result is likewise never a training
  target (`template.training_segments`), for the same reason the budget marker
  is not: training it would teach the model to invent search results.
- **A call may only open at the top level and admits no markup inside it**
  (`Rules.top_level`, `Rules.leaf`). Nested in a `<|par|>` it would read as
  prose and in a `<|codeblock|>` as a sample, and the backend would execute it
  either way.
- **A vocabulary that predates a tool is tolerated, not rejected.** Document
  markers missing from a tokenizer raise; tool markers are skipped, and that
  model simply cannot call anything. This is the one place the two kinds of
  marker differ.

Adding a tool: two reserved slots in `corpus/special.py`, one row in
`chat/tools.py`. Nothing else in the LLM engine changes — the grammar, the
generation loop and the SFT renderer all read that table. Then run the migration
below, and add an `impl Tool` on the backend.

### Adopting a new marker into vocabularies already on disk

The special block is a fixed 64 slots padded with `<|reserved_N|>`, so naming a
marker consumes a slot instead of shifting every id after it. The consequence is
that **a vocabulary saved before a marker existed already has that marker's id**
— it is just still spelled as the placeholder. So adoption is a rename, not a
retrain:

```sh
.venv/bin/python -m scripts.relabel_reserved_slots            # dry run
.venv/bin/python -m scripts.relabel_reserved_slots --apply
```

Ids do not move, so `corpus.bin` stays valid and every checkpoint still loads.
The renamed row is at its original init until an SFT run touches it, which
`train/warm_start.py` already handles off the dataset's `special_ids_seen`.

Memory-bearing requests adopt only the memory pair on a private tokenizer copy
and keep cached/saved tokenizers unchanged. Memory SFT outputs save that adopted
copy; ordinary SFT keeps the base tokenizer unchanged. See
[project-memory.md](project-memory.md#legacy-tokenizer-compatibility).

## Conventions

<!-- TODO: patterns code in this module must follow. -->

## Gotchas

- `tests/test_dataflow_trace.py` is the contract that keeps the LLM engine honest:
  one fixed question, recorded at five layer boundaries under a pinned seed.
  Re-record with `python -m tests.fixtures.record` only when a behaviour change
  is intended, and commit the diff as the evidence of what moved.
- **`bpe.load` hands back a shared object, not a fresh one.** It is memoized on
  the tokenizer file's path and mtime, because `/tokenize/count` is called for
  every message stored and again on every pause in the composer, and re-parsing
  a `tokenizer.json` per call dominated both — 9.9 ms of parse for 0.055 ms of
  encoding, measured in `corpus/bpe.py`. Nothing mutates a loaded
  tokenizer today and nothing may start: it would be mutating every caller's.
  The mtime is in the key so that an SFT run saving its vocabulary over the base
  model's is picked up rather than served from the copy read before it moved —
  `tests/test_vocabulary_cache.py` is what notices.
- **Training plans below the allocator cap, not right up to it.** The planner
  counts already-resident SFT weights once and leaves headroom inside the cap for
  temporary allocations and cached fragments. It recomputes only the blocks
  needed to fit; batch size, context and the training schedule do not change.
  See [`memory.md`](memory.md) for the two budgets and their limits.
- **Diagnostics must not become the largest training allocation.** Recording
  stays in a separate no-grad pass; update snapshots live on CPU, and a
  diagnostic CUDA OOM disables samples instead of discarding the run. The
  telemetry contract and sampling cadence are in [`telemetry.md`](telemetry.md).
- **A failed worker's traceback can retain its GPU tensors.** The runner clears
  failed frames, then collects cycles and releases the allocator cache at the
  run boundary. A busy-looking card alone does not prove training is progressing;
  read the step and loss frames. See [`memory.md`](memory.md).
- **The chat cache holds a model until something evicts it.**
  `orchestrate/chat.py`'s `_CACHE` keeps one model resident per process and only
  ever replaces it when a later chat names a different one. One device means that
  is memory a run cannot have, so `api/runner.py` calls `chat.evict()` before
  starting one and the next chat reloads.
- **Anything asking "which ids are markers?" must use `special.block_names()`,
  not `token_list()`.** A vocabulary older than a marker still calls that slot
  `<|reserved_N|>`, and matching only the current spelling drops those ids out
  of the range the grammar masks — which lets a model emit a placeholder as
  though it were text. That is a real regression that adding the tool markers
  caused once.

<!-- TODO: the rest. -->
