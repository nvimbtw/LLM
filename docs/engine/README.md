# Engine — PyTorch training, inference, evaluation

> **Status: partial.** Sections below are the required shape (see
> [`../README.md`](../README.md)); the ones still marked TODO are not written
> yet. Fill a section the first time you work in the area it covers — do not
> leave a heading empty behind you.

## What it owns

Models, conversations, runs and scores. The backend owns users and presets and
nothing else; requests flow frontend → backend → engine and back the same way.
See [`../decisions/`](../decisions/) for the two-database split.

It owns the **vocabulary** as well, which is the ownership boundary that catches
people out: the marker table, the chat template, and the grammar the model
decodes under are all here, and the other two modules read them rather than
restating them. A tool's *name* is decided here (it is a marker stem); a tool's
*behaviour* is the backend's.

<!-- TODO: the rest of the ownership boundary — what the engine will never ask
     the backend for, and vice versa. -->

## Tech stack

<!-- TODO: PyTorch, FastAPI, SQLAlchemy/alembic, psycopg — and the why where a
     choice was contested. -->

## Layout

The layering is stated in `engine/llm/__init__.py` and that docstring is
authoritative:

```
storage · corpus  ->  model  ->  train  ->  orchestrate  ->  main.py
```

`llm/chat/` sits beside `corpus`: the template, the marker grammar, and the tool
table. It is imported by both the training path and the serving path, which is
the point — the format the model is trained on and the format it is prompted
with cannot drift apart if there is only one of them.

## Schema

Database: `ENGINE_DATABASE_URL`, default `postgresql+psycopg://localhost/slate_engine`,
migrated with `alembic upgrade head`.

`messages.role` is a Postgres enum of `user | assistant | tool`. A tool's answer
is stored as a message of its own, between the assistant turn that called for it
and the one that uses it, because the history a later turn is rendered from has
to carry what the model was told and not merely that it asked.

<!-- TODO: the remaining tables, who writes each, invariants not in the types. -->

## Routes

Every route is behind `Depends(auth.require_service)`; the backend signs a
short-lived token naming who it acts for (`ENGINE_SERVICE_SECRET` on both sides,
unset leaves the engine open and it says so at startup).

`POST /chats/{id}/messages/stream` is the one with a protocol rather than a
payload. It takes **exactly one** of `text` (a person asking) and `tool_result`
(a tool answering a call the last reply made) — both is a caller that has lost
track of which, neither is nothing to answer, and either is a 400.

It answers as SSE, in three event kinds:

```
event: token      data: {"token": "..."}                      (repeated)
event: tool_call  data: {"tool": "...", "argument": "..."}     (a call closed)
event: done       data: {…message, "usage": {…, "tool_calls": [...]}}
event: error      data: {"error": "..."}
```

`tool_call` fires the moment the closing marker is sampled, not at the end of the
turn, so the caller can start the work while the rest of the reply is still
generating. The row is written when the last token lands, so a dropped
connection loses the reply rather than storing half of one.

## Tools

A tool call is a leaf block inside an assistant turn, spelled with the tool's own
marker pair:

```
<|web-search|>tallest mountain in europe<|/web-search|>
```

Its answer comes back as a block of its own, between that assistant turn and the
next — the same *outside every turn* position the system block occupies at the
front of a conversation. The result repeats the call it answers, which is what
tells the model which of its calls it is being handed:

```
<|tool-result|><|web-search|>tallest mountain in europe<|/web-search|>
               <|list|><|item|>Mount Elbrus — 5,642 m<|/item|><|/list|><|/tool-result|>
```

**A marker per tool, not a generic `<|tool|>name:args`.** A model this size
cannot be trusted to spell a tool name and then an argument in one string, and a
dedicated marker is one embedding row the grammar can police. See
[`../decisions/tool-calls.md`](../decisions/tool-calls.md).

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
`chat/tools.py`. Nothing else in the engine changes — the grammar, the
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

## Conventions

<!-- TODO: patterns code in this module must follow. -->

## Gotchas

- `tests/test_dataflow_trace.py` is the contract that keeps the engine honest:
  one fixed question, recorded at five layer boundaries under a pinned seed.
  Re-record with `python -m tests.fixtures.record` only when a behaviour change
  is intended, and commit the diff as the evidence of what moved.
- **Anything asking "which ids are markers?" must use `special.block_names()`,
  not `token_list()`.** A vocabulary older than a marker still calls that slot
  `<|reserved_N|>`, and matching only the current spelling drops those ids out
  of the range the grammar masks — which lets a model emit a placeholder as
  though it were text. That is a real regression that adding the tool markers
  caused once.

<!-- TODO: the rest. -->
