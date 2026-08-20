# Engine — PyTorch training, inference, evaluation

> **Status: partial.** Sections below are the required shape (see
> [`../README.md`](../README.md)); the ones still marked TODO are not written
> yet. Fill a section the first time you work in the area it covers — do not
> leave a heading empty behind you.

## What it owns

Models, conversations, runs and scores. The backend owns users and presets and
nothing else; requests flow frontend → backend → engine and back the same way.
See [`../decisions/`](../decisions/) for the two-database split.

What it does **not** own, and this is the boundary that moved most recently:
**the text it trains on**. Corpora are the data engine's, and a run reads them a
page at a time through the backend — see
[`../decisions/datasets-out-of-the-engine.md`](../decisions/datasets-out-of-the-engine.md).
What is left under `engine/data/` is derived (packed corpora, vocabularies,
weights) plus `eval/`, which is held out from training and is therefore not
training data. A run keeps no copy of what it read.

It owns the **vocabulary** as well, which is the ownership boundary that catches
people out: the marker table, the chat template, and the grammar the model
decodes under are all here, and the other two modules read them rather than
restating them. A tool's *name* is decided here (it is a marker stem); a tool's
*behaviour* is the backend's.

<!-- TODO: the rest of the ownership boundary — what the engine will never ask
     the backend for, beyond datasets, and vice versa. -->

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

`llm/storage.py` is the authoritative map of the data tree; its module docstring
is kept true and nothing else states a path.

### Where the text comes from

`llm/corpus/remote.py` is the client side of the one hop that runs toward the
backend. The engine owns no corpora: what a run trains on is a dataset in the
*data* engine's database, and the two Python services never speak, so a run asks
the backend and the backend re-mints toward the other service carrying the same
acting caller. The seam itself is documented in
[`../backend/README.md`](../backend/README.md).

| Variable | Default | Meaning |
|---|---|---|
| `BACKEND_URL` | `http://127.0.0.1:8080` | |
| `BACKEND_SERVICE_SECRET` | unset | Shared with the backend. Unset signs the token anyway, because `act` still has to travel |

Three properties, and each of them is a decision rather than an implementation
detail:

- **Nothing is written down.** A page is fetched, encoded and dropped. A cache or
  a staging directory would be the engine owning a corpus again.
- **Pages, not the export stream.** The data engine can serve a whole dataset as
  one NDJSON response; a connection dropped forty minutes into a pack would then
  cost the whole pack, where a page costs one retry.
- **A page is sized in bytes.** `limit` counts documents and a thousand books is
  not a thousand snippets, so the last page's measured weight sets the next
  one's limit. The data engine's own ceiling of 200 rows is what binds for small
  documents; the byte target is what binds for large ones.

All three build paths read through it: the pretrain pack, the vocabulary's
strided sample, and the SFT render. Nothing in `llm/` opens a corpus file any
more.

**A selection is a dataset id, not a name.** `{"id": ..., "count": N}` caps to
the first N *documents* by `ordinal` — where it used to mean the first N files
sorted by name, which was reproducible only by accident of how a listing sorted.
A name is carried for readability and for the span tag; the data engine's own
name wins.

**There is no "everything".** An empty selection used to mean every corpus under
`input/`, which is reasonable for a directory one person curated. A shared
database anyone can pull into is not that, and a run that silently widened to
whatever was added last week would key apart every time somebody else worked. A
run names its datasets or it is refused.

**A corpus key includes what answered, not only what was asked.** A dataset
pulled again holds more documents under the same id, so the key carries the
document count and `updated_at` alongside the id and the cap. Without them a run
reuses a corpus packed from less data and reports success. A dataset's *name* is
deliberately not in the digest — but it is in the readable slug, and the slug is
part of the path, so a rename does re-pack. That is the price of a directory
listing a person can read, paid knowingly.

**A conversation arrives as `[{role, content}]`** and is rendered from
`{system?, turns: [{role, text}]}`. `sft._conversation` bridges the two, and the
part that matters is that a **system message is not a turn**: left in the list it
reaches `_turns` as an unknown role and drops the whole conversation — every
conversation, for a dataset that has them.

### What a packed corpus is made of

A build directory holds the tokens (`corpus.bin`, plus `mask.bin` for SFT), the
**document table** (`spans.npy`), and a sidecar (`segments.json`) recording what
the build was made from. The three are keyed together — see `corpus_key` and
`dataset_key` for what a key is a function of.

The table is separate from the sidecar, and columnar, because it is the one part
that scales with the *number* of documents rather than with their size: 22 bytes
a row against 145 for an object, and no JSON encoding of the whole table held in
memory to write it. That is affordable at one span per shard and is not at one
span per document, which is the granularity a corpus fetched from the data
engine has. `llm/corpus/spans.py` carries the measurements.

A build packed before the table moved out of `segments.json` carries its rows
inline there. Those are read, not invalidated — a repacked corpus is hours of
work, and `spans.from_sidecar` takes either shape.

## Schema

Database: `ENGINE_DATABASE_URL`, default `postgresql+psycopg://localhost/slate_engine`,
migrated with `alembic upgrade head`.

`messages.role` is a Postgres enum of `user | assistant | tool`. A tool's answer
is stored as a message of its own, between the assistant turn that called for it
and the one that uses it, because the history a later turn is rendered from has
to carry what the model was told and not merely that it asked.

`attachments` holds files a person put in the composer — bytes in a `bytea`
column, hanging off the message they were sent with. Two things about it are
load-bearing and neither is in the types:

- **Nothing reads them.** They are stored and served back, and no part of one
  reaches a prompt. Nothing under `llm/` may import `api/routes/attachments.py`
  or name the table; `attachment_ids` on a chat request binds rows and does
  nothing else. See
  [`../decisions/attachments-are-stored-not-read.md`](../decisions/attachments-are-stored-not-read.md).
- **`message_id` is nullable and `owner_id` is what scopes a row.** A file is
  uploaded when it is picked, which is before the turn it belongs to exists and,
  on a new conversation, before the chat does. Binding happens when the turn is
  stored; an unbound row is reachable by its owner and nobody else, and only an
  unbound row can be deleted — a sent turn is history.

Bytes in a table only work with caps, and there are two: `MAX_BYTES` (50 MB a
file) and `MAX_PER_MESSAGE` (5 a turn), both in `api/routes/attachments.py` and
mirrored at the edge by the backend's route. It is their **product** that is the
real constraint — a turn's files pass whole through three processes — so the two
move against each other rather than independently. `Attachment.content` is a deferred
column, so listing a conversation's files does not load them.

<!-- TODO: the remaining tables, who writes each, invariants not in the types. -->

## Routes

Every route is behind `Depends(auth.require_service)`; the backend signs a
short-lived token naming who it acts for (`ENGINE_SERVICE_SECRET` on both sides,
unset leaves the engine open and it says so at startup).

`/attachments` is the one route family whose payload is not JSON: `POST` takes a
multipart file, `GET /attachments/{id}` answers with the bytes and the headers
that say what they are, and `DELETE` discards a staged one. All three are scoped
by the acting caller, like everything else here.

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
