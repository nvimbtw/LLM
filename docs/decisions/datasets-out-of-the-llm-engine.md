# The LLM engine stopped owning training data

**2026-08-19**

The LLM engine keeps models, vocabularies and the corpora it derives. The prose
and conversations it trains *on* belong to the data engine, and a run reads them
a page at a time through the Rust backend. `llm-engine/data/input/` and
`llm-engine/data/sft/` are gone, along with the directory walks that read them.

## What forced it

The data engine already stored datasets as rows, cleaned them, and recorded the
provenance of every derived one ([dataset content in
Postgres](dataset-content-in-postgres.md)). The LLM engine had a second, parallel
corpus store: a tree of `.txt` shards with no provenance, catalogued by scanning
directories, selected by folder name. Two answers to "what can this train on",
one of them worse in every respect a person cares about — where did this come
from, what was done to it, which twenty documents did the preview show.

Keeping both meant an import step, and an import step means a copy: the same
corpus in two places, with the LLM engine's copy the one that is actually trained on
and the one nothing can say anything about.

## What was weighed

**The backend pushes at submission.** A run is queued, sits in a line, and packs
when it reaches the front — so the data has to be *somewhere* in between. That
somewhere is a staged copy on the LLM engine's disk, which is the thing being
removed. Rejected.

**The backend pushes at execution.** Requires the backend to know when a run
reaches the front of the LLM engine's own queue. The only way it learns that is the
LLM engine telling it, which is a pull with extra steps. Rejected.

**The LLM engine pulls, through the backend.** Chosen. It inverts the direction of
travel for the first time in this repository — everything else flows frontend →
backend → service — and that is a real cost, paid for the one thing the other
two shapes cannot do: fetch the documents at the moment they are needed, and
write none of them down.

The invariant that matters is untouched. The Python services still never
speak: a request arrives at `/internal`, and the backend re-mints toward the
data engine carrying the same `act` claim, so a run reads exactly the datasets
the person who queued it could have read.

### The export endpoint, which looked like the seam and was not

`GET /datasets/{id}/export` was built for this and its docstring said so. It is
still unconsumed, because two properties of a whole-corpus response make it the
wrong shape here:

- The backend buffers a forwarded response into a `serde_json::Value`.
  Forwarding an export would need a byte-stream passthrough the proxy does not
  have, written for one caller.
- A connection dropped forty minutes into a pack costs the whole pack. A page
  costs one retry.

So a run reads `GET /datasets/{id}/documents?offset=&limit=` instead — a route
that already existed, that the existing proxy already forwards, and that is
resumable by construction. The LLM engine sizes its pages by measured bytes rather
than rows, because `limit` counts documents and a thousand books is not a
thousand snippets; the data engine's own `MAX_PAGE` of 200 is what binds for
small documents.

### The span table, which nearly took the machine with it

A packed corpus records where each document starts. That table was a list of
`DocSpan` objects written inline into `segments.json`, and it was affordable
only because **one file was one span**: spans averaged ~40k tokens, so a
390M-token corpus was ~10k of them.

Reading documents makes a span a real document — the same corpus becomes ~800k
spans, and the cost lands three times over: the objects during the pack, a
`json.dumps` building a ~60 MB string before writing it, and `EpochStream`
copying the list *per epoch*. On a 7 GiB machine that is the failure the corpus
memmap and the epoch index arrays were introduced to prevent, arriving again by
a different door.

Two ways out were weighed. **Grouping documents into shard-sized spans** would
have reproduced the old granularity exactly, for no new memory and no semantic
change — but it keeps a property nobody would choose on purpose: a span holding
many concatenated documents means training predicts across document boundaries
inside it. **Columnar spans** — `start`, `length`, `stage` and a dataset code as
four numpy columns in a `.npy` beside `corpus.bin` — cost 22 bytes a row against
145 for an object, load by mapping, and make true per-document boundaries
affordable rather than merely possible. Chosen.

### What stayed

**`llm-engine/data/eval/`.** 92 KB of held-out prose and conversations, against
134 MB of training corpora, and the reads are already capped in code at 4 MiB
and 256 KiB. Moving it would buy nothing and cost the guarantee the benchmark
rests on: held-out text must never have been trained on, and today that is
structural — `eval/` is a directory the selection code cannot see. In the data
LLM engine it becomes a row that looks like every other row, and "never selectable"
turns into a flag some future selection has to remember to honour.

**The tokenizer's sample.** A vocabulary build writes the bounded 64 MiB it
learned merges from into the build directory. This is the only text the LLM engine
still writes down, and dropping it was considered and rejected on measurement:
`bpe.train` takes file paths and counts across cores, and holding the same
64 MiB as Python strings costs more memory than writing it does. It is a build
artifact, not a corpus, and it is what makes a wrong-looking tokenizer
answerable after the fact.

## What it changed for a caller

- **A selection is a dataset id**, not a folder name. A name is something a
  person can change in the data view without meaning to change what a run trains
  on.
- **`count` caps documents**, by `ordinal`. It used to mean "the first N files
  sorted by name", which was reproducible only by accident of how a listing
  sorted; `ordinal` is the order the data engine maintains precisely so that a
  fixed prefix is a fixed prefix.
- **There is no "everything".** An empty selection used to mean every corpus
  under `input/`, which is a reasonable default for a directory one person
  curates and a bad one for a shared database anyone can pull into. A run names
  its datasets or it is refused.
- **A corpus key carries what answered, not only what was asked**: the dataset's
  document count and `updated_at`, alongside its id and cap. A dataset pulled
  again holds more documents under the same id, and without those two a run
  would reuse a corpus packed from less data and report success.

One accepted wart. A dataset's *name* is deliberately absent from the key's
digest — a rename changes no token — but it is in the human-readable slug, and
the slug is part of the path. So renaming a dataset does re-pack. That is the
price of a directory listing a person can read, and a rename is rare.

## What it cost

The LLM engine calls the backend, which nothing did before: a `ServiceCaller`
extractor, an `aud: backend` token, and `contracts/backend-token.json` — the
only token in this project that Python signs and Rust verifies. Every training
run now depends on the backend being up, where before it depended only on its
own disk. And `BACKEND_SERVICE_SECRET` unset leaves `/internal` open, which is
the development default the Python services already set, on the same loopback
trust boundary.
