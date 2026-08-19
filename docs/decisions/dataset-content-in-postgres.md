# Dataset content lives in Postgres, not on disk

**2026-08-19**

The data engine stores documents as rows in its own database. It writes no tree
under its module directory, and it does not write into the engine's.

## What forced it

`engine/data/` holds packed corpora, model checkpoints and the raw text they
came from. It is gitignored, but gitignored is not invisible: an editor opening
the repository still indexes it, and the tree had grown large enough to make
that a noticeable wait. A second module acquiring *more* corpora would have
doubled the problem, and the obvious layout — `data-engine/data/` beside it —
was the version that made it worse.

## What was weighed

**Files under `data-engine/data/`.** The engine's own arrangement, and the
cheapest to write: the fetch scripts already produce sharded `.txt`. Rejected
because it reproduces exactly the cost that prompted the module, and because
the questions this service has to answer — "the first twenty documents",
"everything longer than 200 characters", "was this one dropped, and by what" —
are queries. Answering them over a directory means either reading shards into
memory or building an index beside them, and an index beside a directory is a
database with extra steps.

**Publishing into `engine/data/input/<name>/`.** Tempting, because the engine
catalogues datasets by scanning that directory and a cleaned corpus written
there would appear in the train page's picker with no change to the engine at
all. Rejected for the same reason as above, plus a worse one: it makes the data
engine a writer to a tree another module owns, which is the ownership boundary
this repository is most careful about. The bridge is
`GET /datasets/{id}/export` instead — a seam, not a shared directory.

**A Postgres schema inside `slate_engine`.** Rejected: this repository draws its
module boundary at the database, not the schema, and cross-schema foreign keys
to engine tables are the coupling that boundary exists to prevent. It would also
have cost the test suite — SQLite has no schemas, so the fast in-memory path
would have gone and every store test would need a live server.

## What it costs, and what pays for it

A corpus in a table is not free. Backups grow with the data, a large pull is a
long transaction stream rather than a file copy, and `SELECT *` over a
corpus-sized table is a way to run the process out of memory.

So the constraints are structural rather than advisory:

- **Every pull carries a row cap.** There is no "fetch it all"; a missing or
  nonsensical cap becomes a default, and a ceiling in `api/handlers.py` is a
  deliberate edit rather than a number typed into a form.
- **Inserts are batched**, and a cancelled run keeps what it already wrote.
- **Nothing in `store/queries.py` returns a whole dataset.** Reads are paged,
  the export streams with `yield_per`, and the functions that touch `documents`
  all take a bound.

## What it bought

The repository stopped growing with the data. Beyond that: a preview over the
first twenty documents is an indexed read rather than a shard scan, provenance
is a column rather than a naming convention, and deleting a dataset is one
statement rather than a directory walk that has to not go wrong.

## Since

The engine stopped owning corpora entirely, which settled the open half of the
"publishing into `engine/data/input/`" question above: there is no such tree any
more, and nothing writes into another module's directory because there is no
directory to write into.

The bridge turned out **not** to be `GET /datasets/{id}/export`. A run reads
`GET /datasets/{id}/documents` a page at a time, through the backend's
`/internal` routes — paged rather than streamed because the backend buffers a
forwarded response, and because a connection dropped mid-pack costs one page
instead of the whole corpus. The export endpoint stands, unconsumed, as the
honest way to hand a whole dataset to something outside this system.
