# Search result cache

- **Date**: 2026-08-22
- **Status**: Implemented

## Context

`web-search` works, but every call pays for a live fetch — the same question
asked five minutes apart by two different conversations runs the search
twice, against a provider whose free tier is metered by the request
(`docs/decisions/tool-calls.md`). Most of what gets searched does not change
minute to minute, so most of that cost is avoidable.

There was a second idea on the table: a knowledge graph, queried before a
search and populated by what came back, so the model could reason over
related facts rather than just re-fetching them. That was set aside for now —
it needs entity/relation extraction this repo has nowhere, on top of a
storage decision — in favour of the narrower thing actually asked for: "check
if we already have it, redo it if it's stale."

## Decision

**A keyed cache with a TTL, nothing fancier.** `WebSearch::run` normalizes
its query into a key, checks the cache, and returns a hit fresh enough to
trust; otherwise it fetches live and writes the result back under that key.
One tool, one marker (`<|web-search|>`), unchanged from the model's side — a
model already trained to reach for it needs no retraining, and there is
nothing to teach it about a second tool or a "the cache didn't have it"
signal. See the rejected `<|search|>` split below.

**The cache lives in the data engine, as a dataset.** A search result is data
on the same terms anything else pulled from the Hub is: `web-search-cache`,
`origin = search` (a new `DatasetOrigin`), `owner_id = NULL` because it is a
fact about the world and not about whoever asked first. It is created
`ready` on first write rather than going through `pulling` — there is no
batch to wait out, every row is complete the moment it lands. Storing it this
way is what makes "use what got searched as training data later" free: it is
already `DatasetSelection`-able (`llm-engine/llm/config.py`) the moment it
exists, without work in the LLM engine.

A re-fetch **appends a fresh document rather than overwriting the old one** —
the same shape a re-pulled Hub dataset already has, "holding more documents
under the same id" (`docs/llm-engine/README.md`). `Document` gained `created_at`
(nothing there had a per-row timestamp before) and `cache_key` (nullable —
null on everything pulled from the Hub or produced by a pipeline, which are
found by ordinal, never by content). The freshest row under a key, by
`created_at`, is what a lookup answers.

**Freshness is the backend's call, not the data engine's.** A lookup returns
the newest match and when it was written; `WebSearch::run` compares that
against `SEARCH_CACHE_MAX_AGE_DAYS` itself. Keeping the threshold out of the
data engine's request means it can be tuned by restarting the backend, no
migration and no redeploy of the other service.

**`tools::SearchCache` is a trait, not a concrete dependency.** `tools` is
documented (`docs/backend/README.md`) as a leaf that knows about the outside
world and the marker vocabulary and nothing about this project's other
services — it must not learn the shape of the data engine's API. The trait
is declared in `tools`; the implementation, `api::search_cache::DataEngineCache`,
lives in `api`, which already depends on both `tools` and `data-engine`, and
is wired into `Registry::from_env` from `main.rs`. `WebSearch` runs exactly
as it did before the cache existed when no `SearchCache` is wired in.

**The lookup is a `POST` with a body, not a `GET` with a query string.** A
cache key is an arbitrary, possibly long, possibly unicode phrase, and a body
is the shape that needs no percent-encoding on the way in — the backend has
no URL-encoding dependency today and this did not need to become the reason
it gained one.

**A live failure with a stale entry on hand returns the stale entry.**
`search::mod::fit` already treats "something beats nothing" as a real
principle — a budget too small for a whole result still returns one rather
than reporting nothing found. The same reasoning applies here: a search that
once found something and cannot be reached again should say what it found,
not that it failed.

## Rejected alternatives

- **A knowledge graph.** Richer, and the reason "graphs allow way more" was
  the opening pitch — but populating one needs entity/relation extraction
  over search snippets, which is new ML work this repo has nowhere today, on
  top of the storage question this decision already answers. Worth its own
  measurement before it is architecture, the way `explore/knowledge-graph`
  measured whether the *model's own embeddings* reflect corpus structure
  before anything was built on that question. This cache is the layer under
  it either way.

- **An in-memory cache inside `tools`.** Simpler and needs no service call —
  and was the plan right up until "store it in the data engine" came up.
  Rejected in favour of the data engine once it was clear the same rows
  double as future training data for free the moment they are a real
  dataset; an in-memory map would have to be exported into that shape later
  by hand.

- **A second marker, `<|search|>`, cache-preferred, with `<|web-search|>`
  reserved for "no, actually check the web."** Teaches the model to choose
  between two tools and, if a within-turn escalation from one to the other
  was ever wanted, needs `MAX_TOOL_ROUNDS` raised from 2 — a real latency and
  token cost for every turn that escalates, on a model whose whole tool-call
  design exists because it is too small to be trusted with unbounded rounds.
  Rejected for the same reason `docs/decisions/tool-calls.md` rejected a
  generic `<|tool|>name:args`: the model does not need to be taught anything
  new to get the win here, and reaches for `<|web-search|>` today. Caching
  transparently under the tool it already knows gets the entire benefit with
  no training-data dependency at all.

- **Filtering the cache dataset out of `GET /datasets`.** Ownership does not
  gate that route today — it returns every dataset to every caller regardless
  of who created it — so "hidden" would need an explicit exclusion. Decided
  against: the dataset is real and nothing is lost by it appearing in a
  listing nothing points at yet.

## Consequences

- `Registry::from_env`/`Registry::with_config`/`WebSearch::from_config` all
  take an `Option<Arc<dyn SearchCache>>` now. Every call site that does not
  care passes `None` explicitly rather than the parameter defaulting away —
  consistent with the rest of this crate's preference for a call site that
  says what it means over one that relies on a default it never looks at.
- A cache *hit* goes from "instant" (nothing existed to ask) to one local
  HTTP call to the data engine plus a Postgres query — real, but on the order
  of low tens of milliseconds on one machine, and it happens inside the same
  spawned task `docs/decisions/tool-calls.md` already overlaps with
  generation, not on a new path.
- `Document.created_at` is now a real column on every document, not only a
  cached search's — nothing in this schema carried a per-row timestamp
  before, and every other table already does.
