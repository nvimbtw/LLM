# Datasets the backend writes

Split out of [`README.md`](README.md) under the ~250-line rule (see
[`../README.md`](../README.md)). Two datasets here arrive outside the `Run`
path — not pulled from the Hub, not derived from anything stored — written a row
at a time by the Rust backend as a side effect of something else happening.

They share a shape, and it is worth stating once: `owner_id = NULL`, created
`ready` rather than `pulling`, counters bumped in place under a lock on the
dataset row rather than recomputed by `finish_dataset`. `ready` because every
row is complete when it is written and there is no batch to wait out; the lock
because `ordinal` is assigned from `document_count`, and two concurrent writers
without it would hand out the same ordinals.

## The search cache

`web-search-cache` (`origin = search`, `owner_id = NULL`) is where the
backend's `web-search` tool lands a result instead of re-fetching it live
every time — see
[`../decisions/search-result-cache.md`](../decisions/search-result-cache.md)
and [`../backend/README.md`](../backend/README.md)'s "The search cache". It
is a dataset like any other pulled from the Hub — created `ready` the moment
anything is written to it, since there is no batch to wait out — which is
what makes it trainable later with zero work here: `store.queries.cache_dataset`
finds or creates the reserved row, `add_document` appends under a lock on it,
and `latest_by_key` is the read a re-fetch decides freshness against.

A re-fetch **appends rather than overwrites** — the same shape a re-pulled
Hub dataset already has — so the newest row under a key is the one a lookup
answers, and nothing here is ever mutated in place. Freshness itself is not
this service's decision: `POST /search-cache/lookup` returns the newest
match and when it was written, and the caller (`WebSearch::run`, in the
backend) decides whether that is fresh enough.

## Gym demonstrations

`origin = gym`, `kind = sft`, `owner_id = NULL`, `ready` on arrival — the search
cache's shape, for the same reasons, with one difference: there is one cache and
there are as many demonstration sets as there are task types and difficulties
worth training on separately, so the **name comes from the caller** rather than
being a reserved constant. A second request under the same name appends.

The rows are expert trajectories from a gym, played through its sandbox and
written here by the backend — never by the gyms service, which cannot reach this
one. They are stored so that they can be trained on with no new machinery
anywhere: an `sft` dataset here is already selectable, cappable, keyable and
mixable with a chat corpus, and the LLM engine needs no reader it does not have.
See [`../gyms/demonstrations.md`](../gyms/demonstrations.md).

**A `tool` message in one of these carries `tool` and `argument` beside its
`content`, and the content is the body alone** — unlike `convert-hermes`, whose
rows carry the whole rendered block. The difference is which side can render:
this service adds the header to neither. The backend renders the body with the
Rust function that serves a live round, and the engine's `render_result` adds
the `<|tool-call|>` header at build time, so no layout is spelled by anything
that cannot verify it.

`gym_dataset` finds or creates the set, `add_conversations` appends under a lock
on the dataset row — the batched sibling of `add_document`, holding the lock for
the same reason: `ordinal` is assigned from `document_count`, so two concurrent
writers without it would interleave two sets under the same numbers.
