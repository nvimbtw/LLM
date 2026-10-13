# Documentation

Every module in this repo has a directory under `docs/`. This page is the map,
and the spec for what goes in one.

## Map

| Directory | Module | Code |
| --- | --- | --- |
| [`llm-engine/`](llm-engine/) | Training, model execution and durable scheduling | `llm-engine/` |
| [`bench-engine/`](bench-engine/) | English benchmark scoring and performance history | `bench-engine/` |
| [`data-engine/`](data-engine/) | Acquiring datasets and cleaning them, before they are training data | `data-engine/` |
| [`backend/`](backend/) | People, sessions, and the seam to the Python services | `backend/` |
| [`frontend/`](frontend/) | Chat, job board, model and benchmark views | `frontend/` |
| [`gyms-engine/`](gyms-engine/) | Where a model is put to work: generated tasks, a sandboxed shell, and the score | `gyms-engine/` |
| [`decisions/`](decisions/) | Decision records — findings that cost real time | — |
| [`sprint-cleanup-validation.md`](sprint-cleanup-validation.md) | Deferred Sprint Cleanup acceptance and measurement plan | — |
| [`conventions.md`](conventions.md) | The active coding rules (C1–C12), language supplements and historical R-ID map | — |
| [`audits/`](audits/) | Dated audit findings and proposed agent standards | — |

## The shape of a module directory

```
docs/<module>/
  README.md      required — the entry point. Everything, or the map to it.
  schema.md      optional — tables, ownership, invariants
  api.md         optional — routes and their contracts
  <topic>.md     optional — one concern per file
```

`README.md` is the only required file, and the only one an agent is guaranteed
to read. Start with everything in it. **Split a topic into a sibling file once
`README.md` passes ~250 lines** — past that, every agent that needs one route
pays to load the whole module. When you split, the README keeps a one-line
summary and a link; it never stops being the complete map.

## What belongs in a module README

In this order:

1. **What the module owns** — and, as sharply, what it does not. Ownership
   boundaries are the thing agents get wrong.
2. **Tech stack** — with the *why* where a choice was contested.
3. **Layout** — directories/crates and the dependency direction between them.
4. **Schema** — tables, who writes them, the invariants that are not in the
   types.
5. **Routes / public surface** — the contract other modules code against.
6. **Conventions** — the patterns code in this module must follow, and the
   ones it must not.
7. **Gotchas** — the load-bearing details someone will otherwise re-break.

## What does not belong here

- **Decision records** → [`decisions/`](decisions/). A doc says what *is*; a
  decision record says what was weighed and rejected, once, with a date.
- **The coding rules (C1–C12)** → [`conventions.md`](conventions.md). Those
  apply across all modules; a module doc describes one module.
- **Anything derivable in ten seconds** from a file listing or a type
  signature. Prose that restates code is prose that will contradict code.

## Format

Markdown, always. No HTML source, no front-matter, no per-file ceremony. Docs
are read far more often by agents than by people, and markdown is what greps
cleanly, diffs reviewably, and costs the fewest tokens to load.

The frontend renders this tree at **`/docs`** (public; see
[`frontend/README.md`](frontend/README.md)), but that changes nothing about how
you write one: add a `.md` file and it becomes a page with a sidebar entry on
the next build. Two consequences worth knowing:

- **Relative links between docs are checked at build time.** A link to a file
  that does not exist fails `next build`. Write links relative to the file you
  are in — they work in an editor and on the site alike.
- **A docs-only change still needs a frontend rebuild** before it appears on
  the deployed site.

## Keeping them true

A doc nobody trusts is worse than no doc — it gets read once, found wrong, and
skipped forever after. So: **a change to a module's schema, routes, layering,
or ownership is not finished until this directory says so, in the same
change.** That is a rule with teeth in [`../AGENTS.md`](../AGENTS.md), not a
suggestion.

When you find a doc that lies, fix it in the change you are already making.
Do not file it for later.
