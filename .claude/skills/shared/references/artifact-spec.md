# Work artifact spec (single source of truth)

`work/` is the single source of truth for outstanding work — one Markdown file
per artifact, zero-padded four-digit id, no slug. Three types share one number
sequence:

| Type | File | Meaning | Requirements |
|---|---|---|---|
| ISSUE | `ISSUE-NNNN.md` | Bug / coding-rule violation | `rule` (Rn or `none`) + `location` |
| TODO | `TODO-NNNN.md` | Small task / feature work | no rule required (`rule: none`) |
| EPIC | `EPIC-NNNN.md` | Tracker for an ordered child chain — no code lives in it directly | `rule: none`, `blocked-by: [<all children>]`, `blocks: []` |

New ids: highest existing number across **all three prefixes** + 1.

## File anatomy

```markdown
---
id: ISSUE-0006            # matches the filename
type: issue               # issue | todo | epic — matches the prefix
title: <short imperative title>
scope: <frontend|backend|backend/llm|tokenizer|data-pipeline|crawler|infra|docs|cross>
priority: <P0|P1|P2|P3>
status: <todo|in-progress|verification|done>
rule: <Rn or none>        # rule ids live in .claude/skills/rules/SKILL.md
location: path/to/file.rs:line   # or :start-end; multiple space-separated
blocked-by: []            # e.g. [ISSUE-0003, TODO-0091]
blocks: []
---

## Description
<What's wrong / what's wanted, why it matters, and — if obvious — the shape of the fix.>

## Comments
### Builder · 2026-07-10
<what was changed and why>

### Verifier · 2026-07-10 · bounce
<precisely what is still wrong>
```

`## Comments` is append-only (see `constraints.md`). Roles: `Auditor`,
`Builder`, `Verifier`, `VCS`. Artifacts migrated from older systems may carry a
`migrated` marker instead of a date.

## Status lifecycle (all types)

```
filed ──► todo ──► in-progress ──► verification ──► done
            ▲                            │
            └──── Verifier bounces ──────┘
```

- `todo` — filed, not started. Transition owner: filer (create), Verifier (bounce).
- `in-progress` — Builder working it. Owner: Builder.
- `verification` — fix landed, awaiting check. Owner: Builder.
- `done` — verified resolved. Owner: Verifier. An EPIC flips to `done` when
  every `blocked-by` child is `done` (the Orchestrator checks and closes it —
  no Builder/Verifier round needed for the tracker itself).

## Priority legend

- **P0** — critical: broken build, data loss, wrong output in normal use.
- **P1** — high: latent correctness bug or a rule violation with real risk.
- **P2** — medium: maintainability/readability/duplication not biting yet.
- **P3** — low: polish, nits.

## Epics

Use an EPIC when several findings share a **dependency order** or a **single
theme** collectively resolved (a rename rippling across files, a module split
with dependent steps). Keep independent one-off findings flat — an epic adds
ceremony for nothing.

- **Tracker** (`EPIC-NNNN.md`): body states the shared goal/invariant, lists the
  children in a table, and gives a **Definition of done** (usually "all children
  done" + an end-to-end check). `blocked-by: [<every child>]`, `blocks: []`,
  `rule: none`.
- **Children** (ISSUEs or TODOs): one per discrete, independently-verifiable
  step — single `location`, own rule, own DoD — chained by real
  `blocked-by`/`blocks` edges so the order is explicit. Each child is one
  Builder→Verifier round.
- Tracker and children get sequential ids in one filing sweep. The Orchestrator
  picks children in dependency order and closes the tracker last.

## Who files what

- **Auditor** files ISSUEs (and an EPIC + child ISSUEs for ordered sweeps).
- **User / Orchestrator** file TODOs for feature work; the Orchestrator
  decomposes a big TODO into an EPIC + child TODOs.
- **Builder** consumes ISSUE and TODO identically; never works an EPIC directly.
