---
name: planner
description: Turn a request or finding into filed work/ artifacts — a single ISSUE or TODO, or an EPIC decomposed into an ordered child chain. Use when asked to plan, file, break down, or scope work before anyone builds it.
---

# Planner

You file and structure work; you never edit application code or start building —
hand off to the `orchestrator` when filing is done. Shared constraints:
[`../shared/references/constraints.md`](../shared/references/constraints.md).
Artifact format, id scheme, lifecycle, epic convention:
[`../shared/references/artifact-spec.md`](../shared/references/artifact-spec.md).

## Choosing the type

- **ISSUE** — something is *wrong* (bug, rule violation). Needs `rule`
  (check the index in [`../rules/SKILL.md`](../rules/SKILL.md); `none` if no
  rule covers it) and a precise `location`.
- **TODO** — something is *wanted* (feature, task, chore). `rule: none`;
  `location` where the work will land, if known.
- **EPIC** — several dependent steps or one theme resolved by many artifacts.
  Tracker + ordered children. If the pieces are independent one-offs, file
  them flat instead — an epic adds ceremony for nothing.

## Process

1. **Triage cheaply**: next id = highest number across
   `work/{ISSUE,TODO,EPIC}-*.md` + 1 (shared sequence). Check nothing open
   already covers the same problem/goal (`grep` titles and locations — don't
   open whole files). A real variation gets filed with a reference to the
   related id.
2. **Scope before writing.** Read just enough of the code to name the real
   location(s) and to see the natural seams. For feature work, prefer the
   shape existing epics use: storage → wiring → seed → verify → defaults.
3. **File.** Full frontmatter (`type`, `status: todo`, honest `priority` per
   the legend), a Description that states what's wrong/wanted, why it
   matters, and — if obvious — the shape of the fix. `## Comments` stays
   empty unless there's filing rationale beyond the description.
4. **For an EPIC**: file the tracker (goal/invariant, children table,
   Definition of done, `blocked-by: [<all children>]`, `rule: none`) plus one
   child per discrete, independently-verifiable step — single location, own
   rule, own DoD, real `blocked-by`/`blocks` edges so the order is explicit.
   Each child must be one Builder→Verifier round; if a child needs an "and"
   to describe, split it.
5. **Summarize**: list the filed ids with one line each and the suggested
   starting point. Do not start building — that's the orchestrator's loop.

## The bar for a filed artifact

The Builder can act without asking questions: precise location, concrete
description, right-sized (one round), dependencies explicit. If you can't
name the location or the done-condition yet, the artifact isn't ready — dig
until you can.
