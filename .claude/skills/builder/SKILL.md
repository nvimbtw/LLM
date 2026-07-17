---
name: builder
description: The only role that edits application code. Pick up a todo ISSUE/TODO from work/, implement the fix to the coding-rules standard, append a Builder comment, hand off to verification.
---

# Builder

Shared constraints (comment rules, ownership):
[`../shared/references/constraints.md`](../shared/references/constraints.md).
Artifact format: [`../shared/references/artifact-spec.md`](../shared/references/artifact-spec.md).
The standard you must satisfy: the `rules` skill —
load full text only for the rule ids involved, from
[`../rules/references/backend.md`](../rules/references/backend.md) /
[`../rules/references/frontend.md`](../rules/references/frontend.md).

## Process (for `work/<ID>.md`)

1. **Read cheaply**: frontmatter + `## Description` + the **newest** comment
   only (on a bounce, the latest `### Verifier · … · bounce` is your exact
   worklist — address *that*, not a re-guess). Open older comments only if the
   newest references them.
2. Set `status: in-progress`.
3. **Implement** at the cited location. Fix exactly the described problem —
   no scope creep; note unrelated problems in your comment for the Auditor
   instead of fixing them. Follow `backend/CLAUDE.md` / `frontend/CLAUDE.md`
   patterns; satisfy the cited rule without violating another.
4. **Verify locally**: `cargo build` (+ relevant tests) or
   `bun run lint`/`typecheck`/`build`. Zero warnings (R7) — the Verifier
   bounces anything less.
5. **Append** `### Builder · <YYYY-MM-DD>`: what changed, *why* (root cause —
   the repo owner is deepening their Rust understanding), and what you ran.
6. Set `status: verification`.

## The bar before hand-off

Problem resolved at the cited location · no rule violated · clean build +
passing tests · Builder comment explains the change and cites what ran.
