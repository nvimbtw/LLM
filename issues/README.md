# issues/

Single source of truth for outstanding work. One file per issue, named
`ISSUE-XXXX.md` (zero-padded to four digits, no slug so the name is stable across
title edits). Managed by the agent workflow in [`AGENTS.md`](../AGENTS.md): the
**Auditor** files issues, the **Builder/Fixer** resolves them, the **Verifier**
confirms, the **VCS** commits — orchestrated by the **Orchestrator**. Read this
format before adding or changing anything.

## Anatomy of an issue file

Machine-readable fields live in YAML frontmatter (so the Orchestrator can
`grep`/parse them); the human narrative and the cross-role dialogue live in the
body.

```markdown
---
id: ISSUE-0006
title: <short imperative title>
scope: <from scope legend>
priority: <P0|P1|P2|P3>
status: <todo|in-progress|verification|done>
rule: <Rn from skills/RULES.md, or none>
location: path/to/file.rs:line   # or :start-end; multiple space-separated
blocked-by: []                   # e.g. [ISSUE-0003]
blocks: []
---

## Description
<What's wrong, why it matters, and — if obvious — the shape of the fix.>

## Comments
### Builder · 2026-07-10
<what was changed and why>

### Verifier · 2026-07-10 · bounce
<precisely what is still wrong, so the Builder can act without re-investigating>
```

## The comment thread

The `## Comments` section is an append-only log — this is what lets the roles hold
a real back-and-forth across rounds instead of overwriting a single notes line.

- Every role that touches an issue **appends** a block headed
  `### <Role> · <YYYY-MM-DD>` with an optional `· <verdict>` suffix
  (`pass` / `bounce`). Roles: `Auditor`, `Builder`, `Verifier`, `VCS`.
- **Never edit or delete another role's comment.** Add a new one.
- The Builder explains *what changed and why*; the Verifier explains *pass* or
  *precisely why it bounced*; the VCS records the commit sha. This dialogue is the
  audit trail.
- Issues migrated from the old single-file `todo.md` carry a `migrated` marker in
  place of a date where the original timestamp wasn't recorded.

## Status lifecycle

```
        Auditor files            Builder starts        Builder done       Verifier passes
  (nothing) ─────────► todo ──────────────► in-progress ─────► verification ─────────► done
                        ▲                                            │
                        └──────────── Verifier bounces ─────────────┘
                                (appends a bounce comment with a reason)
```

- `todo` — filed, not started. **Owner of transition:** Auditor (create),
  Verifier (bounce back).
- `in-progress` — Builder is actively working it. **Owner:** Builder.
- `verification` — fix landed, awaiting check. **Owner:** Builder (sets it).
- `done` — verified resolved. **Owner:** Verifier.

`status` in the frontmatter is the **only** field a non-Auditor role overwrites.
Everything else in the frontmatter (id, scope, priority, rule, location,
blocked-by, blocks) belongs to the Auditor.

## Priority legend

- **P0** — critical: broken build, data loss, or a correctness bug that produces
  wrong output in normal use. Drop everything.
- **P1** — high: latent correctness bug (wrong output under some config), or a
  rule violation with real risk. Fix soon.
- **P2** — medium: maintainability/readability/duplication that isn't biting yet.
- **P3** — low: polish, nits, nice-to-haves.

## Scope legend

`frontend` · `backend` · `backend/llm` · `tokenizer` · `data-pipeline` ·
`crawler` · `infra` · `docs` · `cross` (spans several).

## Adding a new issue

Assign the next sequential id (highest existing `ISSUE-XXXX` + 1). Copy the
frontmatter template above, fill every field, set `status: todo`, cite the rule
id, and leave `## Comments` empty (or with the Auditor's filing rationale). One
file = one discrete problem at one location.
