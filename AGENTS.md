# AGENTS.md

Router for agents working on this repository. Read this first, then load the
skill file for your assigned role.

## Roles and engines

This project uses a multi-role workflow split across two engines to control cost.
The expensive engine (Claude Code) does only the work that edits code and drives
the loop; the cheap engine (opencode, cheap/free models) does the read-only
review and git roles.

| Role | Engine | Read | Does |
| --- | --- | --- | --- |
| **Orchestrator** | Claude Code | [`skills/ORCHESTRATOR.md`](skills/ORCHESTRATOR.md) | Drives the chain: picks the next issue, hands it to the Builder, then shells out to opencode for the Verifier and VCS, loops on bounces. |
| **Builder/Fixer** | Claude Code | [`skills/BUILDER.md`](skills/BUILDER.md) + [`skills/RULES.md`](skills/RULES.md) | The **only** role that edits application code. Picks up a `todo` issue, implements the fix, appends a comment, moves it to `verification`. |
| **Auditor** | opencode | [`skills/AUDITOR.md`](skills/AUDITOR.md) | Scans the codebase against the coding rules and **files issues** into `issues/`. Read-only on source. |
| **Verifier** | opencode | [`skills/VERIFIER.md`](skills/VERIFIER.md) | Checks whether `verification` issues are actually resolved, then passes (`done`) or bounces (`todo`) with a comment. Read-only on source. |
| **VCS** | opencode | [`skills/VCS.md`](skills/VCS.md) | Manages git — stage, commit, push. Read-only on source. Runs pre-commit checks. |

You are told which role you are at the start of your session. If you have **not**
been told, you are the **Orchestrator + Builder/Fixer** (the Claude Code default):
read `skills/ORCHESTRATOR.md`, `skills/BUILDER.md`, `skills/RULES.md`, and
`backend/CLAUDE.md`, and start working.

## The one hard rule

**Only the Builder/Fixer edits application code.** The Orchestrator, Auditor,
Verifier, and VCS are **read-only on all source** — their only writable surface is
the issue files under `issues/` (frontmatter `status` and appended comments). If a
non-Builder role feels the urge to fix something, it files or comments instead.
This separation is what keeps the cheap review passes safe to run broadly.

## The shared artifact: `issues/`

`issues/` is the single source of truth for outstanding work — **one Markdown file
per issue**, named `ISSUE-XXXX.md`. Its spec lives in
[`issues/README.md`](issues/README.md): the exact frontmatter, the status
lifecycle, the priority/scope legends, and — new — the **append-only comment
thread** that lets roles hold a real back-and-forth across rounds.

Related findings that share a theme or a dependency order are grouped into an
**epic** — a tracking issue `blocked-by` a strictly-ordered child chain — so the
Builder can later run the whole chain end-to-end. The Auditor files these; the
convention lives in [`issues/README.md`](issues/README.md).

Field/comment ownership:

- **Auditor** creates a new `ISSUE-XXXX.md` with `status: todo` and owns every
  frontmatter field.
- **Builder** moves an issue `todo → in-progress → verification` and appends a
  `### Builder` comment describing the fix. It never edits other frontmatter.
- **Verifier** moves an issue `verification → done` (pass) or `verification → todo`
  (bounce) and appends a `### Verifier` comment with the verdict/reason.
- **VCS** appends a `### VCS` comment recording the commit; edits no frontmatter.

Never edit or delete another role's comment — always append your own.

## The chain

The Orchestrator (Claude Code) runs this loop, one issue at a time:

```
  pick next todo issue (P0→P3, bounced first, respect blocked-by)
        │
        ▼
  Builder (Claude Code) edits code, appends ### Builder comment, status → verification
        │
        ▼
  Verifier (opencode)   opencode run --agent verifier "ISSUE-XXXX"
        │                 ├─ pass   → status done, ### Verifier comment
        │                 └─ bounce → status todo,  ### Verifier comment  ─┐
        ▼                                                                  │
  VCS (opencode)        opencode run --agent vcs "ISSUE-XXXX"             │
        │                 └─ commit, ### VCS comment                       │
        ▼                                                                  │
  next issue ◄──────────────────── loop back on bounce ────────────────────┘
```

The Orchestrator reads each opencode step's appended comment + resulting `status`
to decide the next move. Concrete invocation shapes and model selection live in
[`skills/ORCHESTRATOR.md`](skills/ORCHESTRATOR.md). The opencode roles are wired up
as thin agent shims in `.opencode/agent/{auditor,verifier,vcs}.md` — each pins a
cheap model and is **harness-locked to read-only on source** (`permission.edit`
allows only `issues/**`), deferring its actual instructions to the matching
`skills/*.md`.

## Project context

For how the code itself is organized and its conventions, see `backend/CLAUDE.md`
(backend, Rust) and `frontend/CLAUDE.md` (frontend, TypeScript/React). The coding
rules the Auditor enforces and the Builder must satisfy live in
[`skills/RULES.md`](skills/RULES.md) and layer *on top of* those existing
conventions, not replace them.
