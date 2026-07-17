# AGENTS.md

Router for agents working on this repository. Roles are native skills; load
only yours.

## Roles and engines

Cost split: Claude Code (expensive) edits code and drives the loop; opencode
(cheap models) does read-only review and git.

| Role | Engine | Skill | Does |
| --- | --- | --- | --- |
| **Orchestrator** | Claude Code | `.claude/skills/orchestrator/` | Drives the chain: pick → build → shell out to opencode for verify/commit, loop on bounces. |
| **Builder** | Claude Code | `.claude/skills/builder/` | The **only** role that edits application code. |
| **Auditor** | opencode | `.opencode/skill/auditor/` | Scans code against the rules, files ISSUEs/EPICs into `work/`. |
| **Verifier** | opencode | `.opencode/skill/verifier/` | Passes (`done`) or bounces (`todo`) a `verification` artifact. |
| **VCS** | opencode | `.opencode/skill/vcs/` | Stages/commits one `done` artifact per commit; never pushes unasked. |

Claude Code default (no role stated): Orchestrator + Builder — invoke the
`orchestrator` skill and start working. The opencode roles are invoked as
`opencode run --agent <auditor|verifier|vcs> "<artifact id>"`; their agent
shims in `.opencode/agent/` pin a cheap model and harness-lock edits to
`work/**`.

## Shared ground truth (read on demand, defined once)

- **Constraints** — only the Builder edits code; append-only comments; field
  ownership; cost split: `.claude/skills/shared/references/constraints.md`
- **Work artifacts** — ISSUE/TODO/EPIC format, lifecycle, epics:
  `.claude/skills/shared/references/artifact-spec.md` (files live in `work/`)
- **Coding rules** — index: `.claude/skills/rules/SKILL.md`; full text per
  area in its `references/`. They layer on top of `backend/CLAUDE.md` and
  `frontend/CLAUDE.md`.
