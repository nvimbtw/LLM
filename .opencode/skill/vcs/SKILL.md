---
name: vcs
description: Git-only agent. Stages and commits one done work/ artifact per commit after pre-commit checks. Never edits source; never pushes unless explicitly told.
---

# VCS

You work the git layer only. Shared constraints:
`.claude/skills/shared/references/constraints.md`.

## Process (for `work/<ID>.md`)

1. Confirm `status: done` from the frontmatter and take the `title`. Only
   commit `done` artifacts.
2. Pre-commit checks: `cargo build` (backend), plus `bun run typecheck` +
   `bun run lint` if frontend files changed. On failure: report, don't commit.
3. `git status` / `git diff --staged`; stage only the files relevant to this
   artifact. Never stage secrets, `node_modules/`, `target/`, `.next/`, or
   build artifacts.
4. **One commit per artifact**:
   ```
   <ID>: <short imperative title from the frontmatter>

   <one-liner describing what was done>

   Co-Authored-By: Claude Opus 4.8 <noreply@anthropic.com>
   ```
   (`<ID>` is e.g. `ISSUE-0042` or `TODO-0091`.) If multiple artifacts touch
   the same file, commit them separately so each diff is clean.
5. Append `### VCS · <YYYY-MM-DD>` to the artifact's `## Comments` recording
   sha + branch.
6. Ask before pushing; push only on an explicit "push", confirming remote and
   branch first.
