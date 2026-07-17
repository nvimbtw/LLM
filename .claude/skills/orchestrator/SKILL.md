---
name: orchestrator
description: Drive the work-resolution loop — pick the next work/ artifact, build it (as Builder), then shell out to the cheap opencode engine for Verifier and VCS, looping on bounces. Use when asked to work through open issues/todos or run the chain.
---

# Orchestrator

You drive the chain and you are also the **Builder** (see the `builder` skill —
the only role that edits code). Everything else runs on cheap opencode agents.
Cost split, constraints, and comment rules:
[`../shared/references/constraints.md`](../shared/references/constraints.md).
Artifact format/lifecycle:
[`../shared/references/artifact-spec.md`](../shared/references/artifact-spec.md).

## The driver loop

1. **Pick.** Frontmatter grep only — do not open whole files to triage:
   ```
   grep -l '^status: todo' work/ISSUE-*.md work/TODO-*.md
   ```
   Order: P0 → P3 (`grep '^priority:'`); within a priority, bounced first
   (`grep -l '· bounce'` among the todos). Skip anything whose `blocked-by`
   blocker isn't `done`. EPICs are never picked directly — work their children
   in dependency order; when all children are `done`, set the EPIC's
   `status: done` yourself and move on.
2. **Build.** Follow the `builder` skill: implement, append a `### Builder`
   comment, set `status: verification`.
3. **Verify (opencode).** Run in the background and **stay silent** — no
   narration, no status text, no file reads until the command exits:
   ```
   opencode run --agent verifier "Verify ISSUE-XXXX."
   ```
4. **Read the result cheaply.** The durable result is the file, not stdout.
   Read only the frontmatter and the newest comment:
   ```
   sed -n '1,15p' work/ISSUE-XXXX.md            # frontmatter
   awk '/^### /{h=NR} END{print h}' …           # or: tail -n 30 and take the last ### block
   ```
   Never read a long `## Comments` thread top to bottom. `done` → step 5.
   Bounced (`todo` + bounce comment) → back to step 2 with the bounce comment
   as the worklist. File unchanged / run failed → report it, don't advance.
5. **Commit (opencode).** Same background-and-silent rule:
   ```
   opencode run --agent vcs "Commit ISSUE-XXXX. Do not push unless I said to."
   ```
   Confirm via the newest `### VCS` comment.
6. **Next.** Repeat until nothing is `todo`/bounced or the user stops you.

## Decomposing feature work

When the user hands you feature work: small → file one `TODO-NNNN.md`
(next sequential id across all prefixes). Big / multi-step → file an
`EPIC-NNNN.md` tracker + ordered child TODOs per the epic section of
`artifact-spec.md`, then run the loop over the children.

## opencode notes

Agents are defined in `.opencode/agent/{auditor,verifier,vcs}.md` — each pins a
cheap model and is harness-locked to `work/**`; instructions live in
`.opencode/skill/`. The prompt only needs the artifact id. Override model with
`-m opencode/<model>`; list with `opencode models`.

Never do Verifier/VCS reasoning yourself, and never edit code outside the
Builder step.
