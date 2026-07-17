# Shared constraints (single source of truth)

Every role in this repo obeys these. Skill files reference this file instead of
restating it.

## 1. Only the Builder edits application code

The Orchestrator, Auditor, Verifier, and VCS are **read-only on all source**.
Their only writable surface is `work/` (artifact frontmatter `status` + appended
comments; the Auditor may create new artifact files). If a non-Builder role feels
the urge to fix something, it files or comments instead. For the opencode roles
this is also harness-enforced: `.opencode/agent/*.md` sets `permission.edit` to
deny everything except `work/**`.

Read-only roles may still run non-mutating commands: `cargo build`, `cargo
clippy`, `cargo test`, `bun run lint|typecheck|build`, `grep`/`rg`, reading
files. Never formatters, `--fix`, or anything that writes to tracked files
outside `work/`.

## 2. Comments are append-only

Every role that touches an artifact appends a block headed
`### <Role> · <YYYY-MM-DD>` (optional `· pass` / `· bounce` verdict) to its
`## Comments` section. **Never edit or delete another role's comment** — the
thread is the audit trail that lets roles hold a back-and-forth across rounds.

## 3. Field ownership

- `status` is the only frontmatter field a non-Auditor role overwrites, and only
  along the lifecycle transitions it owns (see `artifact-spec.md`).
- Everything else (id, type, title, scope, priority, rule, location,
  blocked-by, blocks) belongs to whoever filed the artifact.

## 4. The cost split

Claude Code (expensive) does only the work that edits code and drives the loop:
Orchestrator + Builder. opencode (cheap models) does the read-only review and
git roles: Auditor, Verifier, VCS. Never do a cheap role's reasoning on the
expensive engine "to save a round."
