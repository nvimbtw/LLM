# AGENTS.md

Router for agents working on this repository. Read this first, then load the
skill file for your assigned role.

## Roles

This project uses a three-role workflow. **You are told which role you are** at
the start of your session (e.g. "you are the auditor"). Find your role below and
read the linked skill file before doing anything else.

| If you are told you are the… | Read | You do |
| --- | --- | --- |
| **Auditor** | [`skills/AUDITOR.md`](skills/AUDITOR.md) | Scan the codebase against the coding rules and **file issues** into `todo.md`. Read-only on source. |
| **Verifier** | [`skills/VERIFIER.md`](skills/VERIFIER.md) | Check whether issues marked `verification` are actually resolved, then pass or bounce them. Read-only on source. |
| **Fixer** | [`skills/RULES.md`](skills/RULES.md) | The only role that edits application code. Picks up `todo` issues, implements the fix, moves them to `verification`. |
| **VCS** | [`skills/VCS.md`](skills/VCS.md) | Manages git — stage, commit, push. Read-only on source. Runs pre-commit checks. |

If you have **not** been told your role, you are the **Fixer** by default. Read
`skills/RULES.md` and `backend/CLAUDE.md` and start working.

## The one hard rule

**Only the Fixer edits application code.** The Auditor and Verifier are
**read-only on all source** — their sole writable file is `todo.md` (issues and
issue status). If you are the Auditor or Verifier and you feel the urge to fix
something, file it or note it instead. This separation is what keeps the cheap
audit pass safe to run broadly.

## The shared artifact: `todo.md`

`todo.md` is the single source of truth for outstanding work. Its header defines
the exact issue format, the status lifecycle, and the legends for priority and
scope. Every role reads and respects that format:

- **Auditor** appends new issues (status `todo`).
- **Fixer** moves an issue `todo → in-progress → verification` as it works.
- **Verifier** moves an issue `verification → done` (passed) or back to `todo`
  (bounced, with a reason in `verification-notes`).

Never rewrite or delete another role's issues. Only change the fields your role
owns (see each skill file).

## Project context

For how the code itself is organized and its conventions, see `backend/CLAUDE.md`
(backend, Rust) and `frontend/CLAUDE.md` (frontend, TypeScript/React). The
coding rules the Auditor enforces live in `skills/RULES.md` and are intended to
layer *on top of* those existing conventions, not replace them.
