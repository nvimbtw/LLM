# Shared review workflow

The review and tester skills use this workflow. Follow repository instructions
and respect existing work. Reviews and testing do not modify app code or
implement fixes unless the user separately asks. Temporary test and measurement
artifacts are allowed outside the working tree and must be cleaned up.

## Scope and prerequisites

- Require an explicit scope. `full app` covers the application across its
  modules; partition it into manageable areas and report any coverage gaps.
  `diff` means the unstaged tracked changes shown by `git diff`. `uncommitted`
  covers staged and unstaged tracked changes plus untracked files. A user may
  supply a different explicit scope or Git comparison. Ask if it is ambiguous.
  Record the actual scope and Git state in the report. Read adjacent code and
  contracts as context, but identify findings outside the requested changes.
- Before starting the review, check that the `discord-collab` MCP tools for
  creating and reading a document are available and that the subagent tool can
  be used. If either is unavailable, stop and clearly tell the user to enable
  it; do not substitute a solo review or a local report.
- Use the current orchestrator model for every specialist, without silently
  switching providers or models. If delegation with that model cannot be
  established, stop and explain what must be configured. Do not claim model
  identity was verified if the environment does not expose it.

## Coordinate and verify

Choose enough focused specialists to cover the actual scope, avoiding duplicate
assignments. Give each the scope, applicable repository instructions, a
specific area, and the requirement to return concrete evidence, impact,
uncertainty and suggested remedies. Specialists report to the orchestrator;
only the orchestrator publishes. Await all specialists. If a required delegate
fails or cannot be started, stop and report the blocker; do not present an
incomplete review as complete.

Inspect the relevant code and validate candidate findings independently. Merge
duplicates, discard unsupported or purely stylistic speculation, and prioritize
by actual impact. Distinguish measured results from reasoned hypotheses. State
the conditions that trigger each finding and the limits of the evidence.

## Publish

Create one `discord-collab` item of kind `doc` for the review, with a descriptive
title and Markdown body. Include scope and revision/worktree state, areas and
specialists covered, prioritized findings with file and line references,
evidence, impact and practical ideas for remediation, plus checks performed,
results, skipped checks and coverage limitations. Explicitly say when no
actionable findings were found. Do not create separate issue threads unless
requested. Follow the MCP's current instructions for unique request IDs,
retries and delivery verification; read the created item to confirm its contents.
If publishing or verification fails, tell the user and do not claim success.
Return the document link or identifier and a concise summary to the user.
