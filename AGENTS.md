# Working in this repository

This system has six modules. Each owns its storage and domain behavior:

| Module | Owns |
| --- | --- |
| llm-engine | Models, tokenizers, training, model execution and durable scheduling |
| data-engine | Datasets, documents, acquisition and cleaning |
| bench-engine | Reference corpora, scoring and benchmark history |
| gyms-engine | Tasks, environments, episodes and grading |
| backend | Users, sessions, conversations, messages, attachments and service coordination |
| frontend | User interaction, presentation and browser state |

Requests flow frontend → Rust → service. Python services never call one another.
Shared Python libraries may share code, not service storage or network bypasses.

## Before changing code

1. Before making any code or documentation edits, open and read the complete
   contents of this file, docs/conventions.md and the affected module README(s)
   from the current checkout. This is required reading, not optional reference.
   An index, summary, partial excerpt or memory of an earlier version does not
   count. If a required file changes during the task, read the changed file again.
   Then read linked topics for the contract being changed and inspect adjacent
   callers and tests before choosing the implementation.
2. Check git status. Preserve unrelated and uncommitted work.
3. State the behavior/invariant being changed and the smallest useful scope.
   Do the requested work; report unrelated defects separately.

## While implementing

- Follow the conventions across Rust, Python and TypeScript.
- Keep domain calculations separate from transport/storage side effects.
- Prefer short, cohesive functions and explicit typed inputs/results.
  Extract semantic operations, not arbitrary blocks to satisfy a line count.
- Document each named function briefly at its declaration: purpose, inputs and
  outputs, plus relevant effects or errors. Put needed rationale/invariants there.
  Function bodies contain self-describing code, not comments.
- Give each mutable resource and lifecycle transition one owner.
- Validate external values, authorization and resource budgets at boundaries.
- Treat cancellation, retries, disconnects and restart as part of the behavior.
- Add abstractions for existing variation or a demonstrated test seam.
  Do not build speculative frameworks or generic helper collections.
- Keep applied migrations immutable. Query/schema changes require the relevant
  migrations, SQLx metadata regeneration and PostgreSQL contract checks.

## Before reporting completion

1. Run the checks relevant to changed behavior and its cross-module contracts.
   Use the documented verification entry point once it exists; until then use
   the module test commands and report every skipped/blocked check.
2. Review the diff for scope, duplication, naming and unnecessary indirection.
3. Update docs when ownership, schema, routes, protocols or public behavior move.
   Module READMEs are short maps; substantial topics live in linked siblings.
4. Report behavior changed, validation performed and remaining limitations.
   A passing build does not mean integration tests ran.
5. Do not commit or push unless asked.

## Documentation and commands

- [Module map and documentation requirements](docs/README.md).
- [Coding conventions](docs/conventions.md): read the complete file before edits.
- [Setup, native development and tests](README.md).
- [Architecture decisions](docs/decisions/README.md).
