# Proposed agent rules — 2026-10-13

**Historical proposal, adopted 2026-10-13.** The author approved revision 2;
[AGENTS.md](../../AGENTS.md) and [the active conventions](../conventions.md) now
implement it. The draft and comparisons below preserve the review record. It accompanies the
[codebase audit](2026-10-13-codebase.md) and its
[Discord discussion](https://discord.com/channels/1538659898383011961/1559534257695891487).

**Revision 2 — incorporates the author's Discord replies:** agents must read the
entire coding-rules file, and functions get brief purpose/input/output
documentation instead of explanatory comments within their implementation.

The desired outcome is code a new contributor can change locally: short,
cohesive operations; explicit dependencies and state transitions; readable
contracts; and tests that protect behavior. Extensibility comes from reliable
seams around actual variation. It does not require predicting every future
feature or creating an interface for every function.

## What to retain and what to replace

| Existing rule | Proposed treatment | Reason |
| --- | --- | --- |
| Module ownership, directional dependencies, no Python service-to-service calls | Retain. | These are real system boundaries. |
| Read module docs, keep contracts current, no unsolicited commits/pushes | Retain and shorten; explicitly require opening and reading the entire coding-rules file before edits. | Reading an index, summary or remembered rules does not satisfy the requirement. |
| R1 function length | Apply to Python/TypeScript/Rust; make thresholds review signals. | The largest control flows currently fall outside its Rust-only scope. |
| R2 bounded loops | Replace with bounded work/resources and explicit cancellation. | A finite loop can allocate terabytes; a long-running event loop can be correct. |
| R3/R7 error handling and zero warnings | Retain, automate and cover all languages. | Comments are not substitutes for handling, logging or propagating failure. |
| R4 unsafe | Retain minimal unsafe; put soundness invariants in the function's documentation. | Keep the proof without scattering explanatory comments inside the body. |
| R5 minimal scope | Retain within cohesive ownership. | Small scope and one owner reduce hidden coupling. |
| R6 no duplication | Qualify by shared policy. | Similar-looking independent code need not share an abstraction; duplicated lifecycle policy does. |
| R8 named constants | Require names for policy, units and tuning; allow obvious local literals. | Naming every 0/1 adds noise and indirection. |
| R9/R13 comments and one thing | Require brief purpose/input/output documentation for every named function; no comments inside function bodies. Replace the “and” test with coherent operations. | Put contracts and necessary rationale at the function boundary; names and structure explain the implementation. |
| R10/R11 validation | Retain; include authorization and resource bounds. | Typed input is not automatically safe input. |
| R12/R20 layering/cohesion | Retain; add an application/use-case layer for substantial operations. | Routes should not become the default home for every cross-domain workflow. |
| R14 server-first | Retain where authentication/data access supports it. | localStorage bearer tokens cannot authenticate server rendering through authHeaders. Change session architecture before mandating authenticated server fetches. |
| R15 API/socket layer | Retain and extend to typed events and cancellation ownership. | Centralizing transport alone does not centralize its lifecycle. |
| R16 shared primitives | Retain consistency/accessibility goals; permit semantic HTML and justified exceptions. | A blanket HTML ban makes simple content unnecessarily indirect. |
| R17 one component/file | One primary concern per file; permit private cohesive subcomponents. | Splitting every tiny piece creates import navigation without reducing coupling. |
| R18 no any | Retain; validate unknown at runtime boundaries. | A cast can hide the same unsafety without using any. |
| R21 no module/type stutter | Replace with clarity at the call site and conventional domain names. | transformer::Transformer is often clearer than forcing a vague Model name. Renaming mechanically creates churn. |

Keep the historical R identifiers resolvable. If adopting new C identifiers,
retain a short compatibility table so old reviews and decisions still make sense.
Remove obsolete CUDA-FFI/crate examples from the current architecture guide.

## Draft replacement AGENTS.md

The following is intended as a complete, short entry point. Running/setup
details belong in the project README and module docs, linked from here.

```markdown
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
```

This draft names backend sessions as an intended responsibility; the audit's
revocation finding still needs an implementation. It deliberately does not claim
that a new verification command exists yet.

## Proposed conventions

These twelve rules replace the overlapping cross-language prose. Language
details follow them, instead of silently exempting Python or React.

### C1 · Organize by a coherent domain concern

Put code beside the behavior it owns. Keep dependency direction explicit and
keep transport/storage adapters outside domain calculations. A module should
have a small, understandable public surface. Avoid unrelated utils/helpers
collections and exports used only by an unrelated module.

### C2 · Keep functions small enough to change independently

A function performs one coherent operation at one abstraction level. Prefer
roughly 15–40 lines for ordinary logic; review bodies beyond roughly 60 lines,
more than three nesting levels, or many unrelated inputs. These are signals,
not automatic failure conditions. Declarative tables, cohesive numerical code
and generated bindings are assessed by logical complexity.

Extract when a block has a meaningful domain name, distinct failure policy,
independent test, repeated policy or separate lifetime. Keep trivial expressions
inline. A helper that needs ten closure variables or exists only as step_2 has
probably not found the right boundary. Use a small typed context when its
members share one real lifetime.

### C3 · Separate decisions from effects

Prefer pure functions for validation, scoring, transformation and state
reduction. Keep I/O and coordination in thin adapters/use cases that expose
the order of effects. A workflow may call several named stages; its purpose
remains one operation. Do not distribute a transaction's invariants among
unrelated helpers merely to shorten a body.

### C4 · Express contracts with types and boundary validation

Use typed request/result/event variants for interpreted payloads. Validate
unknown external data once and translate it into domain values. Keep opaque JSON
only for actual pass-through behavior or explicitly versioned extension fields.
Make units, identity and optional/unavailable values unambiguous. Avoid unchecked
casts, sentinel strings and positional arguments that can be exchanged silently.

### C5 · Give state and resources an owner

Each worker, socket, cache, timer, file, database transaction and mutable session
has a clear creator, lifetime and cleanup path. Keep state transitions explicit
and testable. Do not treat booleans such as authenticated as identity. Do not
maintain several writable copies of the same fact without a synchronization
contract. Dependency injection should expose actual dependencies, not hide
them in a universal context or global service locator.

### C6 · Design failure, retry and cancellation together

Specify what commits before acknowledging success and what remains after
partial failure. Retry reads or idempotent operations; use operation IDs where
effects may be repeated. A cancelled waiter is not a stopped worker. Persisted
state must reconcile after crashes. Surface actionable failures and preserve
their internal cause; silent catches require an explicit recovery policy and
appropriate observability.

### C7 · Bound work, memory and waiting

Validate effective resource use before allocating. Bound queues, caches,
pagination, concurrency, retries, input/output and admission waits. Use
backpressure or explicit resynchronization for slow consumers. Long-running
service loops are allowed when ownership, stop conditions and shutdown are
clear. Name policy/tuning constants with their units; leave obvious local
structural literals inline.

### C8 · Share policy when it must change together

Factor duplicated contracts, algorithms and lifecycle policy. Similar syntax
alone is insufficient reason to share code. Keep domain differences explicit;
avoid a generic implementation whose callers supply many mode flags. Before
adding a shared package, identify its stable inputs, outputs, dependency direction
and independently testable contract.

### C9 · Document functions; make their bodies self-describing

Every handwritten named function or method has brief function-level documentation,
including private helpers, hooks and React components. State its purpose, the
meaning of its inputs, and the output it returns or state it changes. Include
relevant preconditions, units, side effects and failure behavior. Usually one to
four sentences are enough; do not turn a simple function into a boilerplate
Args/Returns catalogue or merely repeat its name and type signature.

Use Rust `///` documentation and TypeScript/JavaScript JSDoc immediately above the
declaration. In Python, use the function's opening docstring as the language-native
equivalent. That docstring documents the function as a whole; it is not narration
interleaved with the implementation.

**No comments inside function bodies.** Names and structure must
describe the implementation. Move needed invariants, numerical reasoning, safety
assumptions and edge-case explanations into the function's documentation. If a
block has a separate coherent purpose, extract a named function and document it
there. Trivial anonymous callbacks do not need standalone documentation; a
substantial callback that needs explanation should become a documented named
function.

Remove step captions, banners, obvious paraphrases and obsolete implementation
history instead of moving that noise verbatim into a larger doc comment. Outside
functions, do not caption every import, variable, type or field; document only
information the declaration cannot convey. Put extended rationale/history in
decision records and regressions in tests. Use precise domain names, conventional
vocabulary and consistent units rather than mechanically short paths.

### C10 · Test externally observable contracts and failure transitions

Test invariants, boundary cases and meaningful failures rather than private
helper layout. Include cancellation/retry/restart and concurrency tests for
stateful workflows. Cross-language protocols require shared fixtures/schema
checks; PostgreSQL locking and migrations require PostgreSQL tests. Numerical
behavior changes require intentional evidence, including the pinned trace where
relevant. Cosmetic reversible changes need proportionate verification.

### C11 · Automate mechanical quality

Format with the language formatter and enforce agreed lint/type checks with
pinned versions. Builds and required checks must pass; explain specific blocked
checks without claiming success. Suppressions must be local, narrow and
justified. Baseline existing debt explicitly and prevent new violations; never
mass-format unrelated code as part of a feature.

### C12 · Extend from current requirements

Keep changes small and reviewable. Prefer composition and explicit seams around
real variation. Add a plugin system, generic repository or additional service only
when a present requirement benefits enough to justify its complexity. Preserve
compatibility deliberately with versions/adapters and migration tests. Delete
obsolete paths once their supported migration period ends.

## Language and module supplements

| Area | Additional rules |
| --- | --- |
| Rust | Document functions with `///`, including concrete safety invariants for unsafe operations. Keep unsafe minimal; function bodies have no comments. Handle fallible Results; no external-input panics. Use formatted, compile-checked SQL in the DB layer and regenerate SQLx metadata after query/schema changes. Run fmt/build/Clippy and relevant tests. |
| Python | Give named functions/methods short opening docstrings describing purpose, inputs and outputs; no comments within the implementation. Annotate public boundaries and domain contracts; use dataclasses/Protocols where they express real structure. Validate Pydantic ranges and effective budgets. Keep blocking I/O/CPU work off async executors; own and drain worker lifetimes. Add agreed formatter/linter/typechecker configuration across all four services. |
| React/TypeScript | Document named functions, hooks and components with short JSDoc above the declaration; no comments within their implementation. Use strict types and runtime validation at interpreted network boundaries. Keep event reducers pure; state updaters must not trigger network writes. Scope private state and requests to session/resource identity. One primary concern per file, with cohesive private components allowed. Reuse accessible primitives and centralized API/socket adapters. |
| Model code | Preserve tensor shapes, precision, RNG behavior, gradients and memory contracts when refactoring. Sampling probability, action provenance and training loss must agree or their deliberate approximation must be documented. |
| Documentation | Keep module READMEs near 150–250 lines as maps, split large topics, and keep public contracts current. Use markdown links that exist. Mark proposals and historical snapshots clearly. |

## Adoption plan

1. Agree on C1–C12 and the narrower AGENTS.md. Preserve links from old R IDs.
2. Add formatter/linter/typecheck configuration and a documented verification
   entry point; pin tool versions. Introduce a visible baseline for old debt.
3. Split frontend/backend README topics and correct obsolete ownership/route
   claims. Preserve useful operational gotchas through links.
4. Apply the rules in the audit's focused repair slices. Use concrete review
   examples from those changes as evidence that the guidance works.
5. Revisit thresholds after actual changes. Judge success by fewer coupled edits,
   clearer contracts and reliable tests, rather than by raw function/file counts.
