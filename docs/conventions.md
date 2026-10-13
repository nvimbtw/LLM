# Coding conventions

These rules apply to handwritten Rust, Python and TypeScript/React code in every
module. Read this entire file before editing; an index or summary is insufficient.
Module documentation adds domain-specific contracts without weakening these rules.
Existing code is brought into compliance in the scope being changed; generated
bindings and vendored code retain their upstream form.

## Rules

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

## Historical rule references

C1–C12 are the active rules. Earlier reviews and decision records may cite R IDs;
use this map to find their successor. The old wording is historical, not an
additional conflicting requirement.

| Historical ID | Active successor |
| --- | --- |
| R1 · Function length | C2 |
| R2 · Loops | C5, C6, C7 |
| R3 · Fallible results | C4, C6; Rust supplement |
| R4 · Unsafe | C9; Rust supplement |
| R5 · Scope | C2, C5 |
| R6 · Duplication | C8 |
| R7 · Warnings | C11 |
| R8 · Constants | C7, C9 |
| R9 · Function contracts | C9 |
| R10 · Boundary validation | C4, C7 |
| R11 · External-input panics | C4, C6; Rust supplement |
| R12 · Layering | C1, C3; module documentation |
| R13 · Cohesion and comments | C2, C9 |
| R14 · Server-first rendering | React/TypeScript supplement; frontend documentation |
| R15 · API/socket layer | C1, C5; frontend documentation |
| R16 · UI primitives | React/TypeScript supplement; frontend documentation |
| R17 · Components and props | C1, C4; React/TypeScript supplement |
| R18 · Strict TypeScript | C4; React/TypeScript supplement |
| R19 · Folded into R13 | C9 |
| R20 · Module cohesion | C1 |
| R21 · Naming | C9 |
