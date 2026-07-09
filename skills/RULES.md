# Coding Rules

Rules the Auditor enforces against the codebase. These layer on top of the
project-specific conventions in `backend/CLAUDE.md` (Rust) and the frontend
conventions established in existing code.

Each rule has an id, the standard, and what a violation looks like. File issues
against the id (e.g. `rule: R1`).

---

## Backend (Rust)

### R1 — Bounded function length
A function should fit on one screen (~60 lines of body, excluding signature and
doc comment). Longer functions must be split into named stages.
*Violation:* a single function doing embed → attention → FF → loss → backward in
one body instead of delegating to named per-stage helpers.
*FFI caveat:* code dominated by one-argument-per-line `unsafe extern "C"` calls
(e.g. the CUDA kernel wrappers) is judged by **logical-step count, not raw line
count** — a single kernel invocation is ~10 lines, so a cohesive stage of 6–8
calls will exceed 60 lines while still doing exactly one thing. Split such a
function at real semantic boundaries (e.g. attention vs. feed-forward sublayer),
not to chase the line number; once each piece is one coherent operation it
satisfies R1 even if it runs long.

### R2 — Bounded loops
Every loop must have an obvious, finite bound. An unbounded `loop {}` is only
allowed with a clearly documented exit condition at the top of the loop.
*Violation:* a `loop` whose only exit is buried in the middle with no comment.

### R3 — Handle every fallible result
No silently dropped `Result`/`Option` in non-test code. No `.unwrap()`,
`.expect()`, `let _ =`, or `.ok()` on a fallible operation unless there's a
comment justifying why failure is impossible or ignorable.
*Violation:* `transformer.save(...)` result discarded with `let _ =` and no note;
`.unwrap()` on parsed external input.

### R4 — Contain and justify `unsafe`
Every `unsafe` block is as small as possible and carries a comment stating why it
is sound (invariants the caller guarantees). No `unsafe` spanning more than the
FFI call(s) it wraps.
*Violation:* a large `unsafe { ... }` covering pointer math plus several kernel
calls with no safety comment.

### R5 — Smallest scope for data
Declare variables at the narrowest scope that works; no wide `mut` state reused
across unrelated sections; no shared mutable module state without a documented
reason.
*Violation:* a `let mut` declared at the top of a long function and reused for
three different purposes.

### R6 — No duplicated logic
Repeated blocks (copy-pasted parsing, serialization, kernel-call boilerplate)
must be factored into a shared function. Two copies that must change together are
a bug waiting to happen.
*Violation:* `Transformer::load` and `Transformer::from_bytes` carrying nearly
identical read logic; `save_to_path` and `to_bytes` duplicated.

### R7 — Zero warnings
The build compiles clean. Every compiler/clippy warning is either fixed or has an
explicit, justified `#[allow(...)]` with a comment.
*Violation:* `dead_code` warning on `cuda_sgd_momentum` left standing.

### R8 — No magic numbers
Tunable or meaningful constants are named, not inlined. Hyperparameters, clip
thresholds, epsilons, buffer sizes get a `const` with a doc line.
*Violation:* `0.9`, `0.999`, `1e-8`, `1.0` clip norms scattered inline in the
optimizer/training loop.

### R9 — Documented public surface
Every `pub fn`, `pub struct`, and `pub` module has a doc comment saying what it
does and any non-obvious contract (units, invariants, panics).
*Violation:* a `pub fn` in the inference or training API with no `///`.

### R10 — Validate at boundaries
Functions that take external or cross-module input validate their assumptions
(lengths, shapes, ranges) instead of trusting them and indexing blindly.
*Violation:* indexing `input[0].len()` without checking `input` is non-empty; a
kernel wrapper trusting caller-supplied dimensions.

### R11 — No panics on external input
Parsing, IO, decoding, and request handling return `Result` and never `panic!`/
`unwrap` on data that came from a file, the network, or a client.
*Violation:* `.try_into().unwrap()` on bytes read from a model file.

### R12 — Respect the layering
Follow the strict layering in `backend/CLAUDE.md`: `routes/` is HTTP-only,
`db/` is pure SQL returning `Result`, `crypto/` is self-contained, no db struct
serialized straight to a client. The crate boundaries (`llm`, `tokenize`, …) are
one-directional.
*Violation:* a `db/` function returning `HttpResponse`; a route embedding raw SQL.

### R13 — Self-documenting through naming and structure
A function does exactly one thing and is named after the thing it does, so the
code reads without narration. Split any function that does more than one thing
into the smallest well-named pieces; if you can only describe what a function
does with an "and", it should be two functions. The **name carries the *what***
and the **structure carries the *flow*** — comments are reserved for *why*
(non-obvious rationale, invariants, gotchas), never to restate what a well-named
function or variable already says. A comment explaining *what* a block does is a
signal the block should be an extracted, named function instead.
*Violation:* a function named `process` / `handle` / `do_work` that does several
steps; a 40-line block prefaced by `// compute the attention scores` instead of a
`compute_attention_scores(...)` call; a name that lies about or under-describes
what the function actually does.
*Note:* this does not conflict with R9 — R9 wants a doc comment stating a public
function's *contract* (units, invariants, panics, the *why*), not a restatement
of its name.

---

## Frontend (TypeScript / React)

### R14 — Server-first rendering
Pages and layouts are server components by default. Only add `'use client'` when
browser APIs, React hooks (`useState`, `useEffect`, etc.), or DOM event handlers
are required.
*Violation:* a page that fetches static data at build time using `'use client'`
with a `useEffect` instead of an `async` server component.

### R15 — Centralized API and WebSocket layer
HTTP calls use `lib/api.ts` helpers (`authHeaders`, configured fetch wrapper).
WebSocket connections use the socket helpers in `lib/` (`chat-socket.ts`,
`training-socket.ts`). Don't inline raw `fetch()` or `new WebSocket()` in
components.
*Violation:* a chat component calling `fetch("/api/...")` directly instead of
going through `lib/api.ts`; a component creating its own WebSocket connection
from scratch.

### R16 — UI primitives from `components/ui/`
Use the prebuilt shadcn/ui components in `components/ui/` for buttons, inputs,
dialogs, cards, dropdowns, etc. Don't hand-roll equivalents with raw HTML and
ad-hoc Tailwind classes.
*Violation:* a `<button>` with custom Tailwind classes scattered across the
codebase when `components/ui/button.tsx` already exists; a custom dialog built
with `<div>` + state when `components/ui/alert-dialog.tsx` already exists.

### R17 — One component per file, explicit props
Each file exports one primary component. Props are defined as a dedicated
TypeScript `interface` or `type` (named, not inlined in the function signature
when more than 2–3 fields).
*Violation:* a file exporting two unrelated components; props inlined as
`(props: { a: string; b: number; c: boolean })` instead of a named interface.

### R18 — Strict TypeScript, no `any`
Every function and component is fully typed. `any` is forbidden unless a
comment justifies why the type is genuinely unreachable (rare — almost always
`unknown` + type guard is the right choice instead).
*Violation:* a prop typed as `any`; a `@ts-ignore` without a comment explaining
the limitation.
