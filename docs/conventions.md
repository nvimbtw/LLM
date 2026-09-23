# Coding conventions

The standard every change in this repo is judged against. Ids (R1–R21) are
stable and safe to cite in a commit or a review; numbering is not contiguous.

Module-specific conventions — layering, stack, forbidden patterns — live in the
module's own doc: [`backend/README.md`](backend/README.md),
[`frontend/README.md`](frontend/README.md), [`llm-engine/README.md`](llm-engine/README.md).
These rules layer on top of those.

## Index

| Id | Rule |
|---|---|
| R1 | Bounded function length (~60-line body; FFI judged by logical steps) |
| R2 | Bounded loops — obvious, finite exit |
| R3 | Handle every fallible `Result`/`Option`; no bare `unwrap` |
| R4 | `unsafe` minimal and justified with a SAFETY comment |
| R5 | Smallest scope for data; no wide reused `mut` |
| R6 | No duplicated logic — factor shared blocks |
| R7 | Zero warnings (fix or justified `#[allow]`) |
| R8 | No magic numbers — named `const`s |
| R9 | Doc comments state non-obvious contracts only, never restate names |
| R10 | Validate at boundaries (lengths, shapes, ranges) |
| R11 | No panics on external input — return `Result` |
| R12 | Respect the layering (one-directional crates) |
| R13 | Self-documenting naming/structure; comments say *why*, never *what* |
| R14 | Server-first rendering; `'use client'` only when required |
| R15 | Centralized API/WebSocket layer (`lib/api.ts`, socket helpers) |
| R16 | UI primitives from `components/ui/`, no hand-rolled equivalents |
| R17 | One component per file; named props interface |
| R18 | Strict TypeScript, no `any` |
| R20 | Cohesive modules and honest crate boundaries; no grab-bag `utils` |
| R21 | Names don't stutter; every path segment adds meaning |

R1–R13 and R20–R21 are Rust; R14–R18 are TypeScript/React. R13 and R20–R21 are
about naming and structure and apply to the Python LLM engine too.

## Backend (Rust) — R1–R13, R20–R21


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

### R9 — Document contracts, not names
A `pub fn`, `pub struct`, or `pub` module carries a doc comment **only where the
name cannot carry the meaning by itself** — a non-obvious contract (units,
invariants, panics, ownership, ordering, side effects) or the *why* behind it. A
well-named public symbol whose behavior is obvious from its signature needs no
doc comment, and a doc comment that merely restates the name is itself a
violation (see R13). The bar is "what would surprise a caller," not "every symbol
gets a `///`."
*Violation:* `/// Widens the f16 view to f32.` on `pub fn to_f32_vec(&self) ->
Vec<f32>` (restates the name — delete it); **or**, conversely, a `pub fn
upload(&self, data: &[f32])` that panics on a length mismatch with no note that
it does. The first has a comment it shouldn't; the second lacks one it should.

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
Follow the strict layering in `docs/backend/README.md`: `routes/` is HTTP-only,
`db/` is pure SQL returning `Result`, `crypto/` is self-contained, no db struct
serialized straight to a client. The crate boundaries (`llm`, `tokenize`, …) are
one-directional.
*Violation:* a `db/` function returning `HttpResponse`; a route embedding raw SQL.

### R13 — Self-documenting code; comments say why, never what
A function does exactly one thing and is named after the thing it does, so the
code reads without narration. Split any function that does more than one thing
into the smallest well-named pieces; if you can only describe what a function
does with an "and", it should be two functions. The **name carries the *what***
and the **structure carries the *flow***.

Comments exist to say something the code **cannot**: the *why* (rationale for a
non-obvious choice), an invariant the types don't encode, a gotcha, a link to a
spec/issue. A comment that restates *what* the next line(s) do is noise — delete
it, or, if the block is only legible because of its caption, extract a named
function instead of captioning it. Prefer zero comments in a well-named body
over a comment per step.
*Violation:* a function named `process` / `handle` / `do_work` that does several
steps; a 40-line block prefaced by `// compute the attention scores` instead of a
`compute_attention_scores(...)` call; `// loop over layers` above
`for layer in 0..layers`; a `// --- attention ---` banner splitting a function
into sections that should be separate functions; a name that lies about or
under-describes what the function does.
*Not a violation:* `// bias correction is nonzero on step 1 even for a zero
gradient` (a *why* the code can't state); a `SAFETY:` note on `unsafe` (R4); a
contract doc on a public symbol (R9 — a *contract* is not a restatement of the
name).

### R19 — (folded into R13)
R19 ("comments earn their place: why, never what") duplicated the comment half
of R13 and was merged into it. Older references to R19 mean R13. Do not cite
R19 in new work.

### R20 — Cohesive modules and honest crate boundaries
A module or crate holds **one coherent concern**, named for that concern, and code
lives in the crate its dependencies imply — not wherever it was first typed. No
grab-bag `utils`/`misc`/`helpers` module accreting unrelated functions; no single
file mixing several concerns that each deserve their own module; no logic sitting
in a crate that has no business owning it (e.g. tokenizer logic inside `llm`,
HTTP concerns inside a compute crate). When a file grows to cover multiple
concerns, split it along those concerns. Crate dependencies stay one-directional
(this extends R12 from layering to crate placement).
*Violation:* a 300-line module spanning allocation, upload, save, and optimizer
plumbing that should be several modules; a `helpers.rs` holding three unrelated
functions; a type defined in a crate that only exists to be used by another crate
it could live in.

### R21 — Names don't stutter and the path adds meaning
A symbol's fully-qualified path should read without repetition. A type named the
same as its module (`transformer::Transformer`, `config::Config`) stutters — the
module segment adds nothing. Either give the type a name that says what it *is*
within its module (`transformer::Model`), or, if the module holds exactly one
public type, collapse the module. Every path segment must add information the next
one doesn't already carry. This applies to new and changed code, and existing
stutters are fair game to file.
*Violation:* `transformer::Transformer`; `workspace::Workspace`; a `mod tokenizer`
whose only export is `struct Tokenizer`; a field named `config.config`.

## Frontend (TypeScript / React) — R14–R18


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
