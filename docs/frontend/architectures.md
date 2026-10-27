# Modular architecture frontend foundations

The frontend consumes llm-engine declarations through Rust. It owns presentation,
local drafts and request lifetimes. It does not execute Python, parse recipes to
decide validity, create its own queue or change source when parameters change.
The authoritative [versioned contracts](../llm-engine/architecture-contracts.md)
and [shared fixtures](../../contracts/modular-architecture/README.md) define the
wire boundary. Connected [administrator drafts](architecture-authoring.md) now
provide catalog, source/copy and optimistic metadata editing. Full authoring
validation/acceptance and training selection remain separate unfinished slices.
[Graph presentation](architecture-graphs.md) records the separate fixture-based
renderer and its actual-capture acceptance gate.

## Ownership and interfaces

- `lib/blocks.ts` and `lib/architectures.ts` expose catalog, detail, drafts,
  immutable revisions, optimistic save/metadata patch, validation and acceptance.
- `blocks.copySource` reads a typed `BlockCreate` wrapper for an exact builtin
  revision; normal creation gives the copy its own identity. It never copies the
  builtin implementation hash onto custom Python bytes.
- `lib/model-validation.ts` reads durable reports and compares exact current
  receipt identity. Observer cleanup never cancels a durable execution.
  `getArchitectureTokenizerBinding` reads the selected build's authoritative full
  fingerprint, vocabulary and optional reserved marker IDs through Rust.
- `lib/model-graph.ts` reads bounded lazy fragments and checks cursor identity.
  Graph queries include artifact receipt, view, mode, signature and scope.
- `lib/architecture/http.ts` uses the existing authenticated Rust transport with
  a captured bearer token and abort signal. HTTP statuses and bounded structured
  diagnostics survive forwarding; unstructured failure bodies retain status.
- `lib/architecture/contracts.generated.ts` contains structural TypeScript types
  generated from the shared schemas. The same generator writes
  `schemas.generated.json` for runtime validation inside the frontend build root;
  `--check` verifies byte equality with the authoritative shared schemas. There
  is no manually maintained frontend schema fork.
- `components/authoring/contracts.ts` defines graph/catalog/resource/report props
  for subsequent frontend lanes. Graph presentation belongs in its own directory;
  catalog/workspace and training/run work retain their distinct owners. Shared
  dependencies and these interfaces remain the frontend foundation owner's seam.
  Existing `lib/jobs.ts` changes belong to the training/execution lane.

Response decoding rejects unknown fields, invalid UUID/hash syntax, nonfinite
numbers, invalid declarations, unsupported capability claims, invalid report
stages, duplicate graph occurrences and unsafe source paths. Sources observe UTF-8
byte limits even when a JSON string contains multibyte text. Outgoing validation
preserves field omission, especially metadata patches where omitted description
must not become an explicit clear.

Receipts normalize schema defaults before comparing the exact subject,
dependencies, parameter/input digests, tokenizer, context and runtime profile.
Inspection/provisional receipts cannot authorize acceptance. A pending tokenizer
report may explicitly skip construction/capture/numerical stages while retaining
provisional metadata; only a valid report must pass capture and numerical checks.
The shared positive/negative fixtures enforce this distinction in both decoders.
Engine definition and package digests remain engine authoritative: Python's numeric JSON encoding
preserves `10000.0`, while browser JSON parsing/stringifying emits `10000`.
The frontend validates definition summary hash syntax; it does not recompute
package/definition hashes or claim to interpret the execution package. Shared
`DefinitionPackage`/`ResolvedDefinition` fixture models remain engine checks.
The API/SDK frontend harness runs receipt-currentness and graph-cursor contexts
in addition to positive/negative decoding.

## Identity and cancellation

`ResourceIdentity` includes the actual bearer session and stable resource key.
`DraftIdentity` adds saved version/hash; `ConfigurationIdentity` adds the local
configuration hash; `RunIdentity` names the durable execution separately.
Never use authentication booleans or display names as these identities.

`ArchitectureReadOwner` owns one resource/session's abort controller and generation.
A replacement read aborts the previous request and ignores its eventual result,
even if a transport ignores cancellation. Disposal ignores late values and errors.
`useArchitectureResource` creates/disposes that owner with the actual session,
kind and UUID, and masks prior-session values immediately during render.
Workspace callers must similarly dispose their draft/configuration owners on
identity changes and invalidate receipts as soon as source or values change.
Aborting an observer request does not stop a queued validation worker.

## Authoring controls

`components/authoring/source-editor.tsx` is the locally hosted Monaco surface,
accepting Python/YAML, an accessible name, value, editing policy and change callback.
It has no processor lifecycle or latest-active-callback state. The existing data
`PythonSourceEditor` wraps it with the original Python name/options. Monaco assets
continue to come from `/monaco/vs`; no editor CDN is required.

`ParameterFields` accepts engine declarations, explicit current values/errors and
one local change callback. Numeric blanks remain unset; integers and finite bounds
are checked by `parameterError`, without coercing booleans or strings. Enum indices
preserve the underlying scalar type. Unset booleans show an explicit choice rather
than claiming a false value. Callers own initial defaults, drafts, validation and
request effects. Controls use shared input/select/switch/error primitives.

`/markup/authoring` is a local fixture for both editable languages, a read-only
source and typed controls. It performs no catalog requests or mutations. Browser
checks use the actual Monaco surfaces and keyboard editing.

## Verification

From `frontend/`:

```sh
node scripts/generate-architecture-types.mjs --check
bun run typecheck
bun run lint
bun test
bun run build
```

Regenerate types after coordinated schema changes with the same script without
`--check`. Dev and production build entrypoints enforce freshness before compilation.
The fixture harness deliberately reads the shared API/SDK indexes;
changing generated types alone cannot change the interpreted schema.

For a browser check, start `bun run dev --port 3011` and run
`CHROMIUM_PATH=/usr/lib/chromium/chromium node scripts/check-authoring-browser.mjs`.
Omit `CHROMIUM_PATH` when Playwright browsers are installed. `AUTHORING_URL`
selects another frontend origin. The script also exercises the processor wrapper and real browser decoder acceptance/rejection.

Unit checks cover decoder fixtures, exact receipt/cursor contexts, read/session
replacement, aborted/disposed late completion, typed field bounds, omitted patches,
multibyte source and captured transport identity. Browser evidence and live backend
integration are separate; neither a mock nor a successful build certifies them.
