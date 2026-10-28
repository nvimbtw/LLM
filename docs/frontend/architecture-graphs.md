# Architecture graph presentation

This is the ISS-64 renderer foundation, now connected to retained validation
reports through `SavedArchitectureGraph`. **Actual captured-graph browser
acceptance remains open.**
Synthetic fixture success does not certify capture completeness or live service
integration. The existing training illustration retains its legacy role.

## Data and ownership

`components/architecture-graph/graph-view.tsx` exports `ArchitectureGraphView` and
its props. Parents supply a current bounded fragment, its query, loading/error
state, stale flag and query/source callbacks. Queries identify exact receipt,
view, train/eval mode, serving signature, lazy scope and cursor. The renderer
never parses YAML/Python, captures a graph or changes execution semantics.

`lib/architecture-graph.ts` provides `layoutGraph` and `GraphScopeOwner`.
Layout consumes at most 256 nodes and 1024 edges. It positions the declared
nodes without unrolling loops or multiplying shared parameter ownership.
Edges crossing a page remain explicitly outside it, rather than inventing nodes.
The engine owns node kind, instance, execution occurrence, owner IDs, loop count,
shape, dtype, parameter values and optional unique-parameter measurements.

A graph owner belongs to one actual session/resource. Its loader can call
`getModelGraph` with the captured bearer token and provided AbortSignal. Subscribe
through `snapshot`/`subscribe`; call `load` for changed presentation queries and
`dispose` on session/resource/configuration replacement. A new load clears the
old fragment, aborts its request and ignores late values/errors. Returned queries
must match the requested profile/scope/page. Continued scopes/pages for the same
profile must retain the immutable graph artifact hash. Disposal only detaches
observation; it never cancels a durable validation worker.

`SavedArchitectureGraph` binds that owner to the actual bearer token, resource
and receipt identity. It reads one retained root after mounting and reuses the
owner across lazy scope/page changes. Session/resource/receipt replacement masks
the old fragment immediately and aborts its reads. Validation pages and saved-draft
result panels consume this component; stale editor results remain explicitly labeled.

Block validation reports open Captured operations and omit the Architecture
choice because their artifacts contain operation scopes only. Recipe reports and
saved models retain the Architecture view and can switch to captured operations.

Graph source actions and validation diagnostics open `RetainedSourceDialog`.
Its actual bearer, model/validation identity, revision and expected source hash
own a cancellable read through Rust. Switching that identity remounts the source
state; closing the dialog releases observation without cancelling durable work.
The adapter checks the selected revision, saved definition identity and optional
expected revision hash. Both shared decoders verify the exact UTF-8 byte digest,
relative display path and 256 KiB source budget. Read-only Python/YAML opens at
the selected source line. Frozen builtin manifests display as JSON with an
explicit explanation that package Python source is unavailable; they never
substitute current installed implementation files. Model source reads use the
package inventory, so imported models do not need live catalog entries.

Parents must mask mismatched identity snapshots during render and mark a retained
old receipt as stale immediately when editor configuration changes. A stale graph
is inspectable evidence for its named profile; it does not authorize acceptance.
The fixture demonstrates identity masking and fresh ownership in effect cleanup.

## Interaction and accessibility

The SVG is a bounded visual projection with pointer panning, keyboard arrow-key
panning, zoom buttons, plus/minus and zero/fit. A fit starts with the whole current
page. Zoom remains bounded while allowing inspection of a maximum-size page.
Visual nodes retain parameter owner IDs and distinct execution occurrences;
repeated computation does not imply independent weights.

The equivalent node list supports keyboard selection, child-scope opening and
bounded next-page navigation. The Connections disclosure lists exact named
source/target ports, including connections outside the visible page. Inspection
shows execution/instance identities, owner IDs, declared parameter values,
shape/dtype outputs and package-relative source locations. Missing counts remain
`Unavailable`. A source callback carries immutable revision and optional line.

View/mode changes reset scope and pagination; child-scope changes clear the old
cursor. Root navigation returns to the root scope. Loading and failure states use
the shared primitives. Theme roles style every graph surface, label and edge;
there is no graph animation, so reduced motion retains all interaction/state.

## Verification

`/markup/graph` uses labeled handcrafted fixtures with repeated shared owners,
an independent layer, nested child scopes, a second page, source links and absent
metrics. It has no catalog reads or durable mutation effects.

Run the ordinary frontend typecheck/lint/tests/build. With a frontend on port3011:

```sh
CHROMIUM_PATH=/usr/lib/chromium/chromium node scripts/check-architecture-graph-browser.mjs
```

`AUTHORING_URL` selects another frontend origin. Omit `CHROMIUM_PATH` for an
installed Playwright browser. The check covers keyboard selection and panning,
pointer panning, zoom/fit, inspection/source callbacks, nested scopes, pagination,
mode/view changes, light/dark themes and reduced motion. Synthetic screenshots
are written to `/tmp/llm-architecture-graph` or `GRAPH_SCREENSHOT_DIR`.

Focused unit checks cover shared versus independent owner identities, preserving
repeat metadata, absent metrics, page/layout budgets, out-of-page connections,
stale-scope cancellation, disposal/profile mismatch and immutable artifact changes.
Actual engine-produced operation and architecture graphs have also been rendered,
including startup blocks, nested recipe loops and the reusable two-output child
scope. Real graph/profile identities and inspected screenshots are recorded in
the [acceptance evidence](../llm-engine/modular-architecture-evidence.md).
These service observations remain separate from the synthetic fixture checks.
