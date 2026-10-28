# Administrator architecture drafts

Status: connected catalog, drafts, copies, metadata, declaration inspection,
explicit full validation/acceptance, pinned dependency snippets, paged immutable
history and retained graphs. Browser and live service acceptance remain open;
these pages do not yet satisfy ISS-65 acceptance.
The [shared foundations](architectures.md) own wire decoding, source editing,
parameter controls and session identity. The engine owns syntax and validity.

Typed engine readers now provide the installed prospective profile/device defaults
and safely parsed declarations for an exact saved YAML draft. The response remains
inspection-only. Block source inspection uses the supervised durable queue.

## Routes and authorization

`/arch` and `/arch/blocks` list bounded architecture/block metadata pages with
explicit search, origin and archived filters. Next page replaces the current
bounded list; First page clears its cursor. `/arch/new` and `/arch/blocks/new`
create drafts. Canonical creation links use `/arch/new-arch` and
`/arch/blocks/new-block`; the earlier `/new` routes remain aliases.
`/arch/{id}` and `/arch/blocks/{id}` load resource metadata, exact
saved draft bytes. History has an independent bounded, cursor-paged read owner;
selecting an old immutable source never replaces the editable draft.

`AdminBoundary` mounts private reads only after browser hydration establishes
the actual administrator session. Changing the bearer token remounts all private
state. Rust's administrator extractor and engine signed scope remain the actual
authorization boundary. Server page shells never read browser credentials.
The Workbench Architectures group links directly to Blocks and Architectures;
both catalogues return to Admin. Their shared open rows preserve wrapping names,
metadata and bounded paging without decorative separators. Validation rows link to
`/arch/validations/{run_id}`. Ordinary signed-in users retain normal model/chat pages.

## Draft and metadata ownership

`lib/architecture/draft.ts` owns one immutable published draft snapshot and one
optimistic save operation. It verifies exact UTF-8 text hashes, caps saved bytes
at 256 KiB and retains invalid Python/YAML. Saving captures one version/text;
an acknowledgement advances only those saved bytes. Newer text typed while the
request is pending stays dirty. A 409 preserves local bytes and refuses further
overwrite until an explicit server reload. The shared discard dialog makes loss
of local edits explicit. Disposal aborts requests and ignores late acknowledgements;
it does not claim an in-flight server write was rolled back.

Display name, description and archived status use their separate optimistic
metadata version. A confirmed metadata update replaces metadata alone and retains
the source document, including unsaved text. Builtins remain read-only. Archive
hides new selection and disables source editing; unarchiving retains the saved
draft and accepted revision identities.

Custom names, descriptions and archive choices are edited in the page header,
with the shared unsaved state and Save changes/Cancel controls. Metadata and exact
source bytes have independent [tab-local recovery](editing.md#draft-recovery).
Returning after navigation/reload restores the same session/resource's edits and
warns if the server baseline moved. Reload server draft discards source edits
only; metadata remains independently recoverable.

## Creation and copying

Creation saves exact entered source without implying inspection or acceptance.
Known rejected fields stay editable. Ambiguous transport failures retain source
and require checking the directory before another creation attempt, because the
creation API does not supply an idempotency key.

Copy links include an explicit resource and immutable revision UUID. Custom block
and architecture copies read that exact accepted source/YAML and verify identity
and byte hash; dependency references remain pinned. Builtin block copying uses
the engine's eligible wrapper endpoint for the exact installed revision, rather
than copying internal implementation files or manifest hashes as custom source.
Copying creates no resource until the separate Create draft action succeeds.

## Inspection, validation and acceptance

Saved Python is explicitly inspected through a queued declaration-only request,
without constructor/profile inputs. Saved YAML uses the bounded metadata reader.
Only that saved-source declaration supplies parameter controls. Optional defaults
populate local inputs; required values remain unset. Neither path rewrites source.

Full validation captures parameters, an explicit device/precision/full-prefix
shape domain, synthetic tensor inputs and actual final tokenizer facts. Unknown
symbolic tensor dimensions remain empty until supplied. Client checks cap each
shape entry at 256 characters before parsing, bound tensor elements and reject
malformed values; the engine owns compatibility, context,
dependency resolution, capture, numerical validity and resource admission.
Vocabulary metadata failure does not block declaration inspection. Vocabulary
search renders at most 200 matches plus the current selection.

`ValidationSubmission` owns one explicit queue operation. Unknown network outcomes
reuse the same request ID only for unchanged configuration, allowing server
deduplication; changed inputs get a fresh identity. Disposal aborts observation
and suppresses late acknowledgements without cancelling any committed execution.

Editor or configuration changes immediately mark prior results/graphs stale.
Only a matching fully valid report can build an acceptance request for exact saved
version/hash and the engine's dependency digest. Inspection, invalid and provisional
reports never enable acceptance. The engine adds discovered export guards to an
otherwise matching requested profile; supplied guard expectations must match exactly.
Acceptance refreshes resource metadata/history without discarding newer local text.

Dependency pickers show bounded unarchived catalogs and immutable revision pages.
The chosen revision remains pinned even when a new head appears. Copying a module
snippet is an explicit action; users insert its bindings/flow in YAML themselves.
No refresh, diagram or selection automatically adopts a newer dependency.

## Durable validation observation

`/arch/validations/{run_id}` and editor result panels use `ValidationObserver`.
It owns one actual session/run, serialized REST reconciliation, completed-read
polling and bounded stage replay (16 stages, 64 diagnostics). Socket reconnects
can replay old frames; sequence checks discard them. A terminal event triggers REST
reconciliation, and a queued row's embedded result never substitutes for the
verified publication endpoint. Observation errors require explicit retry.

The job page displays exact receipt source/bindings/profile, diagnostics and shared
lazy graph scopes. Completed-invalid, inspection-only, provisional and fully valid
results have distinct labels. A cancel acknowledgement remains nonterminal until
the service says execution stopped. Retry creates fresh job/request identities over
the original saved snapshot; ambiguous outcomes block duplicate retry pending queue
inspection. Delete removes terminal history only. Navigation/session cleanup releases
timers, reads and sockets without sending cancellation.

`JobsProvider` replaces private queue state on actual bearer/access changes and
ignores late completions/notices. Rust and signed engine scopes remain authoritative.

## Verification

Type checking and lint cover the page/hook boundaries. Focused draft/copy tests
exercise typing during saves, concurrent-version refusal, session disposal,
foreign acknowledgements, multibyte limits, immutable revision bytes, builtin
wrappers, stale acceptance, idempotent submission recovery, replay, explicit cancel
and detach-without-cancel. Fixtures and successful compilation do not certify browser
editing or actual frontend→Rust→engine acceptance.

Graph and validation-diagnostic source locations open verified retained bytes in
a read-only dialog, independently of current draft edits or catalog availability.
Actual live browser/service checks now cover custom block/recipe acceptance,
builtin copying, invalid draft retention, concurrent-save conflicts, stale
configuration during a running validation, archived dependencies, retained source
and ordinary-user refusal. Exact IDs, screenshots, test scopes and remaining
feature gates are in the
[acceptance evidence](../llm-engine/modular-architecture-evidence.md).
The production build requires network access for its existing Google font
downloads.
