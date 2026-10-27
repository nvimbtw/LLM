# Modular architecture implementation evidence

This ledger tracks FEAT-4 delivery gates. A contract/build pass is distinct from
real production integration. GPU support remains in scope; the user explicitly
deferred hardware verification to their other machine. Deferred GPU checks must
remain runnable and must never be reported as CPU-tested GPU evidence.

| Issue | Owner/outcome | Evidence/status |
| --- | --- | --- |
| ISS-48 | Capture/runtime feasibility | CPU scope passed. Independent runtime13/13 in43.190s; capture plus legacy numerical suites56/56 in44.21s. Later capture refinements36/36 plus baseline22/22 passed by capture owner. |
| ISS-49 | Versioned contracts | 488 combined contract/interface/runtime tests passed in3.58s;67 schemas regenerated and drift-checked; compile/doc/link/diff checks passed. |
| ISS-50 | SDK/resolver/core | Accepted independently on resume. Ambient dtype/device reads rejected; core/builtins/legacy numerical checks 80 passed, 3 CUDA skips. |
| ISS-51 | Catalog persistence | Accepted independently: 64 storage/migration/route/chain checks passed, including real PostgreSQL races and retained evidence. Root remains schema/migration owner. |
| ISS-52 | Supervised workers | Accepted CPU foundation: 16 original real-process cases passed; four added parent-death/restart/quarantine/flood/idle cases passed; real supervised runner success/cancel/startup-failure checks 3 passed. CUDA remains user-deferred. |
| ISS-53 | Builtins/recipes | Accepted independently: exact standard seeded weights/RNG/logits/gradients, shared-loop gradients, pinned trace; final builtin/copy suite 24 passed, 3 CUDA skips. |
| ISS-54 | Capture/validation | In progress; actual export guards now bind receipts. Focused CPU capture 17 passed, three deferred CUDA skips; current process/API gates remain open. |
| ISS-55 | Packages/checkpoints | In progress; grant-loading bridge, strict graph/evidence verification and USTAR transfer implemented. Final data-only matrix 34 passed; supervised numerical/grant cases remain open. |
| ISS-56 | Engine APIs/validation queue | In progress: durable validation/publication/acceptance/retry/recovery, nine builtin blocks/four recipe bootstrap, copy-source and authoritative tokenizer bindings implemented. 573 adapted API/contract checks, 58 capture/package/startup checks and subsequent 88-test bootstrap regression pass; real worker/network/PostgreSQL gates remain open. |
| ISS-57 | Memory/optimizer/diagnostics | Accepted library slice: 113 broad regression passes/three deferred CUDA skips; final focused 20 passed; capture after execution changes 17 passed/three CUDA skips. Final-profile candidates and shared loop integration belong58. |
| ISS-58 | Submission/pretrain/recovery | In progress: pinned queue/parent/child training and package adoption connected, conservative explicit occurrence replay and retained planning evidence implemented. Real CPU updates and trusted saved numerical recovery pass. Automatic memory-saving estimation, actual process/PostgreSQL acceptance and deferred CUDA gates remain; see dated entries below. |
| ISS-59 | SFT/token rows | In progress: saved-definition queue/worker/publication path, masks, capability warm-start, private tokenizer authority, reference diagnostics and serialized numerical recovery implemented. Actual process/PostgreSQL gates remain open. |
| ISS-60 | RL | In progress: eval-with-grad/ragged policy, parent-authorized gym effects, frozen reference, complete-iteration packages and exact numerical recovery are connected. Actual process/PostgreSQL gates remain open; optional caching is enabled only with complete supported evidence. |
| ISS-61 | Serving/scoring | In progress: saved-definition full-prefix worker, chat/scoring, cancellation ownership and occurrence-aware attention are connected. Real process/service/browser gates remain open. |
| ISS-62 | Rust | Implemented routes/signed visibility and independently passed genuine Rust→Python HTTP/SSE/WebSocket privacy. In progress until live56/58/61 gates. |
| ISS-63 | Frontend foundation | Accepted independently after browser schema correction: typecheck/lint, 321 Bun tests, generated schema freshness and actual Chromium decoder/editor/controls/processor-wrapper checks. |
| ISS-64 | Graph UI | In progress after accepted63; actual54 graph gate required. |
| ISS-65 | Authoring UI | In progress: connected draft/copy/metadata, declaration/full-validation/acceptance, explicit dependency snippets, paged immutable history and retained graphs. Live56/62/64/browser gates remain open. |
| ISS-66 | Training/model UI | In progress: immutable recipe selection, declared parameters, explicit engine profile preview, versioned presets, inherited saved definitions and effective chat context are connected. Submission/compatibility review and live execution/browser gates remain required. |
| ISS-67 | Validation/navigation UI | In progress: private validation execution route, queue/history/verdict labels, bounded observation and explicit lifecycle controls implemented by sole root owner. Browser/live filtering and cancellation gates remain open. |
| ISS-68 | Cross-module acceptance | Pending all delivery slices; explicit user-deferred GPU verification retained. |

Latest authoring/observation verification (24 October): frontend typecheck/lint and
all **391 tests across 37 files** pass. This includes stale acceptance, idempotent
submission recovery, foreign responses, bounded replay, cancel acknowledgement,
retry identity, observer disposal and captured-token socket cleanup.
Report: `/tmp/feat4-authoring-validation-complete-unit.log`. Tagged-union defaults
now normalize through `oneOf` as well as `anyOf`. Frontend receipt matching accepts
engine-discovered export guards while requiring supplied guards to match exactly.

A separate actual nested CPU validation produced **712 unique parameters, eight
strict guard records and 12 root architecture nodes**. Its generated run/report/
fragment pass frontend decoding, request/report coherence, verdict presentation and
static React graph rendering. Artifacts: `/tmp/feat4-real-validation-ui.json`,
`/tmp/feat4-real-validation-ui.html`; producer/consumer logs end in successful
checks. The initial producer script omitted five required nullable queue fields;
its DTO construction was corrected and the real capture reran. The first consumer
invocation passed but Bun emitted a tsconfig-override directory diagnostic; the
ordinary invocation reran cleanly. This is direct trusted numerical plus static
render evidence, **not supervised containment, browser interaction or a live
frontend→Rust→engine acceptance pass**.

Rust socket cancellation now carries the verified administrator UUID and signed
administration scope, matching REST cancellation. All-target Clippy and three
reader unit checks pass; actual forwarding/browser cleanup remains socket-blocked.
No schema/SQL changes occur in these control refinements. Frozen source navigation
is connected in the 25 October continuation below. No new issue
has been accepted; eight of 21 remain accepted and FEAT-4 stays active.

ISS-49 reproduction from `llm-engine/`:

```sh
../engine/.venv/bin/python -m llm.architecture.contracts --check
../engine/.venv/bin/python -m pytest tests/test_architecture_contracts.py tests/test_architecture_interfaces.py tests/test_architecture_runtime_contract.py
```

Local detailed reports: `/tmp/iss48-runtime-independent.log`,
`/tmp/iss48-capture-independent.xml`, `/tmp/iss49-contracts.xml`.
Capture/runtime commands and numeric/resource limitations are retained in their
[records](architecture-capture.md) and [runtime evidence](modular-runtime-feasibility.md).
ISS-49 treats eight `stage: resolution` fixtures as downstream expectations;
ISS-50 adds their production resolver/builder rejection evidence. No new catalog/routes/worker integration was claimed
by the contract slice. Shared contracts and policies are mapped in
[architecture-contracts.md](architecture-contracts.md).

PostgreSQL checks must own unique disposable database names; never reuse or drop
a fixed shared test database during parallel work. No commit or push is authorized.

ISS-51 reproduction uses the command in [storage verification](architecture-storage.md#verification).
The62-test run covers the complete persistence slice and existing migration/store/chain
behavior. The later40-test run rechecks changed store/route guards, including refusal
to retain evidence before validation completion and refusal to chain validation as a
model producer. JUnit reports: `/tmp/iss51-persistence.xml`,
`/tmp/iss51-final-store-routes.xml`. The coordinated49 fractional queue-rank and
blank-label corrections passed355 affected contract cases;67schemas were regenerated.
No Rust SQL/schema changes occur in51, so SQLx regeneration is not applicable.
Production worker/HTTP/publication integration is explicitly reserved for52/55/56/58.

The user resumed work on 22 October 2026. The earlier pause is historical;
50/51/52/53/63 have now passed independent acceptance. Development subsequently stopped at the user's request. All three GPT-6.1 Sol medium
subagents stopped and temporary test services were shut down. Work remains
uncommitted; no commit or push was made. See the
[resume handoff](modular-architecture-resume.md) for current ownership.

Resume reports: `/tmp/feat4-persistence.xml`,
`/tmp/feat4-builtins-independent.xml`, `/tmp/feat4-builtins-copy-independent.xml`,
`/tmp/feat4-runtime.xml`, `/tmp/feat4-runtime-new.xml`,
`/tmp/feat4-supervised-runner.xml`, `/tmp/feat4-visibility.xml` and
`/tmp/feat4-privacy-order-store.xml`. The initial expanded runtime report contains
one failure for an imprecise quarantine error; the admission error was corrected
and all four new cases reran successfully. Do not describe that earlier report
as passing.

The broad engine run was interrupted after detecting the unfinished validation
handler registration (ISS-56) and a new test fixture leaving the global queue
paused. Fixture cleanup is fixed and its combined 40-test order/route/store check
passes; a later complete engine run is still required. Rust workspace tests on
the default local server were blocked by its existing template1 collation mismatch;
verification moved to an isolated, fresh disposable cluster. The genuine privacy
integration test passes separately with `--ignored`; it uses real Python routes
and isolated storage, not the full catalog/training acceptance workflow.


Final stop checkpoint (22 October 2026): 7/21 issues accepted (about 35% by
estimated effort). Rust isolated workspace: 223 passed, one ignored; the real
Rust/Python privacy test passed separately, and workspace/all-targets Clippy with
warnings denied passed. Reports: `/tmp/feat4-rust-workspace-isolated-final.log`,
`/tmp/feat4-rust-clippy.log`. The isolated PostgreSQL cluster was stopped and
removed without changing the existing server's collation metadata.

Frontend final independent run: 321 passed, typecheck/lint and both Chromium
scripts passed. Earlier production build passed 121 pages; final build was
cancelled after compilation during TypeScript checking. ISS-64 browser evidence
uses synthetic fixtures, not actual ISS-54 publication.

ISS-54 independent targeted run: 15 passed in
`/tmp/feat4-validation-independent.xml`. Owner's expanded final run was stopped
with 10 passes and interrupt exit 2; actual export guard folding into the receipt
profile and publication integration remain incomplete. ISS-55 latest fixture
serialization correction and safety guards were not rerun; archive transfer is
untested and not integrated into loading. Neither issue is accepted. Resume from
the current handoff, preserving these gaps rather than treating partial suites
as full acceptance.

## Active continuation, 22 October 2026

The user resumed the feature and authorized regular commits on `modular-arch`,
prohibited sudo/Docker, and requested periodic main-feature-thread polling.
Those instructions supersede historical stop/no-commit notes. This sandbox mounts
`.git` read-only: attempted `git add` failed creating index.lock; no files were
staged or committed. Source changes are preserved.

Actual strict export mode/signature/scope/domain/guard records now bind returned
receipt profiles and graph manifest identity. Supplied guard mismatches fail.
Package adoption verifies the complete graph inventory and receipt linkage;
USTAR transfers have malformed archive/resource/crash checks. Private
`Load.package` grants have a worker-only loading bridge with definition/pin checks.
Tensor assignment refuses precision/shape/alias contradictions. These library
changes do not claim production training/serving or parent publication wiring.

Current reports:

- `/tmp/feat4-validation-offline.xml`: 17 passed, three GPU skips, two supervised
  cases deselected. `/tmp/feat4-validation-bounded-cpu.xml` repeats this result.
- `/tmp/feat4-package-capture-final.xml`: 50 passed, three GPU skips, six process
  cases deselected; includes actual graph inventory verification after guard binding.
- `/tmp/feat4-package-data-final.xml`: 34 passed, four process roundtrips
  deselected; includes latest tensor-profile and pre-deserialization guards.
- `/tmp/feat4-packages-offline.xml`: 30 legacy train-state checks passed; one
  HTTP-dependent test deselected. Package numerical cases failed/errorred at
  sandbox socket-buffer setup, before any model code ran. This report is not a
  passing package integration report.

The normal autouse HTTP fixture cannot create localhost TCP sockets here. Exact
copies of selected test files in `/tmp/feat4-offline-tests` used the original
isolated data/tokenizer fixtures without the HTTP fixture. Real supervised
tests are blocked at socket-buffer setup. Keep those gates open, along with
PostgreSQL/HTTP/browser service checks unavailable under current restrictions.
No sudo or Docker commands were used. GPU hardware remains user-deferred.

ISS-57 accepted after implementation/review of unique optimizer policy,
exact-export/meta/reference workspace planning, occurrence recomputation and
bounded optional diagnostics. Reports: `/tmp/feat4-training-support-regression.xml`
(113 passed, three CUDA skips), `/tmp/feat4-training-support-final.xml` (20 passed),
and `/tmp/feat4-capture-after-recompute.xml` (17 passed, three CUDA skips, two
sandbox-blocked process cases deselected). Contract freshness: all 67 schemas;
compile/function-doc/diff checks pass. The suite uses original repository test
functions via collection wrappers and isolated data fixtures without the global
HTTP fixture. The feature now has eight of 21 issues accepted. Production
loop/preflight candidate retention remains an ISS-58 handoff, not a claimed
working training flow. See [modular training support](modular-training-support.md).

ISS-56 groundwork: actual engine catalog metadata/draft/create/update/archive,
immutable revision reads and literal schema routes are registered. Administrative
scope/actor checks, typed errors, UTF-8 413 limits and filter/resource-scoped
keyset cursors are exercised against isolated SQLite via real ASGI handlers.
`/tmp/feat4-catalog-privacy.xml`: 22 passed, including existing visibility/SSE
privacy checks. These are in-process HTTP adapter checks, not Rust-network or
PostgreSQL race evidence. This groundwork has now been extended by the validation
integration described below; builtin seeding and real integration gates remain open.

The sandbox also denies Unix socket sends, suppressing asyncio executor wakeups.
The first ASGI run was interrupted; a bounded diagnostic retry timed out with
an idle worker and blocked selector stack. The focused harness now owns a 10 ms
asyncio timer that lets normal executor callbacks drain without changing sockets,
authorization, handlers or storage. The bounded timeout stack is retained in
`/tmp/feat4-catalog-routes.log`; the earlier interrupted run was superseded by
that diagnostic retry. Only the adapted run above is a passing claim.

Final archive review rejected non-block-aligned zero padding, closing a case
where a complete package payload could conceal a truncated trailer. The focused
transfer/publication/evidence regression now passes 22 tests in
`/tmp/feat4-transfer-trailer.xml`. All started verification processes completed
or were explicitly interrupted before continuation; no test services were left
running. Git writes remain unavailable under the current read-only mount.

## Validation queue and builtin block continuation

ISS-56 now freezes exact source/dependency bytes into normal `model_validation`
jobs, verifies cleanup before immutable staged/fsynced publication, and commits
terminal done plus retained valid evidence together. Report/graph reads and
exact-draft acceptance are registered. Retry preserves pinned bytes with fresh
request/job UUIDs; cancellation discards the report and cannot retain evidence.
Recovery adopts only a complete matching parent publication after exclusive
worker reconciliation. Pending tokenizer results remain provisional without
constructing placeholder weights or claiming capture/numerical evidence.

Copied builtin imports are detected structurally, including spaced Python import
syntax, and pin installed compatibility before import, acceptance and package
loading. Startup seeds nine read-only builtin block sources and idempotent normal
CPU validation jobs without constructing models. Valid published output accepts
their exact manifest-derived revision in the same terminal/evidence transaction.
Readable builtin source and its complete implementation digest remain distinct;
copy-source returns eligible custom wrapper bytes under their own future identity.
Builtin architecture template seeding remains unfinished.

Final tokenizer facts now require complete owned bytes/sidecars, full SHA-256,
contiguous actual vocabulary and requested reserved marker IDs. Browser-supplied
hash/size/ID or derived-provenance guesses cannot authorize final bindings. The
admin binding read and builtin copy-source read are forwarded through Rust and
available through the shared typed frontend adapters. No database schema or Rust
SQL changed; SQLx metadata regeneration does not apply.

Current evidence (adapted offline harness, never network/process/PG proof):

- `/tmp/feat4-validation-builtin-api-final.xml`: 573 passed, one supervised HTTP
  test deselected. Includes actual signed ASGI routes, isolated SQLite storage,
  shared contracts, ordinary run/chain regression, nine real numerical builtin
  fixture publications and source/copy/retained-evidence checks.
- `/tmp/feat4-validation-capture-package-final.xml`: 58 passed, three explicitly
  user-deferred CUDA skips, seven supervised/process cases deselected. Includes
  complete CPU captures, package/transfer checks and seven startup failure checks.
- `/tmp/feat4-tokenizer-authority.xml`: final focused tokenizer HTTP authority
  check passed after reserved-name query support and malformed artifact checks.
- Frontend: 203 shared decoder checks, typecheck, lint and generated-schema
  freshness passed. This continuation did not run actual browser interaction or
  a production build.
- `/tmp/feat4-rust-builtin-authorization.log`: focused Rust private-route
  authorization passed. `/tmp/feat4-rust-validation-clippy.log`: offline API
  all-targets Clippy with warnings denied passed. Scoped rustfmt passed. The
  network forwarding/WebSocket cases were not rerun under this sandbox.
- All 67 shared schemas are fresh; targeted compilation, function documentation
  audit and diff whitespace checks pass.

`/tmp/feat4-validation-real-worker.xml` is a failed/blocked integration attempt:
the production HTTP-to-worker path hits denied socket buffer configuration before
child launch and returns `validation_failed`. It is not an acceptance pass.
The initial extended package selection also included a blocked real grant worker
case (`/tmp/feat4-validation-pin-regression.xml`: 52 passes, one permission
failure); the explicit data-only rerun passed 53 with six process cases excluded.
The earlier 512-pass combined run had a teardown error when the test fixture
reopened the global queue before its drain stopped. Pause restoration now belongs
to the outer fixture after drain shutdown; the combined reruns pass. An intermediate
559-pass run rejected an old copied-source fixture without mandatory builtin
pins; the fixture now freezes the same genuine pins as production, and the final
573-test run passes. These earlier reports remain nonpassing records.

ISS-54/55/56 stay open. Remaining gates include builtin architecture examples,
actual supervised validation/package publication, PostgreSQL cancel/accept/archive
races and the later training/serving/frontend acceptance flows. Source and valid
receipt retention remain independent of prunable jobs. The feature is still eight
of 21 accepted issues. No sudo/Docker or Git writes were performed; `.git` remains
read-only. All verification started for this continuation reached terminal status.

## Builtin recipe readiness continuation

`corpus/baseline.py` constructs a real deterministic no-merge byte-level BPE
tokenizer with the engine's reserved markers and every byte. It learns no corpus
and makes no dataset claim. `api/builtin_tokenizer.py` publishes its exact bytes
and complete build marker with staging/fsync/no-replace rename before indexing.
Complete orphans survive index failure for adoption; conflicting artifacts fail;
existing display edits remain. Cross-process identity probes produced the same
full fingerprint and UUID. The artifact is implicit and never substitutes for a
pending training binding.

All four supplied recipes now seed as read-only exact YAML. Their ordinary
`model_validation` checks queue only after required immutable block revisions
exist. Live terminal completion and startup adoption schedule newly ready checks
inside their caller-owned transaction. A complete valid recipe publication
retains evidence and accepts the exact source/pins. Bootstrap IDs bind draft
version/profile/tokenizer; repeated startup does not duplicate jobs, and a code
rollback to previously accepted exact YAML reuses its immutable revision. The
shared block/recipe root seeding transition keeps independent metadata/draft
versions and refuses foreign custom resources under reserved supplied IDs.

Reports:

- `/tmp/feat4-builtin-recipes-api.xml`: 88 passed, two supervised cases
  deselected, covering new recipe/tokenizer bootstrap and existing block,
  validation, visibility, catalog, run-route and chain adapters.
- `/tmp/feat4-builtin-recipes-final.xml`: six focused cases passed, one
  supervised bootstrap case deselected. Includes all four actual CPU numerical
  recipe fixture publications, empty-install waiting, real tokenizer byte/Unicode
  round trips, index/write interruption, conflicting bytes, preserved display
  edits, pending training pins, restored older source and custom identity refusal.
- All 67 shared schemas remain fresh; targeted compile/function documentation/
  diff checks pass. No shared wire schema or database schema changed, and there
  are no Rust SQL changes requiring SQLx regeneration.

These use the previously described adapted original-test harness with isolated
SQLite and a finite loop polling timer. Numerical fixture publication exercises
real arithmetic and parent adapters; it does not prove process cleanup or
PostgreSQL locking. A new `test_supervised_builtin_bootstrap_uses_real_parent_queue_and_child_receipts`
keeps the complete 13-job production process check runnable. It is not claimed
passing under the known socket restrictions. GPU hardware remains user-deferred.
The feature still has eight of 21 accepted issues. Next implementation is ISS-58;
ISS-54/55/56 remain open for actual worker/network/PostgreSQL gates.

## Shared training loop continuation

ISS-58 is in progress. The existing update loop now supports declared language
model facts/logits alongside its deliberate legacy Transformer adapter. Recipe
context is authoritative. Captured memory plans retain exact user input shape
bounds, and the loop refuses missing/mismatched model bytes or full-prefix
batch/context admission before creating the optimizer. Shared Parameter ownership
and actual component temperature policy drive optimizer groups; generic models
never enter the measured width/layer memory planner or receive an invented
recompute depth.

The bounded generic diagnostics adapter reports actual semantic owners and
occurrences and preserves the numerical trajectory. No unsupported attention or
embedding metrics are invented. Existing schedules, masks, checkpoint cadence,
cancellation and next-batch cursor behavior remain in the same shared loop.

Evidence:

- `/tmp/feat4-shared-loop-final.xml`: 65 passed, one HTTP-dependent legacy
  orchestrator case deselected because it requires the unavailable dataset
  service socket. Covers actual standard/looped/nested recipe updates, standard
  versus legacy exact parameter/loss parity, legacy state/memory behavior,
  diagnostics and dropout optimizer/RNG/cursor recovery.
- `/tmp/feat4-shared-loop-occurrences.xml`: all 10 final focused shared-loop
  cases passed after adding preservation of explicit occurrence recomputation
  through real optimizer updates. This includes the nine cases from the broader
  regression and one added case; counts are not additive.
- `/tmp/feat4-shared-loop-legacy.xml` remains nonpassing: 36 tests passed and
  the dataset-dependent resumed-orchestrator case failed at denied service I/O.
  The explicit final selection above excludes that blocked check.
- Function documentation/body-comment audit, targeted compilation, current
  shared schema freshness and diff whitespace checks pass.

These are original repository test functions in the adapted offline fixture
harness. Exact recovery is proved at the loop boundary using captured in-memory
weights/optimizer/RNG/cursor; saved-package serialization, supervised execution,
parent publication/adoption and actual process restart remain separate gates.
The recipe HTTP training path is not wired yet. Next: typed selection/config,
atomic complete-chain enqueue/pins/retry, parent-only corpus/tokenizer preparation,
child final preflight/model/optimizer lifetime and parent package/checkpoint
publication. GPU support remains required, with hardware checks user-deferred.

## Atomic chain submission continuation

The existing `/runs/chain` now validates every effective pretrain/SFT/RL/benchmark
configuration before inserting any queue row. It keeps stated values sparse,
checks the initial pretrain tokenizer, reserves all model-producing output UUIDs
and validates waiting consumers against their predecessor's reserved UUID without
persisting a fabricated input. Chains are bounded to 100 steps. A report-producing
benchmark must be last. All queue/domain rows commit in one caller transaction;
only then does the route wake the runner once. No new scheduler or status owner
was introduced. Recipe source/evidence pins must join this transaction in the
pending modular submission adapter; no recipe HTTP execution is claimed yet.

- `/tmp/feat4-chain-atomic.xml`: 37 original route/lifecycle checks passed in
  6.22 seconds through the adapted offline fixtures. New cases prove malformed
  late steps insert nothing, a late failure after a real first insert rolls back
  that job, and the single wake sees the full committed linked chain through a
  separate connection. Sparse resume/config behavior and waiting promotion pass.
- `/tmp/feat4-chain-postgres.xml`: two new PostgreSQL route-transaction checks
  cannot run because both fixtures fail connecting to the local PostgreSQL socket
  with `Operation not permitted`. They remain runnable checks for complete-commit
  visibility and a real second-insert constraint violation. No PostgreSQL success
  is claimed. Existing historical PostgreSQL storage evidence is unchanged.

No schema or Rust SQL changed, so no migration or SQLx regeneration applies.
GPU checks remain user-deferred. Commits are authorized, but `.git` is mounted
read-only in this session and the earlier staging attempt was denied.

Tracing the tokenizer handoff also found a contract mismatch: the resolver
removed `_id` from `${tokenizer.pad_id}`, while the authoritative binding API and
shared fixtures retain `reserved_token_ids.pad_id`. Resolution now reads the full
declared key. A regression builds real tokenizer bytes, resolves their actual
reserved ID into a custom constructor and checks the constructed model output;
missing final reserved facts still fail. No token IDs or tokenizer bytes change.

`/tmp/feat4-chain-tokenizer.xml`: 101 checks passed in 17.28 seconds across the
original core, validation-queue, run-route and chain tests. One actual supervised
validation case was explicitly excluded under the known socket restriction.
This supersedes the smaller chain check for affected behavior, with overlapping
counts. Compilation and diff whitespace checks pass. It does not replace the
two blocked PostgreSQL checks or establish process/GPU integration.

## Pinned training submission continuation

`TrainConfig.architecture` now uses the shared typed `ArchitectureSelection`.
Selected recipes reject every explicitly supplied flat structural field and
cannot accompany `resume_from`; SFT/RL reject structural overrides instead of
silently discarding them. The legacy flat adapter remains deliberate, including
positive-head validation before division. Legacy `orchestrate.train.run` refuses
selected recipes before opening an authenticated data client or building a model.

`api/training_jobs.py` resolves accepted recipe defaults/types/context without
source imports. It reads actual owned tokenizer facts, including required full
reserved-ID names, or pins the originating training job as the pending tokenizer
operation. Exact prior receipts must match source/dependencies/parameters/final
binding/context and effective runtime/input domain. `train/model_preflight.py`
provides that shared CPU/CUDA/precision/TF32/seed/batch/context declaration. Final
in-job preflight remains required under actual worker admission in every case.

Storage preparation now freezes sources/declarations/block compatibility/origin
and evidence while retaining root/evidence locks. The serialized snapshot caps
at 8 MiB. Chains prepare every selection before their first queue insert and
attach inherited pins to waiting SFT/RL jobs within the same transaction.
Actor/request outcomes retain every original chain job UUID. Replays never
recreate pruned work or return a partial chain. Retry/recovery copy original pins
after draft edits/archive, reserve fresh output UUIDs and can replay a completed
operation even after pruning its original input job. Recovery metadata names the
latest registered generation; no checkpoint tensors are loaded by these checks.

The Rust `/runs/train`, `/runs/chain`, retry and recover adapters now forward
query values such as request_id, sign actor plus administrative scope, and retain
engine status/body under the existing response-byte bound. Engine-owned recipe
JSON stays opaque. This closes a transport gap that otherwise made selected
submissions unusable through Rust even though direct engine admission worked.

Evidence (counts overlap):

- `/tmp/feat4-pinned-submit-regression.xml`: 119 passed in 48.56 seconds across
  submission, routes, chains, validation queue, shared training loop and legacy
  state tests. The actual supervised validation and dataset-service-dependent
  legacy orchestrator cases were explicitly deselected under socket restrictions.
- `/tmp/feat4-pinned-submit-contract-final.xml`: 512 passed in 7.99 seconds,
  including the 22 new submission cases and current shared contracts/interfaces/
  runtime declarations after source compatibility/origin snapshot retention.
- `/tmp/feat4-pinned-submit-retention.xml`: final 59 submission/route/chain checks
  passed in 9.41 seconds after adding retry/recovery replay following original
  input-job pruning. Earlier intermediate reports with fixture expectation errors
  are not passing final evidence.
- `/tmp/feat4-pinned-submit-postgres.xml`: six required checks failed at fixture
  connection to the denied local PostgreSQL socket, before any assertions. These
  include archive/enqueue, late-chain rollback, commit visibility, real constraint
  failure, inherited recipe-chain idempotency and concurrent actor/request dedupe.
  They remain runnable; historical PostgreSQL passes do not certify these edits.
- `/tmp/feat4-pinned-rust-auth.log`: one actual authorization check passed,
  expanded to cover selected submission/chain/retry/recovery routes.
- `/tmp/feat4-pinned-rust-transport.log`: the new full forwarding test compiled
  but failed creating its TCP test server with Operation not permitted. It did
  not verify transport assertions and is not a pass.
- Targeted Python compilation, 67-schema freshness, documentation/body-comment
  audit for 50 changed/new functions, scoped Rust formatting and diff whitespace
  checks pass. `/tmp/feat4-pinned-rust-clippy.log`: strict API all-targets Clippy
  passed with offline SQLx/Cargo and no warnings.

**Open execution gate:** default production handlers still refuse any pinned
recipe before entering legacy execution. The actual supervised adapter must
prepare authorized data in the parent, finalize bindings and capture/admit in the
child, train with the shared loop, and publish/adopt immutable packages and
checkpoints. Removing the refusal without that adapter would violate the task.
These data-only accepted receipt/source fixtures establish admission behavior,
not numerical validation, source containment or saved/process recovery. No new
issue is accepted. GPU hardware remains user-deferred; commits remain blocked by
the read-only .git mount. No migration or Rust SQL changed, so SQLx regeneration
does not apply.

## Child pretraining and parent package-adoption continuation

The actual child pretrain backend now owns final source/model/optimizer execution.
Parent-only data preparation retains caller attribution and recipe context,
acquires through the existing Rust hop, and grants actual tokenizer/corpus/spans.
Typed preparation checks sparse config, request/revision/parameter identity and
final bindings. Meta capture/memory admission precedes real fp32 master weights;
final train/eval strict capture/numerical evidence and full-prefix memory planning
precede AdamW. Effective bf16 fallback is recorded as fp16. SourceLoader survives
through training and package publication. Real consumed batches drive liveness;
bounded acknowledged telemetry reports measured semantic diagnostics.

Recovery/final packages include actual weights, optimizer/RNG/cursor state,
tokenizer, complete sources and final receipts/graphs. Master parameter precision
is declared independently from execution precision. Worker staging stays within
the private session; checkpoint_target=0 still publishes the reserved output.
Null runtime pins are allowed only for a fresh pretrain command with corpus;
SFT/RL cannot bootstrap that way or replace a loaded worker. The four new shared
runtime fixtures and package precision examples remain generated-contract data.

Parent publication now verifies bounded generation/job metadata, complete package
bytes, exact queued configuration/pins/profile and reserved output before immutable
model rename. Training evidence, definition, run/model links and recovery retention
share one transaction before ACK. A separate `record_training_evidence` preserves
the completed-validation-only rule of `record_evidence`; one final-preflight
receipt is retained per admitted training job. Matching evidence survives history
pruning. A registration failure propagates and preserves its complete orphan;
startup recognizes modular packages, clears unpublished package staging, adopts
matching interrupted output and discards superseded unregistered recovery copies.
Human display/publication/keep metadata survives idempotent registration.

Packing now admits uint16 bytes, including separators, and document rows before
buffering/writing beyond declared caps; failures remove partial token output and
preserve prior complete packs. Exact fits remain byte-identical across chunks and
empty documents. Ordinary continuation restores moments while retaining requested
optimizer decay and current device options. Independent real AdamW next-update
comparisons cover changed decay; exact unchanged trajectories remain identical.

Authoritative reports (overlapping counts):

- `/tmp/feat4-worker-runtime-identity-final.xml`: 10 CPU backend/preparation
  checks passed in 31.71 seconds. Real standard/looped weights change, telemetry
  and complete package inventories are verified, failed inputs/capture never
  construct AdamW, actor/context remain correct, and mismatched selections fail.
  The owned saved tokenizer works without its original named build. These trusted
  in-process numerical fixtures do not simulate process containment or DB adoption.
- `/tmp/feat4-worker-contracts-packages-final.xml`: 562 shared contract/runtime/
  loop/support/package-data/transfer checks passed in 25.75 seconds; four actual
  process cases were explicitly excluded. An earlier collection failure from a
  missing new fixture version was fixed and is not counted as a passing report.
- `/tmp/feat4-worker-parent-state-final.xml`: 60 affected child/legacy state/
  submission checks passed in 47.67 seconds, with three process/dataset-dependent
  cases explicitly excluded. Later reports below supersede relevant changed paths.
- `/tmp/feat4-worker-frontend-decoder.log`: 214 decoder checks passed; typecheck,
  lint, generated frontend freshness and 67-schema freshness passed. There are no
  handwritten frontend behavior changes in this continuation.
- `/tmp/feat4-worker-owned-tokenizer-process.xml`: two actual numerical/tokenizer
  checks passed, the CPU process case failed at denied socket setup before child
  launch, and CUDA was skipped under the user's hardware deferral. This is a
  nonpassing integration report. No environment spoofing or parent tensor load
  was used to bypass the worker boundary.
- `/tmp/feat4-packing-continuation.xml`: 31 packing/loop/backend cases passed in
  50.36 seconds, with two process cases excluded. Added real packing checks cover
  exact limits, separators, empty/chunked documents and upstream failure cleanup.
- `/tmp/feat4-parent-publications.xml`: 16 parent/child cases passed in 33.68
  seconds, with two process cases excluded. The parent fixtures intentionally use
  labeled synthetic graphs/tensor bytes to isolate transaction/data-only behavior.
  An earlier metadata check failed because domain tuple shapes were not yet
  normalized to JSON values; this was fixed before the passing run.
- `/tmp/feat4-publication-child-regression-final.xml`: 98 checks passed in 65.30
  seconds, including expanded startup/orphan retention, exact optimizer/RNG/cursor
  trajectories, independent continuation updates, packing, real child numerical
  packages and legacy state. CPU/CUDA supervised cases and the one named
  dataset-service-dependent legacy end-to-end case were explicitly excluded.
  The earlier 76-pass matrix also had that denied dataset-I/O failure and is
  nonpassing; it does not count as full integration success.
- `/tmp/feat4-publication-submission-verified.xml`: 49 final parent-publication,
  pinned-submission and validation-queue checks passed in 17.45 seconds after
  isolating each parent fixture's artifacts. One actual supervised validation
  case was explicitly excluded. It covers publication rollback/adoption, immutable
  identity refusals, receipt dedupe and history-pruning retention.
- `/tmp/feat4-publication-postgres.xml`: both newly required real PostgreSQL
  registration/rollback cases failed connecting to the denied local socket before
  assertions. They remain runnable against migrated owned databases; no schema or
  Rust SQL changed, so no migration or SQLx regeneration is required here.
- Targeted Python compilation, function documentation/body-comment audit of 109
  scoped functions, all 67 shared schemas, frontend generated freshness and diff
  whitespace checks passed. No new Rust implementation changed in this continuation.

Remaining execution work: connect the parent queue handler, enforce effective
parent acquisition/staging capacity and large mapped-input memory budgets, and
verify real saved-package/process recovery. The default runner still refuses
pinned recipes before legacy execution. Parent adapter evidence is separate from
child arithmetic and actual containment; neither substitutes for the blocked live
gates. Final-profile recomputation candidates/retention and nonempty replay remain
unfinished; current child training supports empty/off/zero replay and refuses
nonempty requests before AdamW. SFT/RL/serving/UI and full acceptance remain open.
No issue was completed. GPU hardware remains user-deferred; commits are authorized
but still blocked by the read-only `.git` mount.

## Parent queue handoff, admission and saved-input continuation

Pinned pretraining now dispatches from the default Runner to TrainingHandler
under its existing lease. Parent-only acquisition preserves actor/context,
verifies saved definition/source authority, finalizes tokenizer facts and stages
private inputs. Actual child construction/capture/training/package publication
uses the existing backend. RegisteredCheckpoint is a typed already-committed
event, so live publication cannot repeat legacy registration. Terminal success
follows worker/session and temporary-input cleanup. Parent preparation cancellation
is distinguished from unrelated interrupted I/O, and a cancellation winning the
durable terminal row lock remains cancelled even after the last event check.

Ordinary modular resume now inherits a retained immutable definition/source/
tokenizer pin after original training-history pruning or catalog archive. It
requires admin actor/request identity, rejects structural overrides, and carries
the same pins through waiting chain consumers. Same-actor replay retains its
permission check after the saved input disappears. Saved package verification
precedes corpus access; tensor/source execution stays in the child. Newly derived
actual tokenizer builds are indexed; resumed owned bytes need no named build.

Parent admission observes available Linux memory and a finite cgroup allowance,
leaves host/disk headroom and accounts for input/recovery/raw/sealed/archive/grant/
parent/retained generation copies. It bounds stages/dataset selections, schedule/
optimizer inputs and fitting vocabulary sizes before acquisition. Preparation
grants cap at 8 MiB. Generation estimates and current staging capacity are checked
before AdamW, and actual tensor archive writes enforce exact per-file quotas.
Refusal preserves earlier complete files and removes partial output. Parent
adoption rechecks staging capacity. PostgreSQL publication transactions set 5s
lock and 30s statement timeouts, independently of the worker's adoption deadline.

Pure identity calculation now yields the original exact byte stream in bounded
parts. A separate scan adapter owns cancellation and readonly mapped-page release;
corpus/span checks use those boundaries too. Writable mappings remain untouched.
Corpus-key bounds admit existing filesystem-valid legacy keys without changing
their identities. No migration, public schema or Rust SQL changed.

Evidence (counts overlap):

- `/tmp/feat4-handler-existing-regression.xml`: 54 existing backend/loop/
  submission/publication/recovery checks passed in 56.34 seconds, with two actual
  process cases explicitly excluded, after initial handler connection.
- `/tmp/feat4-training-queue-regression-final.xml`: 179 passed in 93.12 seconds;
  the actual CPU/CUDA worker cases and named dataset-service-dependent legacy
  orchestrator case were explicitly excluded. This includes actual child updates,
  independent continuation and exact loop trajectories, legacy state, packing/
  chunking, parent adoption/retention, signed submissions/routes/chains and new
  handler/admission boundaries.
- `/tmp/feat4-training-queue-cancellation-final.xml`: 41 passed in 14.32 seconds,
  with one actual supervised validation excluded. This verifies the final typed
  prepared-request refactor, cancellation during acquisition, unrelated I/O
  failure, cleanup refusal, terminal-lock cancellation and no duplicate registration.
- `/tmp/feat4-training-queue-saved-admission-final.xml`: 33 final handler/
  submission/chain cases passed in 7.76 seconds. Added assertions cover structural
  override refusal, exact saved pins through waiting consumers, original history
  absence/archive and same-actor permission downgrade after saved-input deletion.
  SQLite fixtures explicitly remove their missing FK-cascade child; the actual
  PostgreSQL cascade remains a separate required gate.
- `/tmp/feat4-training-admission-last-input.xml`: 22 final admission/handler
  checks passed in 5.45 seconds after tightening the per-file input ceiling to
  include spans/tokenizer grants as well as corpus/output. A real sparse-file
  boundary covers a span table larger than both corpus and admitted generation.
- The admission suite includes a real 128 MiB mapping/RSS measurement: mapped
  pages are released and chunked hashing retains less than 32 MiB above the
  measured baseline, with identical hash bytes. Actual PyTorch quota refusal
  preserves prior tensor bytes and removes partial files. These are local host/
  serialization checks, not worker containment or GPU evidence.
- `/tmp/feat4-training-queue-process-final.xml`: two genuine CPU queue/saved-
  recovery cases fail with PermissionError before worker assertions under the
  known socket restriction; CUDA is user-deferred. The recovery case remains
  runnable: real default handler, cancellation after durable checkpoint adoption,
  signed recover, original tokenizer-build removal and exact uninterrupted/final
  weight-byte comparison. Neither CPU integration case is a pass.
- `/tmp/feat4-training-queue-postgres-final.xml`: four fixture connection errors
  at the denied local socket. Registration/rollback, real history cascade/saved
  pins and finite lock-wait cases remain runnable but did not reach assertions.
- Current 67-schema and frontend generated freshness checks pass. Scoped
  documentation/body-comment audit passed for 99 functions; targeted Python
  compilation and diff whitespace checks pass. No handwritten frontend or Rust
  implementation changed in this continuation. Earlier intermediate fixture/
  expectation failures are not counted as passing final reports.

The handler is connected, superseding the prior temporary pinned-pretrain
refusal. Pinned SFT/RL still refuse legacy numerical execution. Remaining work:
actual saved/process acceptance, final-profile recomputation candidates/retention
and nonempty replay, then SFT/RL/serving/UI and full acceptance. No issue is newly
complete; eight of 21 remain accepted. GPU hardware stays user-deferred and
commits remain blocked by the read-only mount.

## 23 October 2026 — explicit eager replay and retained training plans

Pretraining now accepts explicit occurrence policies after conservative meta and
real admission. True selects all eligible calls, false/zero none, and an integer
selects that many calls in deterministic execution order. Unknown counts refuse
before AdamW. Selected eager computation must match ordinary strict capture's
full training batch/context outputs, gradients and post-backward CPU/device RNG.
Probes preserve caller RNG, modes, tensors and gradient references. Evaluation
continues through ordinary eager execution; loops and Parameter ownership do not
change. Explicit CUDA replay refuses compiled execution before numerical model
construction, with an actionable compile_model=false error.

The current full-retention-replay-reserve-v1 estimator reserves an additional
complete captured activation/workspace bound for nonempty replay. It claims no
memory reduction; null selects zero. Automatic memory-saving selection remains
unfinished. Installed Torch strict checkpoint export fails in the higher-order
checkpoint wrapper; an explicit selective context avoids that failure but
export decomposition then fails on body constants. No installed code was patched,
no opaque graph was accepted and no guessed discount was introduced. A separate
trusted probe shows ordinary strict export can retain actual shared/nested call
annotations, but it is not production storage-lifetime evidence.

Each new generation inventories training-plan.json containing definition/receipt
identity, memory bounds and exact-profile selected/candidate occurrence records.
Results repeat the policy; optimizer state retains the selected profile,
occurrences and estimate. Exact recovery requires that retained selection alongside
matching config/data/schedule/cursor. Old state lacking this authority is refused
for exact recovery; ordinary continuation remains available after final preflight.

Evidence (overlapping counts):

- /tmp/feat4-replay-regression-final.xml: 81 passed in 247.25 seconds, nine actual
  process/CUDA cases deselected. Includes real child training, optimizer/cursor
  trajectories, package publication, handlers and strict capture validation.
  Current repository tests are imported through the offline wrappers; previously
  copied support/validation wrappers were replaced to avoid stale test collection.
  One harmless unknown-PostgreSQL-marker warning comes from the offline fixture
  configuration; PostgreSQL tests did not run.
- Real standard, shared-loop and nested-loop explicit replay updates equal
  ordinary final weights exactly. Every saved package's planning inventory is
  checked against its exact definition, receipt and captured profile.
- /tmp/feat4-replay-recovery-final.xml: two passed in 40.74 seconds. Fresh trusted
  CPU sessions load actual serialized standard/shared-loop weights, optimizer,
  RNG and cursor and recover identical final weights. The fixture first verifies
  production parent tensor-load refusal, then substitutes only the tensor-reader
  entry point using the real package/hash/state decoder. No runtime-child marker
  is spoofed. This is saved numerical recovery, not guardian restart or admission.
- /tmp/feat4-replay-rng-final.xml: three passed in 5.60 seconds. Actual dropout
  replay matches ordinary strict export outputs/gradients/RNG; extra hidden RNG
  draws fail even when values/gradients agree. Conservative admission never
  invents an automatic saving or clamps an unknown occurrence count.
- /tmp/feat4-replay-policy-boundaries-final.xml: three passed in 5.15 seconds.
  Unknown counts and compiled-CUDA replay refuse before optimization/construction;
  exact continuation refuses missing/stale selection with otherwise matching
  data/config/cursor. The CUDA branch is a policy-only CPU-grant fixture, not
  hardware arithmetic. It was added after broad collection and checked separately.
- Genuine supervised pretraining now has replay-on/off CPU/CUDA cases with saved
  plan assertions. Process socket denial remains the previously measured open
  gate; these expanded cases were not reported as passing. Actual queue recovery,
  HTTP and PostgreSQL checks remain blocked/unverified as previously recorded.
  CUDA execution remains user-deferred; compiled checkpoint replay is unfinished.
- Scoped AST/documentation/body-comment checks pass for 87 functions, targeted
  compilation and diff whitespace pass. No schema, Rust SQL, handwritten Rust or
  frontend implementation changed. Intermediate failed fixtures are superseded
  by the named passing reports and are not counted as passes. All test processes
  from this continuation reached terminal status.

The feature remains active at eight of 21 accepted issues. Remaining work includes
automatic replay memory estimation, actual process/database acceptance, SFT, RL,
serving/scoring, connected UI and final cross-module acceptance. Main Discord is
polled; no new human instruction arrived. No sudo/Docker used. Regular commits
remain blocked by the read-only .git mount; no commit succeeded.

## 23 October 2026 — requested overnight pause

The user returned, asked for percentages and requested a graceful stop before
turning this computer off. Confirmed acceptance remains eight of 21 issues
(38%). Including partial implementation, the current effort estimate is about
65%, with a rough 60–70% range; it is not an acceptance/test percentage.

The automatic-planning draft now partitions actual ordinary strict captures by
engine-owned occurrence annotations, observes all named output backing storage
(including unused ports and larger aliased bases), and calculates bounded
prefix-policy candidates. It retains unselected work and the largest replay
region using existing tensor/workspace bounds. Tests exercise real shared/nested
models, minimal fitting selection, numerical/RNG equivalence, stale partition
refusal and compiler-metadata source restrictions. The draft also preserves a
matching saved selection under a changed allocator ceiling. These changes are
unfinished and require full review/regression before production adoption.

Trial production wiring failed: the standard builtin's eager positions check
calls int(positions.max()), which cannot run on meta tensors during port-storage
observation. /tmp/feat4-recompute-worker-first.xml reports seven failures,
12 passes and five deselections. It is not passing evidence. Production wiring
was removed before stopping: preflight continues using the previously verified
conservative explicit-replay estimator. No installed Torch code was modified.
The draft library, annotations, tests and separated estimate/admission functions
remain preserved for resumption.

Final stop check /tmp/feat4-nightly-stop-smoke.xml: 10 passed in 21.74 seconds,
covering eight current draft cases, real standard pretraining/package generation
and invalid replay-count refusal after restoring the conservative production
path. This is a focused smoke check, not a complete regression after the latest
annotation/source-policy edits. Scoped declaration/body-comment checks pass for
93 functions; compilation and diff whitespace pass. All test/probe handles from
this continuation reached terminal status; no development/database services or
subagents were started, and no agents remain working apart from root cleanup.

The user can preview frontend components at /markup/authoring and /markup/graph
with the native frontend running on port 3000 (cd frontend && bun run dev).
These are local fixtures: Python/YAML editors, typed controls and handcrafted
interactive graphs, with no catalog mutations. The connected feature workflow
is unfinished. Earlier browser checks passed; a fresh full stack launch was not
verified in this restricted session. Backend/live-process/PostgreSQL checks and
deferred CUDA execution remain separate open gates.

Current state, precise resume tasks, percentages and visual-preview instructions
are recorded in the resume handoff and Discord. The goal is paused at the user's
request; continue only when resumed. Files remain uncommitted on modular-arch
because .git is read-only. No sudo or Docker was used.

## 23 October 2026 — scoped Compose startup repair after pause

The user supplied real startup output showing a frontend schema preflight ENOENT
at /contracts/modular-architecture/v1/schemas and repeated PostgreSQL failures
because a generic FOR UPDATE tried locking nullable eager domain joins.

Compose now mounts contracts read-only at /contracts. The production frontend
Dockerfile copies contracts and the root .dockerignore includes them in build
context. The row-lock helper uses FOR UPDATE OF execution_jobs and retains eager
domain snapshot reads plus populate_existing. This preserves the execution-row
owner for cancellation/completion; no schema/applied migration or Rust SQL changed.
SQLx regeneration is not applicable.

Verification: /tmp/feat4-startup-repair.xml has 24 passes and six PostgreSQL cases
excluded. This covers actual-helper PostgreSQL SQL compilation, existing store
behavior and training handler boundaries. The schema freshness check passes both
natively and in a temporary filesystem layout matching the container's /app and
sibling /contracts relationship. Compose YAML, build-context inclusion and image
copy paths are checked; Python compilation and diff whitespace pass.

/tmp/feat4-startup-repair-postgres.xml contains six connection-setup errors at the
sandbox-denied local PostgreSQL socket, with no runtime assertions reached.
Runnable cases cover all five job kinds with absent domain rows and real
same-job contention/unrelated-job independence. Docker/sudo were not run, so a
live container restart is not claimed. The user must recreate the frontend to
apply mounts; rebuilding with a renewed anonymous dependency volume also picks
up the changed frontend lockfile. Engine source is bind-mounted and needs reload
or restart, without an image rebuild for this query change.

This was a scoped startup repair requested after the pause. The main FEAT-4 goal
remains paused; no new issue was accepted and no background work was resumed.

## 24 October 2026 — automatic captured occurrence planning

The user resumed FEAT-4. Meta and real pretraining preflight now use the captured
occurrence planner. Null selects the fewest fitting prefix calls; explicit
policies must fit their own bound. The estimator charges all named output backing
storage independently of exported users, including unused outputs/views, keeps
unselected work and reserves the largest replay region. Grad-enabled observation
avoids undercounting a cheaper no-grad branch, and hooks/graphs do not escape.
Ordinary strict capture remains complete; checkpoint export failure is not
bypassed with opaque graphs. This is supported-operation policy, not CUDA
calibration or a guarantee about arbitrary custom code.

The standard legacy model permits shape-only meta position assertions without
scalar extraction. Real eager bounds and strict runtime assertions remain.
Builtin manifests/revisions changed and all copyable YAML references, including
stack templates, were refreshed. The existing pinned numerical trace is unchanged.
Exact recovery preserves the saved profile/storage selection under changed
allocator ceilings; absent/stale authority refuses. Compiled CUDA replay still
refuses, including automatic selection during meta admission.

Evidence (counts overlap):

- `/tmp/feat4-auto-worker-regression.xml`: 37 passed in 208.86 seconds, four
  actual process/CUDA cases excluded. Covers real standard/looped/nested updates,
  all meta layouts, grad-dependent storage, least fitting automatic selection,
  full package planning inventories and real saved tensor/optimizer/RNG/cursor
  recovery under a larger allocator. The trusted tensor-reader fixture checks
  parent refusal, then invokes the genuine package/state decoder without spoofing
  a child marker. It proves saved arithmetic, not process containment.
- `/tmp/feat4-auto-library-legacy-final2.xml`: 105 passed in 19.49 seconds,
  three user-deferred CUDA skips. Unchanged pinned trace, initial weights/RNG,
  source examples, shared/dropout replay and memory/diagnostic behavior pass.
- `/tmp/feat4-auto-contract-boundaries.xml`: 427 passed in 73.72 seconds,
  three CUDA skips, seven process cases excluded. Current repository tests are
  imported through isolated wrappers without unrelated TCP fixtures. Covers
  contract/schema freshness, actual capture and byte/package/parent boundaries.
- `/tmp/feat4-auto-supervised.xml`: one real CPU case fails at sandbox-denied
  `SO_SNDBUF` setup before child launch. `/tmp/feat4-auto-postgres.xml`: four
  fixture errors at the denied local database socket, before runtime assertions.
  These fresh checks leave the genuine supervision/database gates unverified.
- Compilation, 74-function documentation/body-comment audit and diff whitespace
  checks pass. No public schema, migration, Rust SQL or frontend code changed.
  Intermediate capture expectation and old-example-pin failures were corrected;
  their reports are not passing evidence. The initial 20-worker pass predates the
  last accounting changes and is superseded by the 37-pass final report.

No issue newly accepted; FEAT-4 remains active at eight of 21 accepted issues.
Automatic planning supersedes the conservative production fallback described in
the historical entries. Remaining work includes actual process/save/recovery
acceptance, SFT/RL/serving/scoring, connected UI and full cross-module acceptance.
GPU hardware remains user-deferred. No sudo/Docker/commit/push was used.

## 24 October 2026 — SFT ownership and private data preparation support

ISS-59 is in progress. Token-row warm-start now requires a declared whole-model
capability; simple full-interface single-child compositions can delegate while
local/repeated child capabilities are not guessed. Standard and legacy share
one original-mean numerical owner; custom implementations own tied/untied tables.
Unsupported models preserve every weight and report a distinct skipped result.
Requested IDs are bounded/deduplicated and custom result ownership is checked.

The data-only SFT tokenizer helper verifies saved byte identity, derives private
memory labels only when their IDs are used, preserves all IDs/merges and records
deterministic UUID/hash plus original fingerprint. No catalog/filesystem writes
occur there, and a claimed derived_from alone cannot authorize publication.
Renderer/storage helpers now accept a caller-owned private directory. Explicit
budgets bound combined token/mask bytes, rows and source characters before
append/encoding; per-row/example cancellation and source failures clean partial
files before publication. The real old-placeholder case also fixed omitted
memory IDs in renderer history, preserving assistant-only targets.

Final available evidence (overlapping counts):

- `/tmp/feat4-sft-support-final.xml`: 94 passed in 39.16 seconds, three CUDA
  skips, after sharing initialization and refreshing all example pins. Includes
  real legacy fine-tuning/private memory tokenizer output, unchanged pinned
  trace, actual custom source table ownership, independent masked AdamW updates,
  and exact in-memory optimizer/RNG/next-batch recovery after targetless batches.
- `/tmp/feat4-sft-data-final.xml`: 62 passed in 17.92 seconds after bounded
  rendering. Includes exact-fit token/mask bytes, document refusal, cancellation/
  source failure after a real partial flush, unchanged prior complete files,
  pre-encoding source refusal, actual old-placeholder rendering/derived identity,
  private/base byte immutability and legacy memory SFT. The first ten-case data
  report had one test constant-name error; its corrected final report passes.

These are library/loop/preparation checks. Pinned SFT is still refused by the
default runner. Mask grants, parent admission, child final preflight/reference
models, derived-tokenizer publication authority and actual saved SFT recovery
remain implementation work. No issue was completed. The full goal stays active;
no subagents, sudo, Docker, commit or push were used.

SFT follow-up quality checks: targeted Python compilation, diff whitespace and
43 scoped function documentation/body-comment checks pass. The first declaration
audit overselected unrelated pre-existing Transformer component forwards; the
corrected class-specific audit covers only the changed Transformer methods.

## 24 October 2026 — fully validated pretraining recovery grants

Tracing SFT's next worker handoff exposed a pre-existing pretraining mismatch:
the actual parent constructs a fresh Train command carrying a self-contained
model_package_v1 recovery grant, but its validator still required checkpoint_v1.
Saved numerical tests had used model_copy updates, which bypassed that validation.

Fresh pretraining now requires the model package for recovery; resident pinned
training retains its separate checkpoint role. Positive and negative shared
fixtures cover bootstrap packages and role confusion in both directions. Real
saved numerical recovery tests now construct the command through model_validate
before the backend runs. No runtime/schema field shape or Rust SQL changed.

- `/tmp/feat4-recovery-wire-final.xml`: 524 contract/worker cases passed in
  184.23 seconds, four actual supervised/CUDA cases excluded. Includes real
  standard/shared explicit and shared automatic saved numerical recovery under
  changed admission, using full command validation and the documented trusted
  tensor-reader seam. It does not prove process containment/restart.
- `/tmp/feat4-recovery-wire-roles-final.xml`: 503 final current contract/role
  cases passed in 1.79 seconds, including all three new shared role fixtures and
  67-schema source freshness. Counts overlap with the larger report.
- Targeted compilation, changed-validator declaration/body-comment audit and
  diff whitespace checks pass. No further genuine process/PostgreSQL retry was
  needed after the independently confirmed socket denials earlier this turn.

All test handles from this continuation are terminal. FEAT-4 and ISS-58/59 stay
in progress; acceptance remains eight of 21 issues. The next implementation is
the actual saved-definition SFT parent/worker/package path, followed by RL,
serving/scoring, connected UI and cross-module acceptance. Work is uncommitted;
no delegation, sudo, Docker, commit or push was used.

## 24 October 2026 — saved-definition SFT queue, worker and publication

SFT now uses the existing parent/child training owner. Private aligned mask/data
grants, effective execution options, inherited model construction and new-optimizer
versus exact-SFT recovery are connected. Optional original-reference construction
preserves RNG and charges exact captured reference weights/workspace during meta
admission. Bounded masked/pretrain/KL diagnostics preserve training state;
optional diagnostic OOM disables sampling without swallowing training errors.

The parent renders through the canonical template with actual actor propagation,
hashes corpus plus mask and keeps tokenizer/cache preparation private. A durable
prepared-input artifact binds source/config/data/final tokenizer/provenance before
worker launch. Publication checks it and actual tokenizer bytes before rename
and registration. Original enqueue pins remain immutable. Exact recovery accepts
an original-tokenizer pin only after original-base/recovery byte proof establishes
the allowed label derivation. Failed registration retains the proof for startup;
settled terminal/pruned proofs retire after package reconciliation. Known pretrain
marker history now reaches packages; unknown ancestry still authorizes no reset.

Public `/sft` now freezes registered saved-definition pins and requires the same
administrative actor/request identity as modular pretrain. Rust `/runs/sft`
forwards its query, signed administration and original status/body through the
shared submit adapter. Shared runtime fixtures cover empty-worker SFT with an
inherited package, absent corpus and mask/probe/package role confusion. Runtime
artifact/envelope schemas and generated frontend types include mask/probe formats.
No database schema, migration or Rust SQL changed; SQLx regeneration does not apply.

Available evidence (overlapping counts, with final follow-ups recorded below):

- `/tmp/feat4-sft-numerical-final.xml`: 16 passed in 276.69 seconds. Actual
  standard/shared-loop/custom SFT, serialized tensor/optimizer/RNG/cursor recovery,
  known marker provenance, mask/probe/source refusals before construction,
  reference-only meta admission refusal and optional KL OOM update/RNG identity.
- `/tmp/feat4-sft-interrupted-numerical-final.xml`: three passed in 128.17 seconds
  after strengthening recovery to actually cancel immediately after its first
  complete package. Fresh trusted sessions recover standard/shared/custom weights
  and RNG exactly. The trusted reader checks parent-load refusal before decoding
  its own safe tensor data. This is numerical evidence, not guardian restart.
- `/tmp/feat4-sft-parent-contract-final.xml`: 632 passed in 31.81 seconds before
  the last public-SFT/derived-recovery additions. `/tmp/feat4-sft-queued-signed-final.xml`:
  104 passed in 26.79 seconds after them, including signed SFT identity/conflicts,
  derived checkpoint recovery with copied original pins, cancellation/cleanup
  ordering, forged proof refusals and post-rename startup adoption. Parent tests
  explicitly use synthetic numerical outputs; the renderer test uses real text.
- `/tmp/feat4-sft-library-data-current.xml`: 138 current repository checks passed
  in 35.02 seconds; one dataset-service-dependent legacy check excluded. Covers
  real legacy memory SFT, independent masked updates, token rows, renderer budgets,
  private adoption, original pinned trace and diagnostics. The earlier library
  report had that external-service failure; it is not a fully passing report.
  Four old copied test files in the temporary harness were replaced with imports
  of current repository tests before this final run.
- `/tmp/feat4-sft-wire.xml`: 145 passed; all 67 schemas and generated frontend
  freshness pass. Frontend typecheck/lint and 214 decoder/transport/identity checks
  pass. Generated frontend files changed; handwritten frontend integration remains
  the later UI slice.
- Rust offline build, strict workspace/all-targets Clippy and the changed API
  crate's edition-2021 formatting checks pass. The extended private-route
  authorization test passes. Its real network forwarding test fails at denied
  Actix socket binding, before forwarding assertions.
- `/tmp/feat4-sft-supervised.xml`: CPU real supervision fails with permission
  denial before the base worker starts; CUDA skips as user-deferred.
  `/tmp/feat4-sft-postgres.xml`: three fixture connection errors at the denied
  local PostgreSQL socket. Runnable SFT reference/cancel/signed-recovery and real
  PostgreSQL derived-binding/rollback/history-cascade checks are retained.

The first SFT worker report exposed the difference between consumed batches and
optimizer steps; final SFT metadata now explicitly records skipped targetless
batches while retaining the original schedule and strict pretrain completion.
Intermediate test-construction/fixture-identity failures were corrected and are
not passing evidence. No issue newly accepted: eight of 21 remain accepted, and
FEAT-4/ISS-58/59 stay in progress. The user received an approximately 68% overall
implementation estimate (rough 65–70% range), distinct from 38% accepted by count.
Remaining implementation is RL, serving/scoring and connected UI, followed by
full requirement acceptance and the still-open actual process/database gates.

SFT final follow-ups: `/tmp/feat4-sft-authority-final.xml` has 24 passes after the
last row-ownership validator. `/tmp/feat4-sft-pretrain-shared-final.xml` has 23
passes and four actual supervision/CUDA exclusions after all shared SFT changes.
The named 81-function SFT declaration audit passed before the last documented
validator addition; targeted compilation and whitespace checks pass afterward.
No live gate was closed from these results.

## 24 October 2026 — RL evaluation policy and worker effect foundation

ISS-60 is in progress. `model.language.evaluating` changes mode without disabling
autograd and restores heterogeneous child modes on success/error. Actual policy
recomputation now uses evaluation behavior rather than activating custom dropout.
Full-prefix logits use explicit LM metadata instead of a Transformer tuple.
Bounded ragged chunks retain drawn-target masks and the original normalization;
invalid provenance/options refuse before forward. Batches without drawn targets
perform no update/weight decay. The existing unfiltered unit-temperature policy
objective versus temperature/truncation/grammar-masked sampling is now documented
as a surrogate approximation, without changing the sampler or grammar.

`train/rollout_decode.py` owns one round's legacy cache or declared full-prefix
fallback with explicit positions/masks, context bounds and cleanup. Standard,
shared-loop and actual custom source dropout models perform real rollout/policy
work through a finite trusted gym seam. Unsupported declared caches stay off;
whole-model capability/profile-backed cache authorization remains unfinished.

`gym/interfaces.py` names the existing rollout/curriculum seam. The new worker
`gym/effect_client.py` has no URL, actor or credentials. Typed bounded interpreted
records validate finite results and public task identities; duplicate JSON fields
refuse. A lock serializes existing rollout threads to one outstanding runtime
effect. Consumed grants retire. Failed/uncertain exchanges latch the client so
queued calls cannot issue additional commands from an interrupted iteration.
The parent dispatcher/resource authorization and supervised RL backend are not
connected yet; client fixtures do not prove those boundaries.

Review also found that parent callbacks could outlive a cancelled/reaped child.
The session now atomically prevents new effect/adoption callbacks during closure
and drains their actual Futures before deleting grants or releasing admission.
The broker stops new issuance and retains write journal semantics. A callback
drain timeout retains the directory and quarantines admission; it never certifies
that the external write stopped. Finished adoption errors still propagate as job
failures without pretending their callback is live. Parent effect failures log
their internal cause while returning bounded generic errors to the worker.

- `/tmp/feat4-rl-foundation-final.xml`: 199 passed in 5.42 seconds across actual
  policy/rollout/legacy RL, worker-effect/grant/journal and runtime wire checks.
  Real standard/shared/custom gradients and AdamW updates match independent
  unpadded objectives across micro-batch sizes. Dropout/RNG/modes and frozen
  reference gradients are checked. Ragged legacy cached and declared full-prefix
  decoding match individual rows. These are library/adapter checks.
- `/tmp/feat4-rl-effects-current.xml`: 154 passed in 0.70 seconds after final
  effect-failure observability. Includes real concurrent callback/grant cleanup,
  uncertainty latch and journal reopening refusal, cancellation before issuance,
  malformed result refusal, active callback drain and honest timeout behavior.
- `/tmp/feat4-rl-foundation-current.xml`: 58 passed, 170 excluded by its selected
  scope; adds available denied-socket partial-setup cleanup checks. It is not a
  broader replacement for the 199-pass report.
- `/tmp/feat4-parent-callback-supervised.xml`: both real effect/publication
  cancellation/reap/drain cases fail at denied `SO_SNDBUF` setup before child
  launch. The earlier one-case effect report has the same setup failure. Actual
  process acceptance remains open; no unconfigured-socket bypass is used.
- Targeted Python compilation and whitespace pass. Scoped foundation documentation/
  body-comment checks cover the new modules and changed shared functions; the
  audit exposed and removed the touched rollout terminal-round body comments.
  All 67 schemas remain fresh; no runtime wire shape or database schema changed
  during this RL foundation. The Rust SFT bridge has no SQL changes.

Remaining work: complete parent-authorized bounded Rust gym dispatch and durable
issued-resource ownership, connect actual policy/reference/meta admission and
completed-iteration package recovery, authorize optional caches with real profile
evidence, then serving/scoring, connected UI and full acceptance. No issue newly
accepted; the full goal stays active. No subagents, sudo, Docker, commit or push.

## 24 October 2026 — queued RL, durable effects and completed recovery

Pinned RL now dispatches through the shared admitted parent/child lifecycle.
Parent-only actor-bound HTTP has one attempt, bounded streamed bytes, finite I/O
timeouts and elapsed checks between reads. Typed gym dispatch checks the saved
schedule, exact seeds, group membership, sibling public contexts and owned resource
IDs before exposing grants. Issued writes and acquired resources are durably
journaled before use. Uncertain writes fence the old job; recovery never replays
them. Separately journaled cleanup closes only known untouched episodes and
abandons live runs after verified callback/worker drain. Startup can reconstruct
cleanup after the original job/configuration is pruned. Unconfirmed cleanup stays
retained. No claim is made that an elapsed check interrupts an in-flight RPC.

Admission bounds parent response/ownership memory, per-job journals, aggregate
disk inventory and private archive copies before execution. The RL child restores
the saved definition/tokenizer, original frozen reference, optimizer and global/
task/sampling RNG. Completed iteration packages retain curriculum and exact next
training-group position; recovery does not repeat initial placement. Partial
iterations publish nothing. Legacy sampler/objective semantics remain deliberate.
RL uses the common package serialization/adoption owner and retains fixed-test
UUIDs as metric links, never as authority to reuse external gym resources.

Public `/rl` and Rust submission preserve actor/admin/request/query/status
contracts. Shared runtime fixtures permit corpus-free inherited RL bootstrap and
reject confused grant roles. All 67 schemas and frontend generated contracts
were refreshed. No database schema or Rust SQL changed, so no migration or SQLx
regeneration applies to this continuation.

Two adjacent regressions were reproduced before fixing: SFT refused an inherited
pending tokenizer even after its predecessor published the real model; RL retained
an overly strong pretraining marker-history claim that could authorize SFT to
reset rows RL had learned. SFT now uses its registered source and durable private
preparation proof without rewriting enqueue pins, including after predecessor
history pruning/root archival. RL marks marker history incomplete until measured
context/action history can establish completeness.

Available evidence (counts overlap):

- `/tmp/feat4-rl-numerical-current.xml`: eight real standard/shared/custom RL,
  exact serialized recovery, interrupted iteration and curriculum cases pass.
  `/tmp/feat4-rl-position-recovery-current.xml`: four exact/curriculum recovery
  cases pass after strict seed-position authority. Weights, AdamW moments and
  global/task/sampling RNG match; these use a trusted tensor-reader seam that
  first verifies production parent loading is refused, not real guardian restart.
- `/tmp/feat4-rl-marker-history-after.xml`: two actual numerical marker/curriculum
  checks pass. The preceding marker report failed by demonstrating learned rows
  would be selected for reset. `/tmp/feat4-sft-rl-chain-current.xml`: 60 current
  SFT/publication/chain checks pass, including pruning/archive of the predecessor.
- `/tmp/feat4-rl-group-authority.xml`: 49 gym dispatch checks pass, including
  exact remaining seeds/groups and malformed sibling refusal.
  `/tmp/feat4-rl-parent-admission-publication.xml`: 101 pass;
  `/tmp/feat4-rl-queue-contracts.xml`: 89 pass;
  `/tmp/feat4-rl-startup-recovery.xml`: six pass. These exercise actual parent
  adapters with explicit synthetic worker/HTTP seams, not live Rust/gym services.
- `/tmp/feat4-rl-preparation.xml`: 19 preparation boundaries pass.
  `/tmp/feat4-checkpoint-rng.xml`: CPU device-mapped RNG restore passes; one CUDA
  case skips. Shared RNG tensors now return to the generators' owning devices.
- Rust offline workspace build and all-targets Clippy with warnings denied pass;
  private-route authorization passes with RL included. Actual forwarding still
  fails at Actix TCP bind permission denial. Frontend typecheck and schema/type
  freshness pass. Scoped compilation/function documentation and whitespace pass.
- `/tmp/feat4-rl-supervised-gate.xml`: three real CPU workers fail before launch
  at denied `SO_SNDBUF` setup; three CUDA cases skip as user-deferred.
  `/tmp/feat4-rl-queued-supervised-gate.xml`: genuine queued CPU base-to-RL recovery
  fails at the same setup before the base worker starts; CUDA skips.
- `/tmp/feat4-rl-postgres.xml`: three connection errors;
  `/tmp/feat4-sft-chain-postgres-current.xml`: four connection errors, including
  the new pending-chain contract. All fail at the denied PostgreSQL socket before
  assertions. Live supervision/network/database acceptance remains open.

FEAT-4 remains active at eight of 21 accepted issues. Serving/scoring, optional
validated caches, connected UI and full cross-module acceptance remain unfinished.
No issue was closed from fixture/build evidence. No subagents, sudo, Docker,
commit or push were used.

## 24 October 2026 — saved-definition full-prefix serving and metadata

ISS-61 is in progress. The existing runtime manager now owns a reusable modular
serving worker selected by `/chat`, `/chat/stream` and `/measure`. Parent-side
package verification, private copies/admission, typed requests/results, bounded
token queues and cancellation are connected. Source/tensor construction and
generation/scoring remain child-owned. Model generation/decoding lives below
training; worker orchestration lives above it. The legacy adapter remains usable.

Actual serving exposed a missing one-row domain in training receipts that had
captured only the configured larger batch. Final pretrain/SFT preflight now captures
batch one through that batch in both modes; old receipts are never widened or
silently reused. The worker requires exact saved context/device/arithmetic and
complete-prefix coverage before loading. Strict meta capture charges unique model
storage, loading copies, supported workspace/logits and runtime host headroom
before real weights. Serving does not load optimizer/RNG history. Weights-only
decoding no longer reads the same sibling tensor file twice, and loaded-package
closure drops model/state tensors before import registrations retire.

Continuation/likelihood use the same LM facts/logits and request-owned decoder as
rollout. Canonical grammar/tool handling and UTF-8 rendering are shared with the
legacy path. Disconnect signals cancellation before waiting for the owned drain
thread and verified process cleanup; queue backpressure remains cancellable.
Completed results are parsed before queue-priority eviction deletes their grants.
Parent checks actual vocabulary/context/ID counts, text and tool calls against
the delivered stream. Final incomplete UTF-8 flushes without inventing an ID.
Parent hash/archive/grant copies observe cancellation between bounded reads.

Administrator `/models/{id}/definition` and `/graph` readers now use bounded
immutable metadata/inventory/receipt bytes without tensor/source loading or live
catalog dependencies. Summary `validation_receipts` retain exact graph identity
after original job pruning. Shared positive/stale nested-receipt fixtures and
generated frontend schemas/types cover that addition. Graph reads fetch only the
requested immutable scope and keep cursor/profile checks. Generic model listings
use saved context/vocabulary/measured counts and null incompatible legacy shape.
Receipts cap at 8 MiB aggregate; complete package verification also enforces the
existing aggregate graph-byte cap. No database schema or Rust SQL changed.

Available evidence (overlapping counts):

- `/tmp/feat4-serving-worker-current.xml`: five pass in 91.50 seconds. Genuine
  trained standard/shared/nested/custom packages perform repeated chat and
  likelihood/continuation in one backend; insufficient meta admission refuses
  before weight deserialization. Production parent loading is first verified to
  refuse; the trusted construction/tensor seam proves numerical behavior, not
  process containment. Immediate end markers are valid successful generations.
- `/tmp/feat4-serving-boundaries-current.xml`: 590 pass, one supervised test
  excluded. Actual data-only definition/graph HTTP reads, signed administrative
  privacy, public model facts, shared fixtures, archive cancellation/integrity,
  synthetic resident reuse/disconnect/eviction/invalid-output boundaries pass.
- `/tmp/feat4-serving-inference-parent-final.xml`: 87 pass, one supervised case
  excluded, after module layering and UTF-8 completion changes. Includes unchanged
  pinned trace, independent standard/shared/nested/custom likelihood/sampling,
  final incomplete-byte output, shared parent pretrain/SFT publication and chains.
- `/tmp/feat4-serving-profile-numerical-final.xml`: five pass, 22 excluded, in
  143.14 seconds. Real standard/shared/nested pretraining and actual serialized
  shared-replay exact recovery pass with the wider captured batch domain.
  `/tmp/feat4-serving-profile-training-regression.xml` is not a passing report:
  54 passed, four excluded and seven setup errors after old imports met files
  edited during that run. Fresh parent/numerical checks above cover the affected
  subsets; a full broad-suite pass is not claimed.
- `/tmp/feat4-serving-legacy-contract-current2.xml`: 188 pass, three external
  dataset-service cases excluded, seven subtests pass. Grammar/tool budgets,
  actual legacy SSE, context fitting, UTF-8 decoder and runtime fixtures pass.
- Frontend typecheck/lint pass; 212 shared decoder/HTTP tests pass, including
  retained/stale nested receipts. The final architecture-folder selection passes
  229 across five files, including graph/layout and request-owner races. All 67
  schemas and generated types are fresh. Final scoped parsing/compilation, 98
  function documentation/body-comment checks, local links and whitespace pass.
- `/tmp/feat4-serving-supervised-gate.xml`: real parent/guardian serving fails at
  denied SO_SNDBUF before child launch. `/tmp/feat4-serving-cuda-gate.xml`: one
  runnable hardware case skips as explicitly user-deferred. The Chrome MCP
  connection also timed out after 60 seconds; no browser/live service pass is
  implied. Earlier Rust forwarding/PostgreSQL socket denials remain open gates.

Full-prefix serving is connected; optional declared cache/context-extension
authorization, occurrence-based attention, connected UI and actual cross-module
acceptance remain unfinished. FEAT-4 stays active at eight of 21 accepted issues.
No new issue was accepted from these checks. No subagents, sudo, Docker, commit
or push were used. See [serving ownership](modular-serving.md) for the current map.

## 24 October 2026: occurrence-aware attention and connected model viewer

ISS-61's full-prefix serving owner now handles typed discovery and attention
requests in the same resident numerical worker. Declared candidates require both
introspection/attention capabilities and an actual matching whole-model or
named-port method. Shared/nested calls have distinct opaque point IDs and retain
one parameter owner. Supplied standard/layer/attention blocks expose stateless
canonical projection/probability views; their changed implementation manifest
produces new builtin identities and all five YAML examples were refreshed.

Each requested view/domain requires strict meta admission and actual complete
capture, singleton/range eager agreement, unchanged ordinary logits, causal finite
normalized probabilities, deterministic RNG and mutation checks. Returning normal
logits internally preserves the full actual work in capture/memory estimates.
Planning charges tensor snapshots, conservative autocast copies and workspace.
Proofs retain no tensors/graphs and bind exact saved definition/profile/view/bounds;
weight/layout changes or worker retirement invalidate them. Cache, training and
extension are not authorized by these observation proofs.

Public discovery and matrix contracts, Rust AuthUser discovery forwarding and
frontend controls are connected. Discovery loads only on explicit action.
Generic views use actual instance/call labels and honest available head counts;
legacy models/direct standard composites retain numeric layer/head adapters.
Shared decoders validate axes, bounded fp32 base64/probabilities and runtime proof
identity. Selection/text/session replacement aborts work and suppresses late
responses; a replacement worker clears old point proofs. Static classified
refusals expose no custom source exceptions. The engine owns shielded draining
for disconnected nonstreamed chat/attention waiters. An explicit null resolved
architecture is authoritative and cannot inherit request defaults in the browser.

Verification, with overlapping scopes:

- `/tmp/feat4-attention-connected-current.xml`: 446 pass, two real supervised
  serving cases excluded, 206.87 seconds. Actual trained standard/shared/nested/
  custom package serving and attention, parent lifecycle/public HTTP, legacy
  attention and shared contracts pass. This precedes static refusal hardening.
- `/tmp/feat4-attention-boundary-final.xml`: 436 pass, two real supervised cases
  excluded, 136.38 seconds after refusal/privacy hardening. Actual saved packages,
  strict per-view evidence, malformed/stale parent outputs, completion-before-
  eviction, signed public discovery/matrices, safe refusals and unchanged legacy
  recording behavior pass. Parent process/source execution remains forbidden in
  these tests; trusted numerical and explicitly synthetic lifecycle seams remain
  clearly distinguished from real containment.
- `/tmp/feat4-attention-owned-final.xml`: 28 pass, 71.96 seconds after autocast
  memory accounting. Actual standard/shared/nested/custom observations, per-point
  evidence growth/reuse/invalidation, budget/cancellation, complete strict capture,
  actual occurrence catalog/legacy selection and unchanged pinned trace pass.
- `/tmp/feat4-attention-catalog-http-final.xml`: 411 pass, 8.98 seconds. Includes
  shared positive/negative observation/proof/axis/base64/probability fixtures and
  actual catalog plus interactive completion/disconnect/cancelled-cleanup checks.
- Frontend typecheck/lint and 264 checks across seven focused files pass; exact
  output is `/tmp/feat4-attention-frontend-final.log`. Shared 71 schemas/types are
  fresh. Checks cover worker/session/selection replacement, finite bytes, transport
  identity and authoritative missing model shape alongside architecture regressions.
- Rust API all-target Clippy and new route compilation pass. The actual
  `attention_discovery_is_a_public_authenticated_read` proxy case fails before
  requests at Actix socket setup with PermissionDenied. It is not a passing live
  contract. No Rust SQL/query/schema changed.
- `/tmp/feat4-attention-cuda-deferred.xml`: six runnable CUDA fp32/bf16 standard/
  shared/nested cases skip because hardware is explicitly user-deferred. No GPU
  success or actual allocator enforcement is inferred from CPU/numerical tests.

Earlier nonpassing reports were corrected, not counted as current successes:
`feat4-attention-worker-connected.xml` had four constructor-call errors (keyword-
only pins/profile), with 16 other checks passing. `feat4-attention-catalog-http.xml`
had one test-only tokenizer/vocabulary mismatch and six passes. The current
reports above run the fixed construction and actual bound vocabulary.

Real guardian serving, Rust/network/browser disconnect acceptance and PostgreSQL
contracts remain required open gates. The Chrome MCP connection previously timed
out. Eight of 21 issues remain accepted; no issue was closed from these fixtures.
No subagents, commit or push were used. Continue remaining connected feature UI
and final actual cross-module acceptance. See [serving ownership](modular-serving.md).

Final attention build/audit notes: all 71 Python schemas and generated frontend
bindings are fresh; changed-file Rust formatting, local topic links and whitespace
checks pass. Scoped parsing/compilation and documentation/body-comment audit cover
18 Python files and 189 changed/new functions; unchanged legacy Transformer
methods are outside that audit. The normal Turbopack production build remained
at compilation without progress for several minutes and was interrupted. The
webpack attempt also failed. A diagnostic rerun with the webpack build worker
temporarily disabled exposed `EAI_AGAIN fonts.googleapis.com` for existing
Fraunces, Geist and Geist Mono downloads in `app/layout.tsx`. The original
`next.config.ts` was restored exactly. No production build success is claimed;
logs are `/tmp/feat4-attention-frontend-build.log`,
`/tmp/feat4-attention-webpack-build.log` and
`/tmp/feat4-attention-webpack-inprocess.log`. No font/source workaround was committed.

## 24 October 2026: first administrator authoring pages and data-only form inputs

ISS-65 is in progress at revision 2. Connected `/arch` and `/arch/blocks`
catalog/new/detail pages now use the existing Rust/engine adapters. Browser
private state remounts with the actual session. Catalog pages are bounded and
filter/cursor reads cancel obsolete requests. Source remains editable during an
optimistic save; only captured bytes become saved and newer typing remains dirty.
Concurrent server versions refuse overwrite. Metadata updates use their separate
version and preserve local source. Builtin blocks copy eligible wrappers for the
exact installed revision; custom block/recipe copies verify exact source identity
and hash. Creation retains uncertain outcomes without automatic duplicate retry.

This remains partial authoring UI: source inspection, full validation/acceptance,
dependency selection/snippets, complete history paging/source inspection, diagrams
and navigation remain to connect. Initial history exposes only the first page.
No saved source is labelled numerically valid by the new draft workflow.

The required engine form inputs are now typed: `ArchitectureSchema` includes
actual schema/limits plus prospective installed runtime profile and reported
devices. Explicit `POST /architectures/{id}/declaration` returns safely parsed
saved YAML declarations under exact draft version/hash and administrator identity.
It neither imports dependencies nor constructs/queues a model. Stale/foreign
subjects and invalid YAML refuse with bounded inspection diagnostics. Metadata is
always inspection-only; existing block Python inspection remains supervised.

- `/tmp/feat4-authoring-metadata-first.xml`: 427 pass, 8.71 seconds. Actual signed
  metadata/declaration/catalog routes use isolated SQLite and forbid custom-source
  execution, model construction and tensor loading; no numerical/queue claim.
  Shared new positive/negative schema/default-device/declaration fixtures pass.
- Frontend typecheck/lint and 280 checks across ten focused files pass; exact output
  is `/tmp/feat4-authoring-metadata-frontend.log`. Includes seven draft/copy race,
  hash/resource and UTF-8 checks, actual complete generated recipe-schema decoding
  and stale declaration responses. Earlier 271-check draft run overlaps.
- All 74 Python schemas/generated frontend bindings are fresh. Rust API all-target
  Clippy and the expanded private-route authorization case pass after fixing
  administrative scope on cancel/reorder/delete. Those operations, submission and
  authoring now share the actor-preserving bounded status transport. This repairs
  a concrete inability to control private validation rows through Rust.
- The fresh `training_writes_preserve_queries_actor_scope_and_status` test still
  fails before requests at denied Actix socket setup. Exact log:
  `/tmp/feat4-authoring-rust-forwarding-gate.log`. This is not forwarding success.
  No SQL/schema/migration changed. Browser and actual cross-module acceptance
  remain open, and production font downloads remain DNS-blocked as above.

Root remains the sole implementation owner. No agents, commits or pushes were
added. Continue the remaining authoring/validation UI and later training/model/run
integration; no issue was newly accepted from fixture checks.


## 25 October 2026: frozen-source navigation and browser submission identity

Saved-model and validation graph/diagnostic links now open exact retained text,
including custom imports without catalog rows and invalid/old validation drafts.
The new administrative source readers verify inventory, revision and UTF-8 hashes
without loading tensors or executing source. Shared `RetainedSource` schema/types
bring the contract inventory to 77. Builtin descriptors remain explicitly JSON;
current installed Python never substitutes for absent frozen implementation bytes.

The training UI omitted query request UUIDs required by modular admission. Root
pretrain/SFT/RL/chain submissions, retry and recovery now preserve captured session
identity and an operation UUID. Unknown outcomes reuse the UUID only for unchanged
inputs; concurrent writes and stale post-disposal navigation are suppressed.
Generic validation retries now honor that UUID as well, with replay before the
original history lookup and administration checked again before private output.
Existing generic retry operation names stay compatible; no migration, SQL query
or Rust schema changed. Canonical creation URLs match the feature plan while old
`/new` aliases remain usable.

Available evidence:

- `/tmp/feat4-source-adapters-final.xml`: 457 pass, one real supervised HTTP
  check excluded. Current repository tests are imported into the existing isolated
  offline fixture harness with explicit `-c pyproject.toml`; this is real signed
  ASGI/SQLite/data-only artifact evidence, not TCP/worker containment.
- `/tmp/feat4-source-training-regression.xml`: 47 pass, five process/CUDA cases
  excluded, 100.43 seconds. Actual capture/numerical behavior and admission/preview
  pass; CUDA remains user-deferred.
- `/tmp/feat4-source-submission-frontend.log`: all 428 tests pass; full lint and
  typecheck pass. Final focused checks pass 14 cases across source, submission
  and selection files. Names and scopes are in the log; counts overlap.
- Rust API all-target Clippy and private-route authorization pass. Actual Rust
  query forwarding fails before HTTP requests at denied Actix socket setup.
- Real supervised validation fails with Operation not permitted. Normal in-tree
  source tests fail at the autouse TCP backend fixture. Both TCP/Unix socket
  setup were independently retried and denied. The first offline invocation had
  the wrong pytest configuration; its errors are not counted as passing evidence.
- Current production webpack build fails with generic webpack errors. Browser MCP
  connection times out after 60 seconds. Earlier font DNS failures are recorded
  separately; neither current build nor browser success is claimed.
- The new `scripts/verify_modular_architecture.sh` passes shell syntax and its
  actual `contracts` group. Reports are `/tmp/feat4-acceptance.SByXqa`.
  It records independent command failures and does not skip process/PostgreSQL
  tests or mark completion. The [acceptance map](modular-architecture-acceptance.md)
  lists prerequisites, formal requirements and required connected scenarios.

FEAT-4 remains active at 8/21 accepted issues. Real process, PostgreSQL, connected
browser/Rust/data/gym/bench acceptance, production build and user-deferred GPU
checks remain open. No issue completion, commit or push is claimed.


Final retry regression: `/tmp/feat4-submission-retry-final.xml` passes 44 with one
real supervised HTTP case excluded. Same-actor access loss, replay after input
history pruning and conflicting UUID reuse pass. The first privacy assertion
used an unrelated public actor, correctly receiving missing-history 404; the
final assertion signs the original actor with public scope and verifies 403.
The dedicated `/tmp/feat4-validation-retry-privacy.xml` passes that case.
A deliberate missing-interpreter runner check exits nonzero and still records
the subsequent passing generated-type check in `/tmp/feat4-acceptance.Rni7s4`.
A whole-file documentation audit additionally found two unchanged legacy label
helpers in `api/routes/runs.py` without docstrings. They are outside this change;
the changed-function audit is scoped accordingly and makes no full-file claim.


Final frontend checkpoint: `/tmp/feat4-submission-final-frontend.log` passes 429
checks after malformed successful queue acknowledgements were made uncertain,
retaining the request UUID instead of authorizing an invalid navigation target.
Typecheck and changed-file lint pass. Shared-worktree seed-reader work was
announced separately in Discord and preserved; its final evidence remains with
that workstream. Its reported producer process was not restarted or declared
terminal from an observation of a log file.


## 25 October 2026: concrete build diagnosis and documentation rendering

`/tmp/feat4-build-inprocess-current.log` confirms that the generic current
webpack failure comes from EAI_AGAIN resolving fonts.googleapis.com for the
existing Fraunces, Geist and Geist Mono downloads. A process-local config-loader
override disabled the webpack build worker for error visibility only. Repository
config/dependencies remain byte-unchanged; session 71964 exited 1. No normal build
pass is claimed. Both TCP/Unix socket setup were retried and denied.

`/tmp/feat4-docs-render-current.log` renders all 102 actual documentation pages
through existing DocMarkdown/lib/docs code, including their actual relative-link
resolver; session 80147 exits 0. This is documentation-render/link evidence only,
not a browser or complete Next build. No code or acceptance status changed.
The pending seed/package workstream has been asked for its live tool handle;
no producer restart/terminal claim was inferred from its log. The host-verification
request remains pending and no extra creator ping was sent.


## 25 October 2026: third-turn blocked audit

The same environment blocker persists after three goal turns. AF_INET socket
creation and the runtime's exact Unix socketpair/SO_SNDBUF setup remain denied.
A direct read-only default PostgreSQL SELECT 1 connection attempt fails at
/var/run/postgresql/.s.PGSQL.5432 with Operation not permitted, before executing
SQL. Unix socket creation alone can succeed; earlier general socket-denial
summaries must not be read as proving every AF_UNIX operation is forbidden.
Font DNS/build and browser acceptance remain unresolved, with no human host
verification results received. No already-passing suite was rerun.

Producer session 11956 is documented by its owning workstream, but polling it
here returns Unknown process id. It was not restarted, terminated or declared
stopped. Its final result remains with that owner. The automation goal is marked
blocked pending external verification; Discord FEAT-4 and its unfinished issues
remain open/in progress. The prior creator ping and runnable acceptance command
remain the concrete next action. No feature completion, commit or push occurred.

## 25 October 2026: owning source and rerun-seed continuation

The owning package-writer regression process finished with exit zero: **55 passed,
10 deselected in 1431.22 seconds**. Execution session 11956 is terminal, with no
restart or termination requested. Final reports:
`/tmp/feat4-retained-source-producers.xml` and its sibling `.log`. This selected
matrix includes actual CPU pretraining/SFT/RL updates, self-contained package
publication and saved recovery through explicit trusted in-process seams, plus
parent-publication adapter fixtures. It does not establish guardian containment
or PostgreSQL concurrency. Supervised CPU/CUDA cases were explicitly deselected;
the separate worker attempts below still fail before child execution. No owned
test process remains active.

The independent source run `/tmp/feat4-retained-source-engine.xml` has **466
passes and five failures**. All failures occur at denied socket-buffer setup in
the actual supervised HTTP/package-grant tests, before child code. These are
blocked required checks, not skipped successes. Source routes use actual signed
ASGI/SQLite and inventoried package/publication bytes, with forbidden source/tensor
execution seams. Invalid source, missing/corrupt/linked files and foreign revision
requests are exercised. This does not prove worker containment or PostgreSQL.
Three additional data-only checks in
`/tmp/feat4-retained-source-edge-cases.xml` pass: invalid Python/YAML dependencies
remain readable; builtin root Python bytes use their own source digest rather
than the manifest digest; unavailable builtin source has no installed fallback.

Training rerun reads now capture bearer/abort ownership and finish before editable
controls mount. Failure requires explicit retry instead of enabling defaults.
Session/from/resume replacement releases the old read and form. Reused settings
retain exact recipe pins and SFT/KTO/RAFT modes, remove previous validation
receipts and honor the selected checkpoint override. Regression checks cover
missing/foreign/non-training/failed seeds, zero/false/required declared parameters,
stale preview identity and bypass of unrelated legacy shape/head estimates.

Frontend **435 tests in 43 files**, full lint and typecheck pass in
`/tmp/feat4-source-and-seed-{frontend,lint,typecheck}-final.log`. Test-only typed
fixture mismatches were corrected before the passing typecheck; no runtime
suppression was added. Rust API Clippy and private-route authorization pass in
`/tmp/feat4-retained-source-rust-*`. The documented verification entry point's
`contracts` group passes both freshness checks in
`/tmp/feat4-acceptance.ZEw5sV/status.tsv`. Suites overlap and are not additive.
No real browser, production build, live service, PostgreSQL or GPU pass is implied.
The existing host-verification request remains pending. No issue was accepted.
## 25 October 2026: live startup reports after harness restart

The user started the full Compose application and restarted the harness with
browser/worker access. At 22:12–22:13 UTC, an isolated browser context signed in
as `agents` read all thirteen fresh startup reports through Rust. Every report
returned HTTP 200, verdict `valid`, no diagnostics and five passed stages:
inspection, resolution, construction, capture and numerical. Every durable
execution row is `done` without error, and every catalog revision list contains
an accepted revision with the exact report receipt ID and identity hash.

All profiles are CPU/fp32, train and eval, strict torch export, batch 2, sequence
3, full-prefix, seed 0, torch 2.13.0+cu130. This concrete builtin example domain
is not evidence for other shapes, custom authoring, GPU or training/serving chains.

| Startup subject | Validation run | Matching accepted revision | Unique parameters |
| --- | --- | --- | ---: |
| Builtin nested Transformer | `4d0b0618-8646-4101-9801-5239d1f966d6` | `9e2441a5-c01b-4d74-bef3-0aca13bfd37b` | 22,880 |
| Builtin looped Transformer | `87d5e5a5-90d1-424a-a81c-ea208b420cbb` | `9e0a5701-a09a-445e-84cc-dfd6510b666f` | 22,880 |
| Builtin expanded Transformer | `275d0ce1-5ebe-4657-b7cd-31fb68506bb4` | `e1116d43-9595-4ca9-a888-26f2c14e7851` | 35,456 |
| Builtin standard Transformer | `7b64cf74-cfd8-4943-a06f-44f74b7ce383` | `14fd17d4-d78f-4f03-bdd5-c3cd07820c33` | 35,456 |
| Channel concatenation | `b9bd9a1b-9f87-4198-9638-dac3671f81a8` | `8a96db29-09eb-5eac-9802-2fa537712662` | 0 |
| Residual addition | `94daaf3d-72c4-4842-933c-91b3ccd7e8a8` | `77ebbd3a-d256-57c4-8928-50020d85f3c2` | 0 |
| GELU feed forward | `b0c13413-4854-48cb-af52-e22d4b03001a` | `4a3491bc-0792-5d99-9995-5d1c7f184cc4` | 8,352 |
| Causal RoPE attention | `f8942c61-0131-42b9-9915-3aff08de27f3` | `72927a82-eedd-58c0-9a84-78c379e05c9a` | 4,096 |
| Vocabulary projection | `da5fa7c9-5d1b-40fe-bcd2-45936f741102` | `380401b5-b447-56d0-ac3a-be34a2670698` | 1,024 |
| Layer normalization | `d7eece6f-56d4-48b1-abfd-d38a68f8802a` | `b73bb4c4-b055-5e26-9d14-2fc2b271680c` | 64 |
| Pre-LN Transformer layer | `240f16c9-4810-43dc-a88a-f9a9e756d760` | `cdb5189b-03a7-59db-91e9-68b72dd792f3` | 12,576 |
| Token embedding | `368b6585-4a5a-4402-b670-bbece47e5f00` | `ccda902c-1e69-53b1-af00-fb3f9efc8947` | 1,024 |
| Standard Transformer | `b6c34784-3efe-4459-a340-2bc950b6ba1f` | `231543ca-6c49-5561-854b-e6799befc1a8` | 26,240 |

Retained operation graphs load in both modes for all thirteen reports; recipe
architecture graphs also load in both modes. All 34 supported root reads match
the receipt graph hash. Block artifacts intentionally have no architecture scope;
18 additional architecture requests correctly returned 422. Full report and
cross-check exports are `/tmp/feat4-startup-validation-results-20261025.json` and
`/tmp/feat4-startup-validation-crosschecks-20261025.json`.

Live inspection exposed a presentation defect: block reports initially selected
the unavailable Architecture view. Both durable detail and editor result panels
now derive graph availability from the published block declaration. Blocks open
Captured operations and omit the unsupported Architecture choice; recipes and
saved models retain both views. Session/resource ownership includes this view
availability. The live Standard Transformer block report renders operation nodes,
its selector exposes only Captured operations, and switching to Training retains
the same graph/receipt without an alert. The nested recipe still defaults to
Architecture and opens both repeat-body scopes. Screenshots were inspected:
`/tmp/feat4-startup-block-operations.png` and
`/tmp/feat4-startup-nested-architecture.png`.

Frontend typecheck, full lint and all 435 tests pass with exit zero in
`/tmp/feat4-startup-graph-{typecheck,lint,tests}.log`. No new unit test was added
for this presentation default; actual browser behavior was checked. The broader
production build and cross-module gates are running through the documented entry
point in `/tmp/feat4-acceptance.Ry03wh` (execution session 7610), with explicit
PostgreSQL administration URLs and browser origin localhost:3000. Socket probes
now permit TCP binding and the exact worker socketpair/SO_SNDBUF operation, and
read-only connections confirm both local and Compose PostgreSQL roles can create
databases. Docker socket access remains denied. No issue is accepted from these
startup checks, and no full-suite completion is claimed while the runner is live.

## 25 October 2026: live custom authoring, training and ordinary chat

The actual agents browser created a decorated gated FFN, inspected declarations,
configured width/expansion and queued full CPU/fp32 validation. Width 8/expansion
2 yields 424 parameters in run `0fe3dd9c-082c-437a-a1d0-16b0d5255ee4`. Editing
width to 4 immediately marks that result stale and disables acceptance; the new
run `889e12c5-89a3-40e8-a53e-ea0fb23488cc` passes all stages with 116 parameters.
Block `e04f2cee-324d-42cb-afc2-58f24f1fb91f` accepts revision
`5a13588c-7687-481d-ae9b-32d14b1424e4`, exact source hash
`abfee9adee29efcbf97fc73d98eae82b5628caee089ac385201f042c85745a84`.

A YAML recipe pins that custom block and the actual builtin revisions, ties
embedding/head weights, calls the same core in two branches, adds them explicitly
and nests two fixed repeats. Actual UI declaration/parameter controls, numerical
validation, acceptance and nested graph inspection succeed. Architecture
`a2517b29-63d3-4f13-8822-bcd137b3ad9a` accepts revision
`2440c3df-67b3-4574-a76e-5dff8243855a`, source hash
`f9eb120ee28fd27456081189106966e713bb5883d7f625b845b7005fcc3c6029`, validation
run `551592f0-403e-42ed-8d11-4f556a405856`, receipt
`29942a07b340b748bf46538b4e236747e4832e5d445f125ea2e919091eec2d14`, graph
`c6c2aab5f3242d300f5ff61b981164d50457aef0dbd3d381592e6962392ad355`.
Its CPU/fp32 train/eval B1–2/S1–8 domain has 1,404 unique parameters at vocabulary
320. Screenshot `/tmp/feat4-custom-recipe-nested-branches.png` was inspected.

First actual training `cfb18aa9-0525-4a07-a151-84853fceaaec` failed before an
optimizer update: the supplied SiLU FFN captured successfully but its operation
was missing from supported memory estimates. The estimator now admits SiLU under
the existing four-copy pointwise bound. Unknown operations still refuse, using
Softplus in that negative regression. The real shipped example now checks
captured training admission as well as gradients. Three focused checks pass in
`/tmp/feat4-silu-admission.xml`. The exact vocabulary8192/B8/S8 final preflight
also passes in `/tmp/feat4-live-training-preflight-fixed.log`.

Actual retry `0ac1977d-6e49-44df-97e2-7f53acb3d351` completes 171 optimizer steps
over one read-only selected Wikipedia document, producing model
`a2362c98-e470-40bb-8d2d-a7963f1d5387`, 32,892 parameters and definition
`ae417e4abcab303f9a7c54c41522934b28f02d82515458d271cc6fc57a74885e`.
The custom catalog block was archived after the failed first attempt and before
this retry. Saved metadata/graphs and exact packaged custom Python remain
readable despite that archive; the actual read-only source dialog was inspected.
This is not proof of editing during a queued job or restarting the server during
training. Those remain distinct scenarios.

Fresh context-64/B1 training `d3576bdb-0422-43d5-81ae-0bd20bc5b86f` completes and
produces model `59c4f4af-eb69-4e6f-9134-5b72a4cadb4b`. A temporary ordinary User
account chats through Rust's real WebSocket with that briefly published custom
model: 59 token frames, final done, persisted user/assistant messages, private
definition read 403. The account/chat were deleted and publication reverted.
This tests transport/saved execution, not useful model quality. Earlier ordinary
browser checks also refuse catalog/report/source/graph/run reads and private
validation WebSocket frames. Full exports are
`/tmp/feat4-live-authoring-and-privacy-20261025.json` and
`/tmp/feat4-live-training-serving-evidence-20261025.json`.

The context-64 run exposed a diagnostics crash: legacy charts dereferenced absent
Transformer arrays on semantic model reports. `JobDiagnostics` now validates and
renders bounded parameter-owner measurements and distinct occurrence paths with
shape/dtype/RMS. It preserves actual zero, missing measurements and legacy
figures. ModelReport now records the supplied optimizer step. Four frontend
regressions use an actual captured report and edge cases; one Python regression
checks the step contract. Full frontend 439 tests/typecheck/lint pass. Actual
rerun `59104b7c-7ad3-4f95-b556-25a9ade3c375` completes 171 steps, producing model
`c754dff4-6c02-4462-a333-0c7ae0c4c692`; its real owner/occurrence tables render
at step 170 without alerts. Screenshot
`/tmp/feat4-live-model-diagnostics-fixed.png` was inspected. No unavailable
attention, layer or embedding measurement is fabricated.

Training ArchitectureError failures now retain a stable code and up to 512
characters of explanation, including unsupported captured operator names.
Non-training worker failures retain exception-class-only responses; ordinary
run projections still hide private diagnostics. Real worker packet/privacy
regressions are included in the pending serial engine run.

## 25 October 2026: full-gate execution and final serial rerun

Runner `/tmp/feat4-acceptance.Ry03wh` has real passes for schemas/types, Rust
build/all-target Clippy, explicit Rust/Python HTTP/WebSocket privacy, frontend
type/lint/439 tests and the normal production build, including all documentation
routes and font downloads. Its engine pass was interrupted after 1,127 passes,
16 skips and three supervised RL failures caused by a one-process numerical test
fixture. Production already budgets two processes. The fixture now permits the
worker plus one owned framework helper; the supervised CPU standard RL case
passes. A subsequent negative helper cleanup check failed under concurrent heavy
compilation/swapping and remains required in the final serial run.

The contending engine attempt `/tmp/feat4-acceptance.yd1Uay` was interrupted after
37 passes, three skips and two prototype timeouts. The resident prototype case
passes separately. These interrupted reports are failures/incomplete checks, not
full engine passes. The current isolated engine run is session 26842,
`/tmp/feat4-acceptance.sJZbI1`, with one OMP/MKL thread and Compose PostgreSQL.

The first Rust workspace run fails because native PostgreSQL template1 has a
glibc collation-version mismatch. No host database template was changed. With
the existing Compose PostgreSQL administration endpoint, all 225 Rust workspace
tests pass (one ignored live privacy check, independently passed) in
`/tmp/feat4-rust-workspace-compose-postgres.log`, exit zero.

Both browser fixture scripts initially redirect to login: the anonymous sidebar
unconditionally mounted a profile hook that requested /user/me. NavProfile now
mounts profile reads only for an actual bearer, keyed by that session. Actual
anonymous fixture pages stay public. The authoring script passes editor typing,
read-only source, typed parameters and runtime decoding with no external editor
requests. The graph script passes keyboard/pointer, nested scopes, pagination,
themes, reduced motion and mobile overflow. Logs:
`/tmp/feat4-{authoring,graph}-browser-final.log`. These are synthetic fixture checks,
separate from the genuine captured graphs and service flows above. Final
frontend type/lint passes are being finalized; the production rebuild after this
sidebar-only change is pending until the serial engine finishes. No issue was
accepted, committed or pushed from partial evidence.

## 26 October 2026: reusable tensor recipes and connected CPU acceptance

Live acceptance found that full validation incorrectly required every recipe to
be a causal language model. Tensor recipes now resolve their pinned closure,
construct named input/output boundaries and use the same exact-shape strict
capture/value/gradient checks as tensor blocks. They retain both architecture
and operation graphs, with no invented tokenizer or language-model context.
Causal recipes still require final tokenizer facts. Four actual numerical
regressions pass, including multiple outputs, a repeated shared child, publication
identity and refusal of missing inputs/ranged tensor domains:
`/tmp/feat4-tensor-recipes-focused.xml` (8.67 seconds).

Actual browser→Rust→engine evidence:

| Scenario | Durable identities and measured result |
| --- | --- |
| Invalid draft and concurrent saves | Block `8274b00f-5bf7-4da9-910a-bbfbff6dd5ef` saves invalid Python. Two saves at version 1 produce one 200 and one structured 409; version 2 remains readable. Inspection `8e0d7fb3-e209-4621-abd8-b79fa88bf564` completes invalid with no receipt. |
| Reusable tensor child | Resource `dffd06ae-ac96-484f-9962-667bb95aaa6a`, accepted revision `5068db46-ddd7-4c08-8c4d-87a77b17757e`, hash `3caa0998af83474466c5877d61cd3e2fe6c86335ae351bdc7f0a80050f1ae11c`. Validation `b9c52cc0-ead8-489f-b84c-815449fdde48` passes all five stages at B1/S2/width4, CPU/fp32 train+eval, 116 unique parameters. Two FFN calls share one owner; addition and concatenation return two named outputs. |
| Nested parent | Resource `46c8308a-161e-4dc4-9cb8-b0c9d378e783`, revision `f2a074d8-cf90-442a-9025-ce030e0e089c`, hash `757e82ec7357c52ec764c8916c6fecdba22d1da0538c872b9926b424e6b6d0ff`. Validation `60853583-d6f5-4c3c-9ac6-9d90c1e33f2f` is valid, 1,404 parameters at vocabulary=320, context 256, two nested repeats and branching calls to the shared two-output child. Actual browser navigation opens `root/core` with both output ports. |
| Archive after enqueue and cancellation | Training `81f9330d-c785-4f03-90f3-5f2e57af0128` was acknowledged queued, then its root resource was renamed/archived (metadata version2) before execution. It registered checkpoint `8deb2c8b-561a-4311-b9fc-afe0cef1a9c2` at step 22, then acknowledged cancellation. |
| Exact recovery | `/recover` creates `0fb8b193-3aa3-4295-814e-494fb18bb04c`, using the saved checkpoint and `recover_exact=true`, with no structural override. It completes 420/420 steps, 32,892 parameters, model `6e2a7dfd-a210-4811-8c14-3a67e5fcab03`, unchanged definition `5f93a6ef11c0a5af8783c934421c8ef1ac693a658ffa70b1386609d64174cdb0`. This live check establishes pinned recovery completion; byte-for-byte uninterrupted trajectory comparison remains the automated supervised test's evidence. |
| Validation retention | Deleting the accepted child validation's history returns 200, `/runs/b9c52cc0-ead8-489f-b84c-815449fdde48` becomes 404, but `/validations/...` remains 200 with unchanged identity `a195072246ecbb57138d4888cd0f6260d106505898a9371f9c77353b9abe4378`. |
| Real gym→data demonstrations | Rust assembles 36 math demonstrations in disposable agents dataset `cbb91d23-d08a-467d-9884-6ca992f30cae`; Python services do not call each other. |
| Custom SFT without token-row initialization | Run `c104beb0-00b5-44eb-b6db-52932fc61f19` completes five updates, 1,432 packed tokens / 216 trainable tokens, model `5ed9e0a8-67be-4075-8ae6-c44a18883cd1`, unchanged definition. Warm-start explicitly reports `skipped_unsupported`, initialized IDs empty, skipped IDs5–10. Actual composer displays inherited context 256, vocabulary 8192 and iterations 2 read-only. |
| Custom RL through real gym effects | Run `6ff23b9a-7a6f-460a-831b-35fbcd09b88f` completes one iteration, model `86d5e385-2c4b-4e23-b181-42ff6a33c8f7`, same definition. Two actual math episodes were graded; both incorrect, pass rate/signal/loss0. This is transport/execution evidence, not learning-quality evidence. |
| Standard queued chain | Pretrain `df10a67d-e999-4e20-83d5-462c7c3205cb` → SFT `2ccceef3-a098-4d78-9cb1-2a4a091a653c` → RL `04372789-0651-4931-a2a9-c0c22e9add25` → benchmark `49dea941-ac39-4a5a-a46d-8089abb635af` all complete. Three models share definition `0467f05b7ee9d9cb6235d0897911647d7b7f2068869881915abdc8ea7f712022`. |
| Custom benchmark | `7c49f052-5658-4453-83e7-049c5f6b9810` completes through bench→Rust→engine, twelve four-token samples, 10.8 seconds, context 256/vocabulary 8192/32,892 parameters. Transformer dimensions/heads/layers and ablation are genuinely null. No reference corpus is installed; compression is null and the actual browser says unavailable. Score-card types now retain those nullable facts and omit empty Transformer shape labels. |
| Pending tokenizer and shared-loop training | `b895b1da-f959-45fb-a91a-d129bb0554ce` starts with `tokenizer_id=null`, learns a 512-token vocabulary, completes final preflight before updates, then 349 updates. Saved model `db1b19ca-442f-4557-a350-583bf0e1d1fd`, definition `516b00a12210aa550ed6160a22efb5697f4190f1ca370050364b7d7b36119c0d`, 4,680 unique parameters, saved iterations 2/context64. |
| Ordinary-user compatibility | Existing published legacy `339d57f5-8442-416d-8da7-2b70d43624f0`, new standard `a719ec58-e9dc-4953-bf59-a4e6b93c269a`, looped `db1b19ca-442f-4557-a350-583bf0e1d1fd` and custom-derived-RL `86d5e385-2c4b-4e23-b181-42ff6a33c8f7` each emit final done and persist user+assistant messages for a genuine User account. All private definition reads return 403. Three scratch publications revert 200 and the scratch account/chats are removed 204; the human legacy model's publication is untouched. |
| Builtin UI copy | Actual Copy accepted block→Create draft→Inspect produces resource `01f7ce01-638b-416b-b2d4-67a1ca861437`; declaration-only job `53c81be2-fc1c-4b50-bc48-d9ec0e0ef17a` completes inspection_only and renders controls without accepting a revision. |

The first four-example SFT request correctly failed before optimizer creation:
it could not supply one full context 256 batch. Plain `ValueError` training
contract refusals now retain at most 512 explanation characters for administrators,
instead of only the class name. Serving and arbitrary exception subclasses retain
class-only errors; public run projections still scrub private details. Four real
worker train/generate/error-packet/cleanup regressions pass in
`/tmp/feat4-training-errors-final.xml` (22.14 seconds). A retry interrupted by the
development autoreloader remains recorded cancelled; it is not a successful SFT.
No batch/context guard was relaxed. The 36-example run above genuinely succeeds.

Artifacts: `/tmp/feat4-live-subarchitecture-recovery-evidence-20261026.json`,
`/tmp/feat4-live-sft-rl-chain-evidence-20261026.json`,
`/tmp/feat4-live-ordinary-compatibility-chats-20261026.json`, and inspected
`/tmp/feat4-live-reusable-subarchitecture.png`. Full engine verification remains
session 26842 in `/tmp/feat4-acceptance.sJZbI1`, with no terminal verdict yet. Its
collection predates the final tensor-recipe/error-diagnostic edits; the focused
current regressions above are separate. Latest frontend typecheck/lint/439 tests
pass in `/tmp/feat4-final-20261026-{typecheck,lint,tests}.log`; production rebuild
still waits for the engine gate. GPU remains explicitly user-deferred.

The copied builtin also exercises a genuine delayed-validation race. Job
`d5a4d77a-3e79-46e4-a8c9-2c1aa64eb8eb` is submitted from the UI at layers=1,
dimensions=8/heads=2/FF multiplier=2/vocabulary=32, exact B1/S1 CPU/fp32. While it is
running the browser changes layers to2. The old request completes valid with 840
parameters, but the editor explicitly says stale configuration and disables
Accept validated revision. The copied resource remains unaccepted. The retained
browser tree is `/tmp/feat4-live-delayed-validation-stale.txt`.

Actual occurrence attention for looped model `db1b19ca-442f-4557-a350-583bf0e1d1fd`
discovers two core calls, `[1,0,0]` and `[1,1,0]`, both with two heads. Each explicit
head 0 measurement of `one two three` succeeds with a 7×7 matrix/196 fp32 bytes,
distinct point/evidence hashes and validated sequence maximum 7. Legacy layer
fields remain null. The FFN-only custom-derived-RL model reports an empty point
list and an explicit unsupported reason. These are actual service observations,
not fabricated layer/head assumptions. Export:
`/tmp/feat4-live-occurrence-attention-20261026.json`.

The development engine parent changed from PID 311276 to 331666 during the Python
edits after the archived recovery model was saved. Inherited SFT subsequently
loads that saved model successfully, and ordinary-user serving of its derived
output also succeeds. This establishes a real saved-package/catalog-archive
cross-reload flow. Deterministic uninterrupted-versus-recovered byte equality and
forced-crash cleanup remain the automated process tests' evidence.

Actual browser cancellation of validation `61d015cb-6565-456a-ba67-1a5763ae0cbb`
transitions running→cancelling→cancelled, with durable result null. Its copied
standard block has a bounded larger 32-layer/64-dimensional profile to leave time
for the interaction. Subsequent ordinary numerical validation
`b038405d-f214-403e-8f2c-f757fedb88aa` completes valid after cleanup and renders its
graph, confirming admission remains usable. Browser trees:
`/tmp/feat4-live-validation-cancel.txt` and
`/tmp/feat4-live-after-cancel-validation.txt`.

The current complete validation module passes 23 CPU cases with three deferred
CUDA skips, `/tmp/feat4-validation-module-final.xml`, 135.39 seconds. This rerun
includes all final tensor-recipe changes. Schema/generated-type freshness passes
again in `/tmp/feat4-acceptance.WjF0M3/status.tsv`.

The serial broad engine run reports two supervised pretraining failures, both
`runtime_process_count_budget`: that test had its own one-process policy outside
the earlier NumericalContext fix. Its policy now matches production's worker plus
one owned framework helper; no production ceiling was relaxed. Both CPU cases,
with recomputation off/on and complete real package adoption, pass in
`/tmp/feat4-supervised-pretrain-final.xml` (74.23 seconds). The unchanged failing
reproduction is `/tmp/feat4-supervised-pretrain-diagnosis.xml`; the original broad
report remains retained. Other cases in that broad run are still executing.

## Stopped for tonight at the user's request: 26 October 2026

The user asked for Discord findings and a checklist for the next agent. FEAT-4
and all remaining issue statuses stay open; no commit or push. The current
checklist is at the top of `modular-architecture-resume.md`.

The broad engine run is terminal, exit1: **1,843 passed, 3 failed, 25 skipped,
13 subtests passed**, 3220.46 seconds. Every skip is explicitly user-deferred
CUDA hardware. Catalog/store, PostgreSQL publication/locking, runtime supervision,
saved pretraining recovery and queued RL recovery checks pass. The three failures
are the two separately configured one-process pretraining policies described
above and the last SFT test assertion calling `.get` on an optional `diag: null`
frame. Its actual reference training, checkpoint cancellation, exact recovered
weight hash, tokenizer immutability and orphan checks had already passed. The
assertion now skips absent diagnostic frames while still requiring a real base-KL
measurement. No production behavior or numerical assertion was weakened.

Both pretraining cases pass separately. The final grouped rerun of all three
failures completed with **3 passed in 189.00 seconds**, terminal exit0,
`/tmp/feat4-engine-failures-final.{log,xml}`. No verification process remains
running. The final complete
validation/source-location module passes **24 CPU cases and three deferred CUDA
skips**, `/tmp/feat4-validation-module-final-locations.xml`, 129.61 seconds.
Python syntax diagnostics now retain positive parser line/column under the exact
immutable relative source file; old published reports remain immutable.

Remaining for the next agent: latest normal frontend production build and
acceptance-matrix/Discord issue reconciliation.
The old full-suite command remains a failed report even when its corrected cases
pass separately. Final complete live exports are in
`/tmp/feat4-live-final-acceptance-20261026.json` (17 runs plus two benchmark reports).

Separate pre-existing display-copy findings outside the feature fixes:
`frontend/components/benchmark/corpus-panel.tsx` names the old data/eval corpus
path/script, and `generation-panel.tsx` describes default sampling even when an
API request overrides it. Actual report configs/results are correct. These copy
defects were not changed.
