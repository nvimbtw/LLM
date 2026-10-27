# Modular architecture resume handoff

## Stopped for tonight at the user's request: 26 October 2026

The user asked to wrap up, post findings/checklist in Discord and leave this for
the next agent. **FEAT-4 is unfinished; do not mark it complete from this handoff.**
Issue statuses are unchanged: eight accepted, twelve in progress, ISS-68 open.
No commit or push. Preserve the large existing dirty worktree.

### Completed tonight

- [x] Read and cross-check all 13 startup validations, their durable jobs,
  acceptance receipts and retained graphs: all genuinely valid, CPU/fp32 only.
- [x] Fix block graph defaults; bounded SiLU memory admission; real owner/occurrence
  diagnostics and step numbers; anonymous sidebar profile requests; tensor-recipe
  validation; private training error details; syntax-error source positions; and
  nullable custom benchmark shape labels. Details and exact files are below and
  in the evidence ledger.
- [x] Run actual custom authoring/acceptance, reusable two-output child/nested
  parent, stale parameter validation, optimistic-save conflict and builtin copy.
- [x] Verify archive-after-enqueue, checkpoint22 cancellation and exact recovery
  to420 steps, plus retained validation after history deletion and package loading
  across a real development-server reload.
- [x] Run real SFT without token-row initialization, RL through Rust/gyms,
  custom benchmarks, the standard pretrain→SFT→RL→benchmark chain and a looped run
  that finalizes its pending512-token vocabulary.
- [x] Complete ordinary User chats for legacy/standard/looped/custom-derived-RL;
  private definitions403. Restore temporary publication changes and remove the
  scratch account/chats. The application queue was verified empty.
- [x] Measure two actual loop attention occurrences; custom FFN-only attention
  is explicitly unsupported. Cancel a running validation from its page and then
  complete another numerical validation, confirming admission is reusable.
- [x] Rust workspace build/all-target Clippy, 225 tests and separate ignored live
  Python privacy seam pass. PostgreSQL tests use existing Compose PostgreSQL;
  native template1's collation mismatch was not modified.
- [x] Frontend typecheck/lint/439 tests, both browser fixture scripts and current
  schema/generated-type freshness pass. A normal production build passed before
  the final UI/docs edits; the latest production rebuild is still pending.
- [x] Broad engine run finished: **1,843 passed, 3 failed, 25 user-deferred CUDA
  skips, 13 subtests passed**, 3220.46 seconds. All three failures were test bugs:
  two separate pretraining fixtures allowed one process despite production's
  worker-plus-helper bound; SFT's final assertion called `.get` on optional
  `diag: null`. Fixes preserve the original numerical/publication assertions.
  Both pretrain cases already pass in their focused rerun. Do not describe the
  original full report as passing.
- [x] Current complete validation module: **24 passed, 3 deferred CUDA skips**,
  `/tmp/feat4-validation-module-final-locations.xml`, 129.61 seconds. Four real
  worker/private-error tests pass, `/tmp/feat4-training-errors-final.xml`.
- [x] The final grouped rerun of **all three** broad-suite failures finished:
  **3 passed in 189.00 seconds**, `/tmp/feat4-engine-failures-final.xml` and
  `/tmp/feat4-engine-failures-final.log`, terminal exit0. It confirms recomputation
  off/on pretraining publication and actual SFT reference/cancellation/exact
  recovery/base-KL telemetry. No verification process remains running.

### Next agent checklist

- [ ] Run the final normal `bun run build` from `frontend/`, after deadline/process
  checks finish. Existing Google font downloads need network access. No build
  bypass was introduced. Do not repeat heavy parallel compilation during worker
  deadline tests.
- [ ] Reconcile the acceptance matrix with the actual evidence below. Its old
  "remaining live proof" table predates tonight's successful scenarios. Audit
  every ISS-54/55/56/58/59/60/61/62/64/65/66/67 gate, then ISS-68/FEAT-4. Their
  latest discussions were read; open items were genuine process/database/browser
  proof, now largely supplied. Close only after final checks are established.
  Keep the failed broad report and remediation reports separate; never claim a
  clean full-suite exit from the failed original command.
  All three original failures are resolved by the passing grouped rerun. Further
  engine checks use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1` and the documented
  disposable `LLM_ENGINE_TEST_ADMIN_URL` on localhost:5433.
- [ ] Reconcile final documentation/test counts, review the diff and update
  Discord. GPU arithmetic/allocator checks remain explicitly user-deferred;
  missing GPU hardware alone does not hold downstream implementation open.
- [ ] No commit/push unless authorized. Separate baseline display-copy findings:
  benchmark corpus help still names the old data/eval path/script, and generation
  help hardcodes default sampling even for non-default API requests. Actual report
  configs/results are correct; these unrelated copy defects were not changed.

Primary evidence: `/tmp/feat4-acceptance.sJZbI1/engine.xml` and `engine.log`;
`/tmp/feat4-live-final-acceptance-20261026.json` (17 complete run snapshots and two
benchmark reports); ordinary compatibility/attention exports listed in the
evidence ledger. Rust final log:
`/tmp/feat4-rust-workspace-compose-postgres.log`. Frontend final logs:
`/tmp/feat4-final-20261026-{typecheck,lint,tests}.log`. Latest freshness report:
`/tmp/feat4-acceptance.WjF0M3/status.tsv`.

## Evidence accumulated: 25–26 October 2026

### Latest connected evidence: 26 October 2026

The serial engine run below is terminal with the three failures documented above.
It reported two pretraining test-policy failures: that separate
fixture still allowed only one process. It now matches production's two-process
bound, and both CPU cases pass in `/tmp/feat4-supervised-pretrain-final.xml`
(74.23 seconds). Production admission is unchanged. Preserve the original failed
report and the remediation result separately. Current additional fixes are tensor-recipe validation dispatch,
bounded private plain-ValueError training diagnostics and nullable benchmark
Transformer labels. Four tensor-recipe regressions and four real worker/private
error regressions pass. Latest frontend typecheck/lint/439 tests pass in
`/tmp/feat4-final-20261026-{typecheck,lint,tests}.log`. The complete current validation
module's final source-location rerun passes 24 CPU cases with three user-deferred
CUDA skips as recorded above. Final production build remains pending. The broad
suite's collection predates the final tensor/error/source-location test additions.

Actual live flows now include a two-output reusable tensor child and nested
causal parent, archive after enqueue, registered checkpoint 22 cancellation and
exact recovery to 420 steps, retained validation after history deletion, SFT with
explicit unsupported token-row initialization, RL through two real graded math
episodes, custom benchmarking and a complete standard pretrain→SFT→RL→benchmark
chain. Looped training with an unset tokenizer builds 512 tokens and completes 349
updates/4,680 unique parameters. Ordinary User chats for legacy, standard, looped
and custom-derived-RL all complete and persist messages; definitions stay 403.
Scratch publications/account/chat changes are reverted. All detailed IDs,
profiles, failures and limitations are in the latest evidence-ledger section.

Builtin UI copy/inspection works. Copied block
`01f7ce01-638b-416b-b2d4-67a1ca861437` also verifies the real delayed-result race:
validation `d5a4d77a-3e79-46e4-a8c9-2c1aa64eb8eb` submitted layers=1, the browser
changed to layers=2 while running, and the completed valid/840-parameter result
is labeled stale with acceptance disabled. It remains unaccepted. Actual looped
attention measurements for both core call paths return distinct 7×7 proofs and
null legacy layer fields; the FFN-only custom model reports no attention method.

The engine development server reloaded during Python edits: earlier parent
PID 311276 was replaced by331666. The archived saved custom model was subsequently
loaded successfully for inherited SFT and serving. The cancelled retry during
that reload is recorded separately; no interrupted job is treated as a pass.

The user restarted the harness and resumed verification. **FEAT-4 remains
unfinished.** Browser access, local TCP binding, worker socketpair/SO_SNDBUF and
PostgreSQL connections now work. The prior blocked audit is historical. The
completed broad engine verification was session 26842, reports
`/tmp/feat4-acceptance.sJZbI1`, with OMP/MKL threads limited to one and disposable
databases created through Compose PostgreSQL. It exited1; its three failures and
remediation are documented above. Work is paused at the user's request for tonight.
Issue status stays 8/21 accepted, twelve in progress, ISS-68 open.
No issue was closed from adapter evidence. Root remains the sole agent; no
commit, push or deployment was performed.

All thirteen new startup validations are genuinely valid: nine builtin blocks
and four recipes, five passed stages each, no diagnostics, durable jobs done and
matching accepted receipt/revision identities. CPU/fp32 train+eval example domain
only. Both modes of all retained operation graphs and recipe architecture graphs
read successfully through Rust. The live block-report default incorrectly chose
Architecture; both detail/editor panels now choose Captured operations and hide
the unsupported choice for blocks. Actual browser inspection confirms rendered
block operation nodes, mode switching and nested recipe scope navigation.
Frontend typecheck/lint/435 tests pass. Full report identities, exports and
screenshots are recorded in the latest evidence-ledger section. Training chains,
privacy, forced termination/recovery and deferred GPU gates remain separate.

Live authoring now also covers a custom gated FFN and nested branch/merge recipe.
Block revision `5a13588c-7687-481d-ae9b-32d14b1424e4` and architecture revision
`2440c3df-67b3-4574-a76e-5dff8243855a` were accepted through the actual UI. Width
edits invalidate acceptance and change counts from 424 to 116. The full recipe
has 1,404 parameters with the 320-token example vocabulary. Real pretraining with
the existing 8,192-token vocabulary produces 32,892 parameters, 171 updates and
complete saved packages; the block remains archived and its retained source
still opens. Ordinary-user WebSocket chat with the context-64 output completes
and persists both messages while private definition reads return 403. Temporary
publication/account/chat changes were reverted.

Live checks found and fixed three additional gaps: the supplied SiLU FFN was
unsupported by the memory estimator; custom diagnostics crashed legacy layer
charts and omitted their optimizer step; anonymous sidebar profile requests
redirected public markup fixtures to login. SiLU has bounded pointwise admission,
unknown operations still refuse; typed owner/occurrence diagnostics retain actual
steps and missing values; anonymous navigation mounts no profile reader. The
custom report renders at actual step 170 in run
`59104b7c-7ad3-4f95-b556-25a9ade3c375`. Training domain failures now preserve a
bounded explanation/code, while serving errors retain generic exception classes.

Runner 7610 is terminal: schemas/types, Rust build/Clippy, explicit Rust/Python
privacy, frontend type/lint/439 tests and production build pass. Its engine pass
was deliberately interrupted after three RL helper-budget fixture failures;
the fixture now matches production's two-process bound. A contending rerun in
`/tmp/feat4-acceptance.yd1Uay` was also interrupted after prototype timeouts during
heavy compilation/swapping; neither is a full pass. Rust workspace tests initially
hit the native PostgreSQL template1 collation mismatch, then passed (225 tests,
one ignored) against Compose PostgreSQL in
`/tmp/feat4-rust-workspace-compose-postgres.log`; the ignored live privacy test
passes separately. Host database templates were not altered.
Both browser fixture scripts pass after the sidebar fix, separately from the
actual service evidence. Final type/lint logs are
`/tmp/feat4-final-frontend-{typecheck,lint}.log`. The final production rebuild after
the sidebar-only change remains pending until the engine run ends.

Frozen source now opens directly from saved models and validation
graphs/diagnostics, including draft-result panels and imported models without
catalog rows. Administrator-only Rust/engine readers return bounded hash-checked
`RetainedSource` text, without importing Python, loading tensors or following
catalog heads. Builtin descriptors remain clearly labeled JSON; no installed
implementation is substituted for absent packaged Python. Schema/type coverage
is now 77. The source hook owns the actual session/resource/revision/definition.

Review found and fixed a real submission break: browser pretrain/SFT/RL/chain,
retry and recovery omitted query `request_id`, which modular admission requires.
`JobSubmission` now owns a captured session/UUID and prevents concurrent writes;
unknown unchanged retries reuse that UUID, changed input gets a new one, and
navigation/session disposal suppresses late acknowledgements without cancelling
server work. Composers/header/queue retries share this owner. Canonical authoring
creation routes now include `/arch/new-arch` and `/arch/blocks/new-block`, with
the existing `/new` routes preserved as aliases. The RL composer is admin-gated.
Generic validation queue retries now honor supplied UUIDs as well, replay before
original history lookup and recheck administrative scope before private output.

Current verification:

- `/tmp/feat4-source-adapters-final.xml`: 457 pass, one actual supervised HTTP
  case excluded. Actual signed ASGI/SQLite readers, retained-source integrity,
  invalid/missing/linked bytes, publication and shared contracts pass. The prior
  `/tmp/feat4-source-adapters-current.xml` has 457 pass and one actual process
  failure at denied socket setup; it is not a passing report. Normal in-tree
  `/tmp/feat4-source-live.xml` has ten fixture setup errors at denied TCP sockets.
  The first offline run also had a pytest configuration error; the final run
  explicitly uses `-c pyproject.toml` and imports current repository tests.
- `/tmp/feat4-source-training-regression.xml`: 47 pass, five process/CUDA cases
  excluded, 100.43 seconds. Includes actual numerical captures plus training
  admission/preview contracts; no real guardian containment claim.
- `/tmp/feat4-submission-retry-final.xml`: 44 pass, one actual supervised HTTP
  case excluded after the validation retry fix. The new test covers replay after
  original history deletion, conflicting UUID reuse and same-actor access loss.
  Its first version used a different public actor and expected 403 instead of
  the correct missing-history 404; the final test signs the same actor's changed
  scope. `/tmp/feat4-validation-retry-privacy.xml` independently passes that case.
- `/tmp/feat4-source-submission-frontend.log`: 428 tests pass. After adding
  malformed queue-acknowledgement refusal, `/tmp/feat4-submission-final-frontend.log`
  passes all 429 tests. Typecheck and changed-file lint pass; full lint passed
  at the preceding checkpoint. Final focused report is
  `/tmp/feat4-source-submission-focused-final.log`. Tests cover frozen-source
  identity and queue UUID/session/unknown-outcome/disposal transitions. A separate
  shared-worktree continuation announced a training rerun-seed race fix in Discord;
  preserve its edits and use its own final verification/handoff for that work.
  Its package-producer run was reported active without a pollable handle in this
  session; no claim is made that it stopped or completed.
- Rust API all-target Clippy and `private_routes_require_administration` pass.
  Actual `training_writes_preserve_queries_actor_scope_and_status` fails before
  requests at denied Actix socket setup. No SQL/schema/query changed.
- Production webpack build failed with generic webpack errors:
  `/tmp/feat4-source-submission-build.log`. Browser MCP `list_pages` timed out
  after 60 seconds. No browser or production-build pass is claimed. Both TCP
  and Unix socket creation still fail with Operation not permitted.
- `bash -n scripts/verify_modular_architecture.sh` and its `contracts` group pass;
  `/tmp/feat4-acceptance.SByXqa/status.tsv` retains both freshness statuses.
  The deliberate missing-interpreter check returns failure while still recording
  the later passing type check in `/tmp/feat4-acceptance.Rni7s4`.
  Python parsing/function documentation and whitespace checks pass.

See [acceptance map](modular-architecture-acceptance.md). The new grouped script
runs the real module gates without silently excluding supervised/PostgreSQL
checks. It records failures separately and does not equate exit zero with feature
completion. Current UI/source/submission implementation is ready for live review;
the remaining required process/database/browser and connected scenarios still
need an environment permitting their sockets. GPU remains user-deferred. Keep
the full feature scope and audit every ISS-48–68 gate before completion.

### Shared-worktree source and rerun-seed verification

The source cross-check `/tmp/feat4-retained-source-engine.xml` has 466 passes and
five failures, all at denied socket-buffer setup before actual worker execution.
`/tmp/feat4-retained-source-edge-cases.xml` adds three data-only checks: invalid
Python/YAML dependencies remain readable, builtin root snapshots expose a raw
Python hash distinct from manifest identity, and absent builtin source cannot
fall back to installed files. Root YAML is retained as inventoried
`architecture.yaml`; package verification checks its exact revision hash and
parsed recipe. Source transport checks cover captured token/signal and mismatched
revision/definition/hash refusal. Rust API all-target Clippy and private-route
authorization pass in `/tmp/feat4-retained-source-rust-*` reports.

The rerun-seed fix is now implemented. `lib/training-seed.ts` reads the exact
execution using captured bearer/abort ownership, preserves pins and parameters,
clears old validation receipts and applies an explicit checkpoint override.
`TrainComposer` keys its private form by actual token/from/resume selection and
loads settings before editable controls mount. Failure shows explicit retry;
replacement disposes the read so a late seed cannot overwrite another draft.
`trainConfig` now retains recorded SFT/KTO/RAFT modes instead of overwriting them.
`/tmp/feat4-training-seed-checks.log` passes 18 focused cases. Typecheck and scoped
lint pass; test-only Record/TrainConfig and recipe-fixture type mismatches were
corrected before the successful typecheck. Three additional form regressions
cover declared required/zero/false inputs, stale preview identity and freedom
from hidden legacy shape/head restrictions. The complete updated library run
passes **435 tests across 43 files**, with full lint and typecheck in
`/tmp/feat4-source-and-seed-{frontend,typecheck,lint}-final.log`.

The owning package-writer regression process finished with exit zero: **55 passed,
10 deselected in 1431.22 seconds**. Execution session 11956 is terminal; no
restart or termination was requested. Reports:
`/tmp/feat4-retained-source-producers.log` and
`/tmp/feat4-retained-source-producers.xml`. This matrix includes actual CPU
pretraining/SFT/RL updates and saved-package recovery through explicit trusted
in-process seams, plus parent-publication adapter fixtures. It does not establish
real supervised containment or PostgreSQL concurrency. Supervised CPU/CUDA cases
were explicitly deselected; the separate actual worker attempts still fail at
denied socket setup. These suites overlap preceding reports and must not be
summed. No issue acceptance or feature completion follows from adapter, isolated
numerical or static rendering checks. No owned test process remains running.

### Follow-up acceptance audit: 25 October 2026

The previous goal turn made implementation progress. This follow-up recovered
concrete evidence behind the generic production-build failure:
`/tmp/feat4-build-inprocess-current.log` reports EAI_AGAIN for
`fonts.googleapis.com` and refuses Fraunces, Geist and Geist Mono. The diagnostic
used a process-local Next config-loader override to disable only its webpack
build worker; no repository configuration or installed dependency was edited.
Exec session 71964 terminated with exit 1. It is not a normal production-build
pass. Both AF_INET and AF_UNIX socket creation remain denied.

All 102 actual documentation pages also render through the existing
`DocMarkdown` and `lib/docs` code with successful relative-link resolution:
`/tmp/feat4-docs-render-current.log`, session 80147, exit 0. This validates current
Markdown rendering/links; it does not replace Next's production build or browser
acceptance. No additional feature gate was closed.

The separately announced seed/package continuation has been asked in Discord
for its live exec/cell handle and final report. A growing producer log is not a
pollable process handle; it was neither restarted nor declared stopped here.
No human host-verification results have arrived at this checkpoint. Keep the
feature active: full completion still needs genuine worker/PostgreSQL/Rust/
browser flows and the production build; the existing host-side request remains
pending. Do not rerun already passing checks merely to restate this blocker.

### Blocked audit: 25 October 2026

The previous goal turn made verification progress by identifying the exact font
DNS failure and rendering the current documentation. This third consecutive
turn revalidated the same final-acceptance blocker. AF_INET creation remains
denied. AF_UNIX creation itself can succeed, but the exact runtime socketpair/
SO_SNDBUF setup is denied. A direct read-only SQLAlchemy/PostgreSQL SELECT 1
attempt fails connecting to /var/run/postgresql/.s.PGSQL.5432 with Operation not
permitted, before any statement or database mutation. The production-build
EAI_AGAIN and browser connection gates remain unresolved; no human host results
have arrived in Discord.

Producer handle 11956 is now documented by its owning workstream. Polling it
from this session returns Unknown process id; it is inaccessible here, not proof
that the owning process stopped. It was not restarted or terminated. Preserve
that workstream's eventual report. Current seed verification is 432 frontend
checks in its own report, as recorded above; no fresh rerun was started.

The automation goal is marked blocked rather than repeatedly consuming turns
on unchanged acceptance restrictions. The creator was already pinged with
`bash scripts/verify_modular_architecture.sh` and the acceptance prerequisites.
Resume when host results or an environment permitting the required sockets,
PostgreSQL, browser and font fetches is available. Keep all current code and the
full original acceptance scope. No issue or feature is marked complete.

## Previous continuation: 24 October 2026

### Latest checkpoint after the return-home status request

Status remains **8/21 accepted issues (38.1%)**, 12 in progress and final
acceptance open. The implementation estimate given to the user was approximately
88% (85–90%), explicitly a rough code estimate rather than verified acceptance.
The user explicitly requested pausing after this checkpoint to preserve the
remaining weekly allowance. Resume only when requested; the feature is unfinished.

ISS-66 is now in progress. Both training layouts select exact accepted recipe
revisions, render declared parameters and explicitly request an engine-owned
prospective profile. `/architectures/revisions/{revision_id}` and its
`/training-preview` POST read immutable metadata without constructing a model or
creating validation evidence. The two new shared DTOs bring schema coverage to
76. Actual vocabulary facts are optional during preview; pending tokenization
never receives a fabricated build identity. Final job preflight remains required.
Reusable presets use a version-two envelope, preserve pins and strip old
validation receipts. SFT/resume serialization removes fresh-architecture fields.

Saved modular models expose scalar definition identity and effective context
capability to ordinary model/chat views; administrator-only details read retained
definitions and graphs. Unknown parameter counts remain unavailable and measured
zero remains zero. Chat uses multiplier one unless actual model metadata supports
extension. Standalone numerical validation now scopes and restores CUDA TF32
settings to the requested profile, including exception paths. Hardware CUDA
verification remains explicitly user-deferred.

Verified reports retained under `/tmp/`: `feat4-training-preview-boundary.xml`
(458 passed), `feat4-model-public-definition.xml` (25 passed),
`feat4-validation-math-policy.xml` (17 passed, two hardware CUDA skips),
`feat4-training-selection-connected-frontend.log` (typecheck/lint and 417 tests,
39 files), and `feat4-training-preview-rust.log` (offline SQLx API Clippy and
private-route authorization). Suites overlap; do not add these counts. The
checkpoint's additional shape-text boundary regression passes with 249 focused
frontend checks, typecheck, changed-file ESLint and whitespace review. Shape text
is capped at 256 characters before splitting; existing rank/element limits remain.

Next implementation: connect graph/diagnostic locations to verified retained
source bytes, including imported models without catalog rows. Do not substitute
current installed code or mutable catalog heads for frozen source. Current
revision-query links select exact immutable catalog history but do not solve
missing catalog sources. Review training submission session/idempotency ownership,
add meaningful UI domain regressions, and update endpoint/contract topic docs.
Live frontend→Rust→engine, genuine supervised worker, PostgreSQL and browser
acceptance remain unproven because local socket access was denied. The production
frontend build remains blocked by Google font DNS access. No feature completion,
new issue acceptance, commit or push is claimed.

The dated continuation notes below describe earlier checkpoints; where counts or
connected-surface descriptions differ, the checkpoint above supersedes them.

The user resumed the full FEAT-4 goal. Preserve all existing tracked/untracked
work. Root is the sole implementation owner; do not spawn agents without a new
explicit delegation instruction. No commit or push was made in this continuation.
Eight of 21 issues remain accepted; genuine process/database acceptance,
optional cache integration, remaining connected UI and full feature acceptance
remain unfinished. Saved-definition SFT, RL, full-prefix serving and occurrence-aware attention API/UI are connected.

Latest authoring continuation: ISS-65 and ISS-67 are in progress (revision 2), with root
as the sole owner. `/arch` and `/arch/blocks` catalog/new/detail pages call Rust
adapters, preserve session/resource identity and retain invalid source. The draft
owner preserves typing during saves and refuses concurrent overwrite. Metadata
updates retain local source. Exact custom copies verify immutable source hashes;
builtin blocks use the eligible wrapper endpoint. Creation does not automatically
retry ambiguous outcomes. Saved-source declaration inspection, parameter/tensor/
profile controls, full validation and exact-receipt acceptance are now connected.
Dependency snippets pin explicitly chosen immutable revisions; history is paged
and source inspection never replaces local draft bytes. Validation results reuse
the shared retained graph renderer. See [authoring topic](../frontend/architecture-authoring.md).

The engine now supplies typed `ArchitectureSchema` (prospective installed profile,
devices, recipe schema and limits) and explicit exact-draft YAML declaration reads:
`POST /architectures/{id}/declaration` → `RecipeDraftInspection`. This is pure
bounded parsing, not source imports, construction, resolution or numerical evidence.
Block source inspection remains supervised/queued. All 74 schemas/types are fresh;
`/tmp/feat4-authoring-metadata-first.xml` passes 427 actual metadata/catalog and
shared contract checks. Frontend typecheck/lint and 280 focused checks pass in
`/tmp/feat4-authoring-metadata-frontend.log`; seven initial draft/copy race/identity
checks are included. No browser/live acceptance or production build pass is implied.

Tracing validation controls found missing signed administration scope on Rust
cancel/reorder/delete. These now use the shared bounded `upstream.admin_json`
transport with verified actor and exact status; catalog/submission reuse it.
Rust API all-target Clippy and the expanded private authorization test pass.
The fresh actual forwarding attempt remains socket-blocked; see
`/tmp/feat4-authoring-rust-forwarding-gate.log`. No SQL/schema changed. Continue
with live browser/service acceptance. `/arch/validations/{id}` now has explicit
cancel/retry/delete, exact receipt/profile details, bounded replay and serialized
REST reconciliation. Queue/history labels distinguish all verdicts. Actual-token/
access replacement clears private queue state and suppresses late notices. Rust
socket cancellation also preserves verified actor/administration scope; Clippy
and three reader unit tests pass, without a live socket forwarding pass.

The latest complete frontend typecheck/lint/unit run passes **391 tests in 37
files**, `/tmp/feat4-authoring-validation-complete-unit.log`. Tagged-union defaults
and engine-discovered strict guards were corrected in request/report matching.
An actual nested CPU capture also passes frontend decoding, report coherence and
static React graph rendering: 712 parameters, eight strict guard records and
12 root nodes. JSON/HTML/log artifacts start `/tmp/feat4-real-validation-ui`.
This does not certify browser interactions, real supervision or live Rust/engine
acceptance. Direct frozen-source links from graphs/diagnostics, ISS-66 training/
model integration and the recorded live gates remain unfinished.
No issue was newly accepted; FEAT-4 remains active.

Latest SFT continuation supersedes the unfinished-adapter notes below. The
saved-definition SFT queue/worker/package path is connected: private corpus/mask
identity and tokenizer derivation, durable parent preparation authority, known/
unknown marker history, capability-owned warm-start, admitted original frozen
reference and bounded diagnostics, serialized exact next-batch recovery and
explicit final skipped-batch accounting. Public `/sft` freezes saved pins with
actor/admin/request checks; Rust `/runs/sft` forwards them. Copied original pins
remain immutable across derived-tokenizer recovery, which requires exact
original-base/checkpoint byte proof. No issue was newly accepted.

Current reports: 16 numerical SFT cases pass, plus three strengthened genuine
in-process interruption/recovery cases; 104 final queue/HTTP/authority checks and
138 current library/data/trace checks pass (one unavailable dataset-service check
excluded). Earlier 632 parent/contract and 145 wire checks overlap. Frontend
typecheck/lint, 214 decoder tests and all 67 schema/generated freshness checks
pass. Rust build/Clippy/changed-file formatting and private authorization pass;
real forwarding, supervisor and PostgreSQL checks remain socket-blocked. Exact
reports and distinctions are in the final dated evidence entry and
[SFT ownership contract](modular-sft.md).

The final SFT authority smoke passes 24 checks and shared-pretraining regression
passes 23 with four process/CUDA exclusions. ISS-60 and its discussion were read
and it is now in progress (revision 2). RL policy/rollout foundations pass actual
standard/shared/custom eval-with-grad and ragged-prefix checks; `evaluating`
preserves modes without disabling autograd. The typed worker gym-effect client
serializes threads, bounds/cleans grants and latches uncertainty without retry.
The parent/worker RL continuation below supersedes the earlier foundation status.

RL reports: 199 foundation checks pass; the final effects/wire subset passes 154.
The shared session also now drains parent effect/publication callbacks before
grant retirement/admission release and prevents new callbacks during closure;
timeout quarantines rather than certifying a stopped external write. Both real
callback-cancel process cases remain socket-blocked. Current code compiles and
whitespace is clean. All test handles are terminal. Details are in the final
dated ledger entry and `docs/llm-engine/rl.md`.

Parent-authorized bounded Rust gym dispatch, durable issued-resource ownership,
the supervised RL backend, final policy/reference preflight and complete-iteration
packages/recovery are now connected. Actual standard/shared/custom numerical
updates and saved recovery pass. Recovery restores the original frozen reference,
optimizer, global/task/sampling RNG and curriculum without repeating placement;
new jobs receive fresh gym authority. Known cleanup uses separately journaled
identities, and uncertain effects remain fenced across restart. Admission bounds
parent memory, per-job journals, root inventory and private package copies.

SFT waiting chains now finalize pending inherited tokenizer pins using the actual
registered source plus durable parent preparation proof, without changing the
original pin. A regression prunes the predecessor job and archives its root before
the SFT successor runs. RL metadata conservatively marks marker history incomplete:
later SFT cannot reset rows learned during unmeasured RL contexts/actions.

Latest evidence: 60 shared SFT/chain checks, 49 strict gym seed/group checks and
two actual marker/curriculum numerical checks pass. Earlier 8-case RL numerical,
4-case exact/curriculum recovery, 101 admission/publication, 89 queue/submission
and 6 startup checks also pass with overlapping coverage. Rust build/strict Clippy,
frontend typecheck and all 67 generated schema/type checks pass. Real RL direct
and queued CPU supervision still fails before child execution at denied socket
setup; PostgreSQL contracts fail at denied connections. CUDA remains user-deferred.
No new issue was accepted. The dated evidence entry records exact reports.

Next implement serving/scoring and capability/profile-backed caches. Do not enable optional
cache behavior merely from method presence; all declared models currently use
full-prefix rollout fallback. Continue serving/scoring and connected UI afterward.
No subagents, sudo or Docker. Preserve all work; no commit/push was made.

Latest serving continuation supersedes that next-step note: ISS-61 is now in
progress. `/chat`, `/chat/stream` and `/measure` select a reusable admitted child
for modular packages. Canonical generation/UTF-8 handling lives in model/generation;
the backend lives in orchestrate/inference_worker to preserve layering. Parent
state is in api/serving. Saved definition/receipt/graph reads are administrative,
data-only and independent of live source rows. Generic summaries no longer guess
the legacy five-field shape. No optional declared cache/extension is enabled.

Final training profiles now capture batch one through the configured batch so
one-row serving/scoring has genuine receipt coverage. Older fixed-larger-batch
draft packages refuse that unsupported serving domain. Weights-only serving does
not load optimizer/RNG state. Meta admission precedes real construction, and
bounded parent verification/archive/grant copies observe disconnect cancellation.
Completed result bytes are consumed before queue-priority cleanup removes grants.
Final incomplete UTF-8 bytes flush as text without changing generated-ID usage.

Current serving evidence: five actual saved-model worker cases pass; 590 parent/
metadata/contracts/package checks pass with one actual supervised case excluded;
87 fresh inference/parent/shared training checks pass with one process exclusion;
five fresh pretraining/saved-replay numerical checks pass after the profile change.
Frontend typecheck/lint and all 229 selected architecture/graph/decoder/HTTP/owner
checks pass. All 67 schemas and generated types are fresh; final parsing,
compilation, 98-function documentation/body-comment and local-link audit pass.
Actual CPU serving fails at denied SO_SNDBUF before child
launch; one runnable CUDA serving case skips as user-deferred. An intermediate
long regression mixed old imports with files edited during its run (54 passes,
seven setup errors); it is not a passing report. Fresh checks above supersede
the affected parent/numerical subsets, not an unrun full-suite claim.

Attention API/UI is now connected; see the later attention continuation. Next: optional validated caches,
connected UI and requirement-by-requirement acceptance. Browser MCP connection
also timed out after 60 seconds. Real Rust/network/guardian/PostgreSQL/browser
gates remain open; no issue was newly accepted. Preserve all uncommitted work.

Automatic occurrence planning is now connected to both meta and real preflight.
`captured-occurrence-lifetimes-v1` partitions complete ordinary strict capture
using engine-owned annotations, keeps all leaf output backing storage, retains
unselected work and reserves the largest replay region. Output observation runs
with gradients enabled, so a cheaper no-grad branch cannot hide training storage.
All ports count even if unused by export, including aliased larger bases.
Unknown operations/stale partitions/nonfitting budgets refuse; no layer formula
or opaque checkpoint graph is used. Null chooses the fewest fitting prefix calls.
The standard composite is one region and may offer no estimated saving.

The legacy explicit-position boundary now supports shape-only meta tensors using
the same tensor assertion as capture; it still checks real values eagerly.
Builtin implementation identities consequently changed. All five copyable YAML
examples now pin the new revisions, including references inside stack templates.
No pinned numerical trace was rerecorded.

Exact recovery keeps its retained profile/storage selection under a changed
allocator ceiling. Missing/stale selection refuses. Selected CUDA replay still
requires `compile_model=false`; automatic selection enforces that restriction
during meta admission too. Hardware CUDA execution remains user-deferred.

Final evidence (overlapping counts):

- `/tmp/feat4-auto-worker-regression.xml`: 37 passed, four actual process/CUDA
  cases excluded. Includes all builtin meta layouts, grad-dependent storage,
  actual automatic shared/nested updates, minimal fitting selection, immutable
  package inventories and saved numerical recovery under a larger allocator.
  Trusted recovery verifies parent-load refusal before using the real tensor/
  package decoder seam; it is not guardian restart evidence.
- `/tmp/feat4-auto-library-legacy-final2.xml`: 105 passed, three user-deferred
  CUDA skips. Includes the unchanged pinned trace, standard initialization,
  custom source, shared/dropout replay, memory and diagnostic regressions.
- `/tmp/feat4-auto-contract-boundaries.xml`: 427 passed, three CUDA skips,
  seven actual process cases excluded. Current repository contract/capture/
  package/transfer/handler/publication tests are imported by isolated wrappers
  without the unrelated global TCP fixture. No production environment marker
  was spoofed. All 67 generated schemas match their source.
- `/tmp/feat4-auto-supervised.xml`: actual CPU supervision fails at denied
  `SO_SNDBUF` setup before child execution. `/tmp/feat4-auto-postgres.xml` has
  four connection-setup errors at the denied PostgreSQL socket. These fresh
  attempts do not pass their acceptance gates.
- Targeted compilation, documentation/body-comment audit of 74 scoped functions
  and whitespace checks pass. No schema/Rust SQL/frontend implementation changed.
  Earlier intermediate failures are superseded by the final named passing reports.

ISS-59 is now in progress. `train/warm_start` dispatches explicit token-table
capabilities; standard/legacy share one original-mean initializer. Unsupported
models retain weights and train normally. Actual custom source tests cover
tied/untied ownership. Independent masked AdamW comparisons and exact in-memory
cursor/RNG recovery pass for standard, looped and a custom dropout LM without
the capability. Saved numerical SFT/package recovery now passes the separate
latest checks above; actual process recovery remains unverified.

`train/sft_tokenizer` now verifies saved bytes and prepares deterministic private
memory-label derivations with unchanged IDs and `derived_from` provenance.
`corpus.sft.build` can render to an explicitly owned private directory with
token/mask byte, document and source-character budgets plus cancellation cleanup.
A real old-placeholder case exposed/fixed missing memory IDs in renderer history.
These helpers now connect to the SFT worker and parent-owned durable preparation
authority described above.

Final SFT support evidence: `/tmp/feat4-sft-support-final.xml` has 94 passes and
three CUDA skips, including real legacy fine-tuning, original pinned trace,
source/metadata, token rows and masked loop checks after shared initialization
and example-pin refresh. `/tmp/feat4-sft-data-final.xml` has 62 passes after the
renderer change, including actual exact-fit budgets, failures after partial
flush, source-size refusal, old-slot memory masks, private derivations and legacy
memory SFT. Counts overlap. Earlier focused 12/3-case passes are superseded by
these broader reports; an intermediate test constant-name error was fixed.
Compilation, diff whitespace and a 43-function scoped documentation/body-comment
audit pass for the SFT changes. No source/database/schema frontend integration
is implied by these library/preparation checks.

Tracing the next handoff exposed a pretraining recovery wire mismatch: the real
parent constructs a fresh Train command with `model_package_v1`, while its
validator still required `checkpoint_v1`. Bootstrap recovery now requires the
self-contained package; resident pinned training retains the separate checkpoint
role. Saved numerical recovery tests now construct fully validated commands
instead of model_copy updates that bypassed this boundary.
`/tmp/feat4-recovery-wire-final.xml`: 524 contract/worker cases pass, four actual
supervised/CUDA cases excluded, including all real saved numerical recovery cases.
`/tmp/feat4-recovery-wire-roles-final.xml`: 503 current contract/role cases pass,
covering positive bootstrap package recovery and both directions of role-confusion
refusal. Counts overlap; all 67 schemas are fresh with no shape regeneration
needed. Validator declaration audit, compilation and whitespace checks pass.
This fixes a real constructor/transport boundary but does not unblock denied OS
sockets or establish genuine process recovery.

Next: continue ISS-60 RL,
ISS-61 serving/scoring, connected UI and requirement-by-requirement cross-module
acceptance. Pinned SFT dispatches through the supervised adapter. Existing
ISS-58 genuine process/recovery and PostgreSQL gates remain open. Poll Discord
at milestones and keep concise updates; never close an issue from fixtures alone.

## Historical pause: 23 October 2026

The user returned and requested a graceful stop before turning off this computer.
Do not continue goal work until the user resumes it. FEAT-4 remains incomplete;
the persistent goal is paused. Work is preserved on `modular-arch`.
The user previously requested autonomous work, regular
commits and periodic main Discord-thread checks, with no sudo or Docker. `.git`
is mounted read-only; the authorized staging attempt failed creating index.lock.
No commit succeeded. Preserve all tracked/untracked work and do not bypass the
mount. Eight of 21 issues are accepted: ISS-48/49/50/51/52/53/57/63. Other live
integration gates remain open; no completion ping has been sent.

Root owns all shared runner/config/loop edits. No subagents are active or allowed
without a new explicit delegation instruction. The earlier stop/no-commit notes
below are historical and superseded by the resumed user instructions.

Current estimate given to the user: **8/21 issues accepted (38%); approximately
65% implemented overall, with a rough 60–70% range**. The latter includes partial
work and is an effort estimate. Automatic planning trial code is not accepted.

After pausing, the user supplied actual Compose startup errors. A scoped repair
added the frontend's read-only `/contracts` mount, production image copy and
Docker build-context inclusion. `lock_run` now compiles PostgreSQL
`FOR UPDATE OF execution_jobs`, preserving eager domain reads without locking
nullable outer-join sides. Twenty-four available store/handler/SQL-compilation
checks and a reproduced container-relative schema preflight pass. Six required
real PostgreSQL cases still fail at sandbox socket connection setup; Docker was
not run. User should rebuild/recreate frontend (renew its anonymous dependency
volume) and restart the bind-mounted engine. The larger goal remains paused.

Visual progress is available in the existing frontend fixture pages:
`/markup/authoring` (Python/YAML/typed controls) and `/markup/graph` (interactive
handcrafted graph). Start the native frontend with `cd frontend && bun run dev`
and open port 3000. These pages perform no catalog mutations and are not the
connected feature workflow. They were previously browser-tested; a fresh full
stack launch was not verified under the current socket restriction.

Current implementation:

- Catalog/evidence/validation HTTP APIs, exact queued source snapshots, report/
  graph publication, acceptance/retry/cancel/startup adoption and nine builtin
  block plus four recipe readiness flows are implemented. Startup creates no
  fake acceptance; actual CPU numerical fixtures validate supplied examples.
  Builtin descriptor hashes never enter the custom SourceLoader. Copied public
  builtin imports retain installed implementation compatibility pins.
- Shared contracts/schemas, source/parameter/tokenizer/context/profile identity,
  strict captures and complete graph evidence, worker-only source/tensor package
  loading and bounded USTAR transfer exist. New packages declare fp32 parameter
  precision separately from execution/autocast precision. Parent tensor readers
  reject modular files. SourceLoader stays alive through numerical lifetimes.
- TrainConfig selection and whole-chain validation/reserved outputs/pins/actor
  idempotency are implemented. Waiting steps inherit pins; retry/recovery preserve
  them after edits/archive. Rust forwards request identity/admin scope/status.
  Ordinary modular resume now inherits a retained semantic definition/source/
  tokenizer after original history pruning or root archive, requires admin actor
  plus request UUID, and rejects structural overrides. Same-actor replay retains
  its permission check after the saved input is deleted.
- Pinned pretraining now dispatches from Runner to `api/training_handler.py`,
  using the existing lease. It acquires data in the parent, verifies saved package
  authority, finalizes bindings, stages private inputs and runs the real child
  factory. `RegisteredCheckpoint` publishes already committed output without
  duplicating legacy registration. Success follows session and input cleanup;
  durable cancellation winning the terminal row lock remains cancelled. Pinned
  SFT/RL still refuse their legacy numerical paths.
- `api/training_admission.py` observes memory/cgroup/disk capacity, leaves host/
  disk headroom, bounds input cardinality and budgets input/recovery/package/
  archive/grant copies. Private preparation grants cap at 8 MiB. Exact serialized
  tensor write quotas and generation estimates/refusals precede AdamW; current
  staging capacity is rechecked on publication. PostgreSQL publication locks and
  statements have finite transaction-local limits.
- Parent `orchestrate/modular_data.py` retains actor/context and uses saved owned
  tokenizer bytes without needing the original named build. Streamed packing
  enforces token bytes including separators and document rows before exceeding
  caps, cleaning partial output on refusal. Pure `train/data_identity.py` emits
  identical original hash bytes in bounded parts; `mapped_data.py` owns cancellation
  and readonly-page release. A real 128 MiB mapping test measures bounded RSS.
- `train/modular_worker.py` and modular_inputs/preflight/checkpoints implement
  actual final source/model construction, meta admission, effective train/eval
  capture/numerical checks, shared-loop optimization, measured telemetry and
  complete recovery/final packages. Current precision fallback records fp16 when
  bf16 is unavailable. Shared diagnostics preserve RNG/update trajectories.
  Compatible continuation restores moments and keeps current optimizer options.
- Explicit occurrence replay now passes conservative meta/real admission plus
  full-batch outputs/gradients/post-backward RNG equivalence to ordinary strict
  capture before AdamW. Every generation inventories `training-plan.json` with
  exact definition/receipt/profile, memory bounds and selected/candidate policies.
  Exact recovery requires the retained selection. Explicit CUDA replay requires
  `compile_model=false`. Automatic memory-saving estimation remains unfinished:
  nonempty replay reserves an additional complete captured activation/workspace
  bound; `null` selects zero. No guessed reduction is applied.
- `api/training_publications.py` verifies complete grants plus queued job/config/
  pin/profile/output identity, publishes immutable bytes and commits receipt,
  definition/run/model links plus retention before ACK. Registration failure
  leaves an adoptable orphan and stops publication. `record_training_evidence`
  remains separate from done-only validation evidence. Startup recognizes modular
  packages and discards only superseded unregistered recovery generations.

Latest evidence and limitations:

- Final stopping check `/tmp/feat4-nightly-stop-smoke.xml`: 10 passed in 21.74
  seconds, including the eight current automatic-planning draft tests plus real
  standard pretraining and invalid-count refusal after restoring conservative
  production preflight. The trial automatic integration report
  `/tmp/feat4-recompute-worker-first.xml` has seven failures/12 passes/five
  deselections and is **not** a passing regression. Standard builtin meta
  observation calls `int(positions.max())`; record and fix this on resume.
  Production automatic wiring was removed; the new library/annotation code stays
  preserved for review. No full regression was rerun after these latest changes.
- `/tmp/feat4-replay-regression-final.xml`: 81 passed in 247.25 seconds, nine
  actual process/CUDA cases excluded; current child/support/shared-loop/
  publication/handler/capture tests are collected from the repository. Final
  `/tmp/feat4-replay-policy-boundaries-final.xml`: three passed in 5.15 seconds
  for unknown counts, compiled-CUDA policy refusal and missing/stale exact-replay
  authority. Explicit standard/shared/nested updates equal ordinary updates;
  trusted serialized standard/shared-loop recovery reaches identical final
  weights. Its tensor-reader seam calls the real package/hash/state decoder after
  checking parent refusal; it proves no process containment or worker admission.
  Automatic saving estimates and compiled replay remain unfinished. Scoped
  87-function documentation/body-comment audit, compilation and whitespace pass.
- `/tmp/feat4-training-queue-regression-final.xml`: 179 passed in 93.12 seconds;
  two actual process cases and one dataset-service-dependent legacy case excluded.
  This combines parent boundaries, admission, submission/routes/chains, real child
  arithmetic, loop trajectories, package adoption/retention, legacy state, packing
  and chunking. Counts overlap previous reports.
- `/tmp/feat4-training-queue-cancellation-final.xml`: 41 passed in 14.32 seconds,
  with one actual supervised validation excluded. It includes final handler
  refactoring and cancellation-at-terminal-lock behavior. The subsequent saved
  admission focus is recorded in the evidence ledger.
- `/tmp/feat4-training-queue-process-final.xml`: two CPU cases fail before child
  execution at the known socket permission restriction; CUDA is user-deferred.
  Genuine queue training and checkpoint-cancel/signed recovery/exact final-weight
  comparison remain runnable, with real parent packing and finite corpus pages.
- `/tmp/feat4-training-queue-postgres-final.xml`: four fixture connection errors
  at the denied local socket. Real registration/rollback, history cascade/saved
  pins and publication lock-timeout checks remain unverified. No schema or Rust
  SQL changed; no migration or SQLx regeneration applies to these additions.
- Shared contract/package regression previously passed 562; frontend decoder
  passed 214, with lint/typecheck passing. Current 67-schema and frontend generated
  freshness checks pass. Scoped compilation/documentation/body-comment/whitespace
  checks pass; final counts are in the delivery ledger.

Adapter fixtures explicitly use synthetic graphs/tensor bytes where appropriate;
child tests separately train real weights. Neither substitutes for actual process,
HTTP/network, PostgreSQL or GPU acceptance. Never set LLM_RUNTIME_CHILD artificially
to run parent tensor loading. All current test sessions are terminal once the
latest ledger entry is written; do not restart a live process merely on timeout.

Next work:

1. Complete automatic memory-saving recomputation estimation. Explicit eager
   replay and final-profile conservative candidate retention now exist; current
   policy reserves extra full-graph workspace rather than claiming savings.
   Strict checkpoint export fails in the installed Torch runtime; supplying a
   selective context fixes that error but decomposition then fails on body
   constants. Never accept an opaque graph or patch installed Torch to bypass it.
   The draft `train/recompute_capture.py`, `captured_recomputation` and
   `tests/test_recompute_capture.py` now implement/test real occurrence partitions,
   complete returned backing-storage accounting and minimal fitting prefix
   selection. Capture annotations are in the ordinary executor; custom source
   cannot manipulate compiler metadata. `estimate_model` separates calculation
   from ordinary `plan_model` admission. These are **not production-wired**:
   resolve standard builtin eager meta observation, review lifetime assumptions,
   preserve exact-recovery selection under changed admission and rerun full
   numerical/capture/package regressions before connecting them.
2. Continue ISS-59 SFT, ISS-60 RL and ISS-61 serving/scoring through the existing
   shared owner. Prepare authorized data/effects in Rust-scoped parent adapters;
   custom model/reference/optimizer and tensors stay worker-resident. Preserve
   private tokenizer adoption, masks/cursors and uncertain gym-write recovery.
3. Finish actual connected authoring/graphs/training UI and acceptance slices.
   Inspect each issue/plan and current code before narrowing scope. Keep live
   gates open, maintain runnable checks and record blocked results truthfully.
4. Revisit genuine saved/process recovery when sockets are available. Current
   child rejects incompatible optimizer group/order changes before AdamW; an
   ordinary decay change that preserves group ownership/order is numerically
   tested. Hardware CUDA execution remains deferred to the user's other machine.
5. Perform full requirement-by-requirement acceptance only after implementation
   and required checks justify it. Do not close issues from mocks/builds alone.

The main Discord thread was polled during work. The user's latest local
instruction requests tonight's pause and an honest frontend-runnability report.
Shutdown/handoff and current percentages are posted to Discord. The prior ~55%
estimate is superseded by the current ~65% estimate above. Resume only when asked.

## Historical checkpoint: stopped 22 October 2026

Development stopped at the user's request to conserve usage. All three GPT-6.1
Sol medium agents stopped; temporary test services were shut down. No further
implementation or tests should run until the user resumes. Work remains
uncommitted on `modular-arch`, starting HEAD `e44b0af`; no commit or push was made.
Preserve tracked and untracked files. The historical checkpoint below describes
20 October and is superseded by this section and the current evidence ledger.

Accepted: ISS-48/49 from the previous session, plus independently reviewed
ISS-50/51/52/53/63 this session. This is 7 of 21 issues (33% by issue count),
approximately 35% by estimated effort; it is not an end-to-end acceptance claim.
FEAT-4 remains in progress, with development stopped.

Incomplete work and next gates:

- ISS-54: production capture/numerical implementation exists. Root's targeted
  independent matrix passed 15 tests; the owner's final expanded run was
  interrupted after 10 passes. Actual export guards still need folding into the
  receipt profile, remaining review/tests, frontend graph export and parent
  publication integration. Do not accept the slice yet.
- ISS-55: standard package roundtrip passed. A looped fixture serialization
  failure was corrected using YAML serialization but was not rerun. Latest
  pins/device/precision and parent checkpoint refusal guards remain unverified;
  archive transfer is untested and not wired into `Load.package`.
- ISS-62: signed Rust/Python visibility and genuine HTTP/SSE/WebSocket boundary
  checks pass. Live catalog, submission and serving gates await ISS-56/58/61.
- ISS-64: synthetic graph browser acceptance passes; actual ISS-54 artifact
  acceptance remains open. ISS-63 generated browser schemas/freshness correction
  was independently reaccepted after 321 tests and real browser decoder checks.
- ISS-56/57/58/59/60/61/65/66/67/68 remain delivery work. The broad engine run
  was interrupted with a missing validation handler (ISS-56) and a test fixture
  pause leak. The leak was fixed and a combined 40-test check passed; the full
  engine suite has not passed and must be rerun after integration.

Verification records and commands are in the
[implementation evidence](modular-architecture-evidence.md) and affected topic
files. Final Rust workspace verification: 223 passed, one ignored; the genuine
privacy test passed separately, and strict workspace Clippy passed. Frontend:
321 tests, typecheck, lint and browser scripts passed. Earlier production build
passed 121 pages; the final build was cancelled during TypeScript checking after
compilation. GPU hardware verification remains explicitly user-deferred.

Before resuming, read current Discord FEAT-4 and issue discussions, full AGENTS.md,
conventions and affected module READMEs, then inspect git status and the complete
diff. Keep one owner for shared storage/migrations and runner/dispatch. Resume
54/55 and their integration gates before claiming the live APIs or graph flow.

## Historical checkpoint: 20 October 2026

Checkpoint: 2026-10-20, branch `modular-arch`, HEAD `49c14b1`. All feature work
is uncommitted in the shared worktree. No commit or push is authorized. Preserve
tracked modifications and untracked files; `git diff --stat` excludes the many
new contract/source/test/doc files. Read `git status --short` before resuming.
The user confirmed pause. Root marked the goal PAUSED and posted the Discord
pause;50/51/52 remain incomplete pending final review. No further implementation,
review or tests may run until the user resumes.

## Authoritative requirements and state

Discord FEAT-4 revision4, its amendments and ISS-48 through ISS-68 remain the
requirements authority. Local navigation copies are `/tmp/modular-arch-created.json`,
`/tmp/feat4-full-4.md` through `9.md`, `/tmp/feat4-current-amendments.md` and
`/tmp/feat4-current-index.md`. Root manages Discord states. The detailed local
acceptance ledger is `/tmp/modular-arch-evidence-ledger.json`; the persistent
[implementation evidence](modular-architecture-evidence.md) records current gates.
Read full AGENTS.md, conventions and affected READMEs again before resumed edits.

The user explicitly deferred actual GPU verification to their other machine.
GPU support remains required; keep runnable hardware checks and record them as
user-deferred rather than passing on CPU. No GPU hardware is available here.

## Attention continuation: 24 October 2026

ISS-61 now includes a reusable resident attention owner, actual module/nested-loop
view discovery and per-view strict capture/numerical proof before measurement.
Builtin standard/attention/layer declarations explicitly opt in; five example YAML
refs match the changed builtin manifest. Normal math/initializers and the pinned
trace remain unchanged. `InspectionCatalog` separates candidates from evidence;
`AttentionMatrix` binds definition/profile/point/worker/sequence bounds. Proofs
last only for that worker and invalidate after weight/layout changes. Legacy
layer/head requests remain supported only through deliberate adapters. Generic
models receive no invented fixed layout.

Public attention HTTP discovery/measurement, the authenticated Rust discovery
route and the model viewer are connected. Discovery is explicit, never automatic
on metadata view. Session/model/point/text replacement aborts reads and suppresses
late matrices; worker replacement clears old view proofs. Public refusals use
static actionable wording, never custom source exceptions. Interactive engine
HTTP ownership now cancels/drains nonstreamed chat and attention on disconnect.
Real browser-through-Rust disconnect semantics remain a required live gate.
An authoritative engine `architecture: null` cannot fall back to training defaults
and fabricate a Transformer diagram.

Current reports: `/tmp/feat4-attention-boundary-final.xml` 436 pass, two real
supervised cases excluded; `/tmp/feat4-attention-owned-final.xml` 28 pass including
unchanged trace and actual custom/shared/nested measurements;
`/tmp/feat4-attention-catalog-http-final.xml` 411 pass. Scopes overlap. Frontend
264 focused checks, typecheck and lint pass. All 71 generated schemas/types are
fresh. Rust compilation and API all-target Clippy pass; the actual new proxy test
fails at denied socket setup. Six runnable CUDA attention cases skip as explicitly
user-deferred. Production build is blocked by DNS fetching the existing Google fonts; a diagnostic webpack run exposed EAI_AGAIN and its temporary config change was restored exactly.
No live browser pass has been obtained. Exact logs are in the latest ledger entry. No issue was newly
accepted. Continue remaining connected authoring/training/model/run UI and actual
cross-module acceptance; optional cache/context extension remain disabled.

## Current owners and handoffs

| Agent | Files and current slice |
| --- | --- |
| Root | Orchestration, Discord states, final gate acceptance. |
| `sol_coordination` | ISS-49 authoritative `spec.py`/`wire.py`/schema harness and API fixtures; ISS-51 `store/architecture_schema.py`, `architecture_queries.py`, sole Alembic migration/schema ownership, `store/schema.py`, `llm/storage.py`, persistence/route tests and documentation maps. |
| `sol_capture` | ISS-50 SDK/source/parser/resolver/build/execute/cache/plan files and core tests; ISS-49 interfaces and SDK/YAML fixtures/docs. Authorized minimal `contracts.decode_fixture_yaml` delegation to production `parse.decode_yaml`; confirm its final result before resuming shared harness work. |
| `sol_runtime` | ISS-52 runtime supervisor/guardian/worker/artifact/effects modules, `llm/model/device.py`, initial `api/runner.py` and `main.py` integration and runtime tests/docs; sole current runner owner until handoff56 then58. |

No competing Alembic heads, schema copies, store edits or runner edits. ISS-55
will consume storage path helpers; ISS-56 and58 consume explicit Session-owned
catalog transactions. No further subagents or new slices were started at this
checkpoint. Root must accept current gates and assign the next work explicitly.

## ISS-51 completed implementation and checks

Forward-only migration `b8e246d913af` follows `a4e79c1026bf`, adds11 catalog,
revision/dependency, retained receipt, definition, validation-domain, immutable
run-pin and actor/request tables, and nullable model/training definition refs.
It adds `model_validation` RunKind/output without a second lifecycle authority.
PostgreSQL triggers reject immutable-row updates; restrictive FKs preserve pinned
source/evidence. Legacy IDs and null definition semantics are retained.

[Persistence documentation](architecture-storage.md) gives exact transaction APIs
and helper paths. The caller owns commit/rollback of the complete chain. Drafts
save exact invalid Python/YAML without imports; acceptance locks current draft
and evidence; enqueue snapshots immutable transitive sources; definitions retain
profile receipts outside semantic identity; actor/request retries serialize.
New `record_evidence` requires the validation job already `done`;56 must finish
job and retain evidence in one transaction after verified worker cleanup and
artifact publication. Validation cannot be a model-producing chain step.

Executed from `llm-engine/` with installed `engine/.venv` Python3.14.8,
torch2.13.0+cu130 CPU and local PostgreSQL18.6:

- `PYTHONPYCACHEPREFIX=/tmp/iss51-pycache ../engine/.venv/bin/python -m pytest tests/test_architecture_store.py tests/test_migrations.py tests/test_store.py tests/test_run_routes.py tests/test_chain.py --log-level=WARNING --junitxml=/tmp/iss51-persistence.xml`:62 passed9.78s.
- After final completion/chain guard changes, `PYTHONPYCACHEPREFIX=/tmp/iss51-pycache ../engine/.venv/bin/python -m pytest tests/test_architecture_store.py tests/test_run_routes.py --log-level=WARNING --junitxml=/tmp/iss51-final-store-routes.xml`:40 passed8.32s.
- Coordinated49 fractional queue-rank/blank-label correction:355 affected contract tests passed3.28s;67schemas regenerated and checked.
- Targeted compileall, function-doc AST audit, local documentation links and `git diff --check` passed.

All migration/concurrency databases are unique `slate_arch_<pid>_<uuid>` names,
created from template0 with verified ownership markers before forced cleanup.
Existing app DBs were untouched. Template0 avoids a pre-existing template1
collation mismatch. No SQLx metadata regeneration applies to51: no Rust SQL/schema
changed. Storage fixtures supply typed receipts, not numerical execution proof.
HTTP/worker/publication integration remains52/55/56/58. Root/peer review was
pending at the budget checkpoint; do not claim those downstream slices complete.

## ISS-50 brief independent review

`PYTHONPYCACHEPREFIX=/tmp/iss51-pycache ../engine/.venv/bin/python -m pytest tests/test_architecture_core.py --log-level=WARNING --junitxml=/tmp/iss50-core-independent.xml`:
40 passed9.67s. Read-only review covered parser, source eligibility/loader,
resolver, construction/alias placement, eager/strict invocation and occurrence
cache routing. No confirmed blocker was found in those boundaries.

One narrow policy caveat remains for54/runtime integration review: a custom
forward containing `torch.get_default_dtype() == torch.float32` passes source
eligibility and strict export. The worker/profile must pin this ambient setting
or the supported-source policy must reject the read. This is a recorded concern,
not a proven eager/export mismatch. `torch.initial_seed()` and
`torch.get_num_threads()` were separately observed to fail strict export. Do not
expand the loader into a hostile-Python sandbox; the supported policy is trusted
admin source, with bounded process ownership and explicit capture checks.

## Next resume steps

1. Read peer final reviews and root gate decisions for50/51; inspect their last
   files and targeted reports. Resolve only concrete recorded defects.
2. Confirm52 final runner/main lifecycle and real-process evidence. Keep its
   runtime/HTTP wiring obligations explicit.
3. Root assigns53 to capture after50 and Rust62 to coordination after51;62 may
   start against49 fixtures but cannot close before live56/58/61 contracts.
4. Continue54/55/57 and then56/58 in dependency order. Keep one migration owner
   and serial runner ownership; preserve actor privacy and atomic chain pins.
5. Maintain per-issue evidence and actual user-deferred GPU commands through68.
