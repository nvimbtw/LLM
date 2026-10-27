# Modular training support

ISS-57 supplies worker-local optimizer, memory, recomputation and diagnostic
operations. Production pretrain/SFT/RL integration belongs to ISS-58–60; these
APIs do not move source execution into the parent or add a queue/device owner.

## Parameter ownership

`llm/train/model_policy.py` exposes typed `OptimizerPolicy` and `ParameterPolicy`
capabilities plus `optimizer_groups`. It delegates alias conflict/ownership
checking to the same `architecture.build.parameter_groups` used by construction.
Every trainable Parameter enters AdamW once, with stable first-owner group/order;
frozen tensors stay out. Unknown paths, foreign capability Parameters and
conflicting policies for tied aliases fail before optimizer construction.

The existing attention component supplies its actual temperature Parameter for
zero weight decay. A custom parameter merely named `log_temperature` receives
normal decay. Other components can supply explicit ParameterPolicy capabilities;
the trainer never infers numerical policy from display names or layer indices.

## SFT token-row ownership

`train/warm_start.initialize_token_rows` returns the explicit `TokenRowResult`
instead of assuming an arbitrary model exposes an embedding table. The legacy
Transformer adapter and the standard recipe preserve the original pre-update
table mean policy. IDs are validated and deduplicated before mutation;
out-of-vocabulary rows are reported as skipped. Unknown marker history still
requests no reset.

A custom initializer requires both the resolved `token_rows` declaration and
the matching protocol. A composition delegates only through a single child
called once with its whole named interface. Repeated/local block capabilities
are not inferred to describe the complete model. An unsupported model keeps all
weights and reports `skipped_unsupported`, while ordinary masked training remains
usable. The declared implementation owns all tied/untied table policies; table
attribute names do not select them. The old count-returning
`warm_start_embeddings` remains a compatible wrapper for legacy SFT callers.

Current tests load actual custom source for tied/untied ownership and compare
real standard/looped/custom masked AdamW updates against an independent masked
loss. Separate in-memory recovery checks distinguish consumed batches without
targets from optimizer steps. These are library/loop evidence. The connected
[saved-definition SFT adapter](modular-sft.md) has separate numerical/package and
parent-authority checks. Genuine process/PostgreSQL results and live SFT without
token-row initialization are recorded separately in the acceptance evidence.

`train/sft_tokenizer.prepare_tokenizer` verifies exact saved tokenizer bytes and
returns them unchanged unless memory IDs used by the rendered corpus require
private label adoption. Derived bytes keep all existing IDs and merges, receive
a deterministic new tokenizer UUID/hash, and retain the original fingerprint in
`derived_from`. Already adopted vocabularies keep their exact saved bytes,
including formatting; foreign slot collisions and false saved facts refuse.
This data-only helper performs no catalog/filesystem mutation. Publication and
durable authorization use the parent-owned SFT preparation record;
`derived_from` alone is not proof of an authorized transition.

## Memory estimates and admission

`llm/train/model_memory.py` plans from the exact constructed parameter snapshot
and its complete strict export. It works with meta tensors before real model
allocation, counts shared Parameters/buffers once and retains gradients, AdamW
moments/master storage and autocast storage for trainable elements. Activation
policy conservatively retains four times all bounded captured tensor outputs
and separately reserves fp32 quadratic attention scores. It has an explicit
supported operation set; unknown workspaces/shapes fail, without applying
Transformer width/head/layer formulas to custom code. This estimate is
conservative policy, not hardware calibration for arbitrary operators.

The supplied custom gated FFN's SiLU is included in that supported set. Its
pointwise sigmoid/product intermediates fit the same four-copy tensor bound;
accepting SiLU does not enable unknown operator workspace estimates.

An optional reference model needs its own captured workspace evidence. Shared
storage counts once, while reference execution workspace still counts. Other
resident allocations exclude tensors already modeled by these arguments.
The caller supplies its actual admitted allocator ceiling; the planner leaves
the existing allocator headroom and rejects a configuration that cannot fit.
Plans retain the actual bounded upper shapes of user tensor inputs, excluding
captured Parameters/buffers. The shared training loop requires the exact
full-prefix token/position/mask batch and context shapes; a plan from another
batch or context cannot reach optimizer construction. Byte bounds are checked
at plan construction, and summaries distinguish captured-model estimates from
the measured legacy planner.
The existing measured legacy planner remains available for legacy models.
`memory.guard` can tighten an existing allocator fraction but cannot increase a
worker's admission cap. CPU mock checks verify this policy; real CUDA allocation
and cleanup remain user-deferred hardware gates.

## Occurrence recomputation

`RecipeModule.recomputation_targets()` enumerates actual calls across independent
stacks, shared fixed loops and nested subarchitectures. Targets identify both
module instance and flow/iteration path. `configure_recomputation` rejects
unknown/duplicate occurrences and sets one between-step execution policy.
It never changes loop counts, saved semantics or parameter ownership.

Selected training calls use nonreentrant PyTorch checkpointing with CPU/device
RNG preservation; evaluation/no-grad and cached execution retain eager behavior.
The supported source contract still requires explicit tensor state and forbids
persistent/input mutation or unrecorded randomness. Named-port and mutation
checks apply during original and replay computation. The worker may only enable
the optional capability after exact-profile numerical evidence; method presence
alone does not authorize it.

`llm/train/recompute_plan.py` chooses the fewest replay occurrences that fit
from bounded exact-profile preflight estimates. It refuses duplicate, stale,
unknown or nonfitting candidates. It applies no guessed activation-memory
discount. Admitted pretraining retains all bounded prefix-policy candidates,
bound to the actual captured profile. `true` selects every
eligible occurrence, `false`/zero selects none, and an integer selects the first
count in execution order. Counts outside the actual targets fail before AdamW.

The `captured-occurrence-lifetimes-v1` estimator partitions the ordinary strict
capture by engine-owned leaf-call annotations. It retains unselected captured
work and every leaf output's complete backing storage, including unused ports
and larger aliased bases, plus the largest selected replay region. Port storage
is charged separately even for unselected calls so export dead-code elimination
cannot hide a retained eager output. Ordinary inputs/parameters and unannotated
operations stay charged. The existing four-copy tensor bound and supported
operator workspace policy apply; there are no layer/width discounts.

Both meta and real admission select the least replay that fits. `null` selects
zero when possible and otherwise the fewest fitting prefix occurrences; if none
fits, training refuses. A single composite occurrence may offer no estimated
peak reduction. Output-storage observation uses training grad mode, retains no
graph and always removes its hooks. The standard builtin now supports shape-only
meta position checks without extracting a scalar; actual position values still
require real numerical validation. This changes its manifest/revision identity.

Strict checkpoint export/decomposition remains unsupported by the installed
Torch runtime. Planning uses complete ordinary captures and explicit eager
checkpoint semantics; no opaque checkpoint graph is accepted. Selected eager
execution must match ordinary
strict capture's full-batch/context outputs, gradients and post-backward CPU/
device RNG before optimization. Selected CUDA replay requires
`compile_model=false` until checkpoint compilation is validated.

Every new pretraining generation inventories `training-plan.json`, containing
definition/receipt identity, memory bounds, estimator provenance and selected/
candidate occurrences. Measured-result metadata repeats the replay policy;
recovery state stores the selected profile/occurrences/estimate. Exact recovery
requires the same retained selection as well as the original config/data/cursor.
A larger allocator ceiling cannot silently select less replay during exact
recovery. A stale/missing profile or changed storage estimate refuses exact
recovery; ordinary continuation can run a new final preflight. Compiler metadata
access remains outside custom source eligibility. This is a supported-operation
estimate, not hardware calibration or a guarantee about arbitrary custom code.

## Optional diagnostics

`ModelDiagnostics` in `llm/train/model_diagnostics.py` owns CPU before-update
snapshots and request-local hooks. It reports semantic Parameter owner gradients/
update ratios and actual module/occurrence output shapes, dtypes and RMS values.
No missing attention/embedding-specific values are filled with zero.

Probe allocation/work budgets are separate from training admission. Snapshots
and reductions are checked before allocation; a deterministic no-grad evaluation
restores heterogeneous modes, RNG, Parameters, buffers and gradient references.
All temporary hooks are removed on every outcome. Diagnostic CUDA OOM or budget
exhaustion disables optional sampling and releases snapshots; model/programming
errors propagate. Training and optimizer failures never pass through this
optional diagnostic recovery policy.

## Shared training loop integration checkpoint

ISS-58 now extends the existing `train/loop.py` rather than introducing a second
scheduler or update loop. `model/language.py` supplies the deliberate legacy
tuple-to-logits adapter and declared vocabulary/context facts for protocol-based
models. Recipe facts select the actual training context; flat Transformer shape
defaults never configure a recipe. Legacy parameter updates remain identical
under the shared optimizer ownership policy, including actual attention
temperature exclusions.

Declared models require the caller's captured final-profile memory admission
before model training or optimizer construction. The loop checks actual owned
tensor bytes and token/position/mask batch/context shapes. It does not call the
legacy width/layer planner or set an invented recompute depth. Caller-validated
occurrence recomputation remains configured. Producing valid final preflight
receipts/candidates under the actual worker admission is still the supervised
training adapter's responsibility.

`train/diagnostics_model.py` connects the existing bounded `ModelDiagnostics` to
the shared observe/before/after update cadence. It reports semantic Parameter
owners and actual occurrences; attention/embedding metrics are absent when
unavailable. Probes preserve modes, RNG and update trajectory. The existing
legacy collector and telemetry remain deliberate compatibility behavior.

The same loop retains epoch/batch scheduling, learning-rate/report/checkpoint
cadence and cancellation boundaries. Standard, shared-loop, nested-loop and
dropout models have real CPU update/recovery checks. Exact optimizer/RNG/cursor
restoration is checked at the loop boundary with captured in-memory state; this
does not prove package serialization, process restart or parent checkpoint
adoption. Resume restores saved moments/RNG/cursor while retaining newly
constructed optimizer group options, including requested weight decay and
admitted device policy. Independent real AdamW update checks cover changed
continuation decay and retained moments; unchanged exact trajectories still pass.
Whole-chain request validation and one-transaction queue insertion are
now implemented in the existing run route. Typed recipe selection, accepted
source/evidence pins, inherited waiting-step pins, actor/request deduplication
and retry/recovery metadata are also implemented. Parent-to-worker pretraining
handoff is connected; saved-package/process recovery has dedicated supervised
tests and a live archived-definition checkpoint/recovery scenario in the
evidence ledger. Pinned SFT/RL cannot enter their legacy numerical path.

`train/model_preflight.py` declares the actual batch, recipe context, full-prefix
train/eval domain, effective CPU/CUDA precision, TF32 and seed used to compare
explicit prior evidence. Final preflight is still required inside the admitted
worker. A matching prior receipt cannot bypass actual allocator/capture checks.

## Child pretraining implementation checkpoint

`orchestrate/modular_data.py` prepares authorized datasets/tokenizers and packed
span/schedule facts without source/model loading. `api/training_jobs.py` can turn
the immutable queued closure into a final revision/profile snapshot, checking
builtin compatibility before any custom import. Typed `modular_records.py`
grants preserve sparse config serialization and distinct tokenizer/span/corpus
roles. Pure packed-data identity now lives in `train/data_identity.py`; the
legacy recovery import remains a compatible alias with unchanged hash bytes.

`train/modular_preflight.py` constructs meta weights and estimates memory before
real allocation, then constructs fp32 master weights with the declared seed.
Actual train/eval strict captures and numerical probes run under effective
arithmetic and preserve post-initialization RNG/modes/gradients. Real full-prefix
capture supplies the loop's shape-bound memory plan. The returned owner retains
SourceLoader for the complete training/package lifetime. Scalar tensor creation
is now included in supported bounded activation estimation; unknown work still
fails. Effective bf16 fallback to fp16 is recorded rather than mislabeled bf16.

`train/modular_worker.py` now implements the complete child pretrain command.
It checks tokenizer facts, corpus IDs, structured spans and data/schedule digest
before model work; no service client, actor or device gate is constructed in the
child. It trains through the existing loop, emits actual consumed-batch liveness
and acknowledged measured telemetry, and creates full recovery/final packages
through `modular_checkpoints.py`. Recovery loading and exact continuation logic
are implemented but require actual supervised saved-package verification.

In-process CPU checks train real standard/looped recipes and verify changed
Parameters, measured frames, complete optimizer/RNG package inventories and final
output publication even with automatic saves disabled. Failed input/capture
checks refuse optimization. These are trusted numerical adapter tests; their
publication sink verifies/materializes packages without pretending to perform
durable database registration or process containment.

The default runner now dispatches pinned pretraining to this backend under its
existing lease. Parent preparation enforces streamed token/document output budgets
and checks reused builds, preserving complete prior bytes on refusal. Remaining
work: actual supervised saved/process recovery acceptance. Automatic and explicit
nonempty eager replay use the captured admission and retained evidence described
above. Real CPU standard,
shared-loop and nested-loop updates equal ordinary updates. Fresh trusted CPU
sessions also load actual saved standard/shared-loop tensor/optimizer/RNG/cursor
files and recover identical final weights. That numerical test explicitly checks
the production parent-load refusal, then uses a trusted tensor-reader seam with
the real package/hash/state decoder; it neither spoofs a worker environment nor
proves guardian restart. CPU/CUDA supervision cases cover replay on/off and remain
runnable. CPU is blocked by socket setup here; GPU hardware is user-deferred.

## Parent publication implementation checkpoint

`api/training_publications.py` verifies complete USTAR grants, bounded typed
job/generation metadata, reserved final output and exact queued configuration,
frozen recipe/context/parameters and effective input/runtime profile. It renames
complete immutable bytes into authoritative model storage, then records the
training receipt, semantic definition, run link and model in one transaction
before returning for acknowledgement. Source imports/tensor deserialization stay
out of this adapter. Metadata now explicitly carries the original job UUID and
a versioned generation frame, distinct from arbitrary measured result fields.

`store.record_training_evidence` requires an admitted pinned job and one exact
valid final-preflight receipt; `record_evidence` retains its separate done-only
validation rule. Definition/model references preserve evidence after history
pruning. Registration failures retain one complete adoptable orphan and propagate
to stop further publication. Re-registration preserves human model display,
publication and keep metadata. Retention commits before retiring old directories.

Startup routes modular package directories through verified registration, clears
unpublished package staging and discards only superseded unregistered recovery
generations. A failed/cancelled job's published preflight can be retained without
rewriting its outcome. Adapter tests use explicitly synthetic tensors/graphs to
prove transaction/publication behavior; separate child tests train real weights.
Neither establishes actual process containment or full saved-package recovery.
The default runner connects this adopter through the supervised handler. Its
typed RegisteredCheckpoint event skips duplicate legacy registration; success is
committed only after session and parent-input cleanup. Adapter evidence alone
does not certify process acceptance; current actual worker/database results are
tracked in the acceptance map and evidence ledger.

## Parent admission and input scans

`api/training_admission.py` observes Linux available memory, tightens it by a
finite enclosing cgroup allowance and leaves host headroom. Host admission caps
at 8 GiB and refuses less than 2 GiB after headroom. It reserves 512 MiB of free
disk, constrains acquisition and accounts for private input/recovery copies plus
raw/sealed/archive/grant/parent/retained generation copies. Real tensor writes,
package inventories and runtime aggregate limits enforce distinct byte bounds;
current publication capacity is rechecked as explicitly kept output accumulates.
Preparation grants cap at 8 MiB, as do bounded result/telemetry reads.

Before acquisition, selected training rejects invalid batch/seed/warmup/optimizer
values, nonpositive epochs, more than 128 stages or 4096 dataset selections, and
unsupported derived byte-vocabulary sizes. Saved inputs verify registered exact
definition/source authority before opening corpus access. Immutable stored
definitions support ordinary saved-model submission after original job pruning
or root archive. Saved tokenizer bytes need no original named build. Administrative
actor/request identity and replay checks apply to those submissions too.

`train/data_identity.py` exposes the exact original hash stream in bounded parts.
`train/mapped_data.py` owns cancellation and Linux readonly-page release between
blocks; pure hash calculation has no mapping effects. Corpus ID/span validation
also checks cancellation and releases readonly pages, and bounds document rows.
Writable mappings are left alone. A real 128 MiB mapping check measures resident
page release and bounded scanning, separately from numerical/process evidence.

## Verification

Run from `llm-engine/`:

```sh
../engine/.venv/bin/python -m pytest tests/test_modular_training_support.py tests/test_architecture_core.py tests/test_architecture_builtins.py tests/test_memory_plan.py tests/test_training_memory.py tests/test_diagnostics_memory.py
../engine/.venv/bin/python -m pytest tests/test_modular_training_loop.py tests/test_train_state.py tests/test_training_memory.py
../engine/.venv/bin/python -m pytest tests/test_training_submission.py tests/test_run_routes.py tests/test_chain.py
../engine/.venv/bin/python -m pytest tests/test_modular_worker.py
../engine/.venv/bin/python -m pytest tests/test_training_publications.py tests/test_training_publications_postgres.py tests/test_packing_budget.py tests/test_recovery.py
../engine/.venv/bin/python -m pytest tests/test_training_admission.py tests/test_training_handler.py tests/test_supervised_pretraining.py
../engine/.venv/bin/python -m pytest tests/test_token_rows.py tests/test_modular_sft_loop.py
../engine/.venv/bin/python -m pytest tests/test_sft_tokenizer.py tests/test_sft_rendering_budget.py tests/test_project_memory.py
```

The first full regression passed 113 tests, with three user-deferred GPU skips,
using original tests/isolated data fixtures without the unrelated global HTTP
fixture that cannot bind in this sandbox. Expanded focused results and current
integration limits are tracked in the [delivery ledger](modular-architecture-evidence.md).
Supervised submission/worker/package behavior and CUDA behavior remain separate
evidence; this library suite does not prove them.
