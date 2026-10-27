# Modular model and checkpoint packages

Self-contained packages are consumed by supervised pretraining, recovery, SFT,
RL, serving and measurement. They preserve immutable definitions independently
of live catalog names or archival. Current verification and remaining gates are
tracked in the [acceptance map](modular-architecture-acceptance.md) and
[evidence ledger](modular-architecture-evidence.md).

Implemented surfaces:

- `architecture/package.py`: data-only manifest/inventory/source/closure/receipt
  verification; bounded staged/fsynced publication; registration reconciliation;
  conservative semantic-definition deduplication without package dependency on
  the shared cache.
- `package_integrity.py`: local typed inventory, canonical bounded JSON/regular
  files, complete bounded no-link traversal, Linux no-replace directory rename.
- `package_execution.py`: worker-only source/builtin load, recipe reconstruction,
  aliases before safe tensor state loading, optimizer ownership/order checks.
  The loader now requires exact RuntimePins and a retained receipt matching its
  definition, parameter/profile digests and actual admitted device. New packages
  can state `parameter_precision` independently of execution precision, retaining
  fp32 master weights under bf16/fp16 autocast. Absent fields preserve the earlier
  receipt-precision loading behavior; tensor precision is still checked exactly.
- `package_metadata.py`: bounded saved definition/receipt summaries and individual
  graph-scope reads without source execution, tensor reads or live catalog access.
  Receipt collections cap at 8 MiB; graph bytes retain the aggregate engine cap.
  Summary receipts retain exact graph query identity after source-job pruning.
  Selected source reads verify inventory length/hash and immutable revision hash
  without live catalog access. The root `architecture.yaml` is inventoried with
  exact accepted bytes; complete package verification checks both its revision
  digest and parsed recipe against the saved definition. Custom dependency
  Python/YAML remains exact frozen text. Builtin `.builtin.json` entries are
  readable frozen descriptors, explicitly labeled JSON rather than current
  implementation source. Missing/corrupt source fails instead of falling back.
- `package_transfer.py`: uncompressed USTAR model-package container export and
  bounded fixed-header materialization. Truncation, traversal, links, duplicate
  paths, invalid trailers and resource overruns fail with staging cleanup.
  `load_granted_package` verifies private `Load.package` and definition grants,
  materializes a worker-local package and checks exact definitions/pins before
  source construction. Supervised roundtrip/restart cases verify the actual
  child loading boundary separately from data-only container checks.
- `model/checkpoint.py`: legacy v1 remains separately discriminated from
  modular v2 tensor files; modular optimizer/RNG/cursor state references exact
  sibling weights. Parent load/load_state now refuse modular tensors before
  deserialization; modular recovery requires `LLM_RUNTIME_CHILD=1`.
- `dependencies.py` shares exact direct pin/digest policy with the catalog's
  delegating store wrappers. `storage.py` adds package manifest/staging paths.

Serving Load grants now restore weights alone and run caller-owned meta admission
before constructing real model storage. Recovery retains its explicit training
state path. A weights-only read reuses its decoded tensor dictionary instead of
loading the same sibling file a second time; assignment releases that dictionary.
Closing a loaded package drops resident tensors before revision imports retire.
Parent verification/archive/grant preparation has cancellable finite read
boundaries. These are connected to the [serving worker](modular-serving.md).
Actual CPU serving, archived-definition recovery and ordinary-user compatibility
flows are recorded in the evidence ledger.

Package publication accepts an explicit staging root for admitted worker-local
generations. The child uses its private runtime directory; parent callers retain
the storage-owned staging path. The new pretraining checkpoint writer uses real
tensor/optimizer/RNG/cursor files, full sources/receipts/graphs and acknowledged
USTAR publication. Its in-process numerical checks do not establish parent
database registration or supervised saved-model recovery.

New pretraining generations inventory `training-plan.json` alongside these files.
It binds the memory plan and explicit replay selection/candidates to the exact
definition and receipt. Optimizer state retains the selected profile/occurrences/
estimate for exact recovery. Trusted CPU numerical fixtures recover identical
final weights from actual serialized packages while explicitly substituting only
the worker-only tensor-reader entry point; parent refusal remains checked. This
does not establish guardian restart or permission enforcement inside a real child.
See [modular training support](modular-training-support.md) for conservative
captured-occurrence memory estimates and explicit/automatic replay admission.

The parent pretraining publication adapter now verifies grants and versioned
job/generation metadata before immutable model rename. Matching frozen-pin
receipt, definition, run/model links and retention commit before acknowledgement;
failed registration leaves a complete startup-adoptable orphan. Startup recognizes
modular directories and skips legacy tensor handling, removes unpublished package
staging and discards superseded unregistered recovery generations. These parent
transaction checks use labeled synthetic graph/tensor fixtures; they remain
separate from child numerical and actual supervised execution evidence.

The pretraining handler now connects these adapters through the existing queue
lease. It snapshots observed host/disk capacity, budgets corpus/recovery/package/
archive/grant copies, and passes explicit generation/file ceilings into the child.
Before AdamW, the writer checks owned tensor/moment/metadata estimates and current
staging capacity. Actual tensor archive writes enforce their byte ceiling before
each block and preserve prior complete files on refusal. Parent adoption rechecks
free staging space; PostgreSQL publication transactions bound locks to 5 seconds
and individual statements to 30 seconds. These limits do not establish a passing
live-process or PostgreSQL result; see the delivery ledger.

Current data-only checks verify graph manifests against retained receipt identity,
strict mode/guard records and every scope reference. Parent checkpoint readers
refuse modular files before tensor deserialization. Worker tensor assignment
refuses mismatched names, shapes, precision and conflicting shared aliases rather
than silently casting or overwriting them. Synthetic package-test evidence is
explicitly distinguished from numerical capture/execution evidence.

## Verification

Use `tests/test_architecture_packages.py`, `tests/test_architecture_package_transfer.py`,
legacy `tests/test_train_state.py` and the documented grouped verification entry
point. Tests cover stale pins before custom import, optimizer group/order refusal,
bf16/fp16 behavior, actual `Load.package` grants, archive truncation/traversal/
links/resource budgets, interrupted copy/fsync/rename/registration, concurrent
publication and missing/tampered/incompatible package/evidence/builtin bytes.

### Historical sandbox checkpoint: 22–25 October 2026

The combined capture/package-data matrix passed 50 tests with three GPU skips
and six process cases deselected. A subsequent tensor-profile/parent-refusal
check extends that matrix; see the delivery ledger for its final report.
Legacy train-state checks passed 30 tests with their HTTP-dependent end-to-end
case excluded. The standard repository fixture cannot bind localhost TCP in
this sandbox, and real supervised package tests fail at socket-buffer setup
before model code runs. Focused tests therefore used exact copies in
`/tmp/feat4-offline-tests` with only isolated data/tokenizer fixtures, without
the unrelated autouse HTTP fixture. These results do not certify process or
HTTP integration. The earlier looped YAML serialization correction still needs
a successful supervised numerical rerun. Parent pretraining publication,
registration, startup, retention and queue-handler handoff have adapter evidence;
actual saved/process recovery verification remains open. GPU checks
remain user-deferred. The user authorized commits, but `.git` is read-only in
this execution sandbox; the attempted `git add` failed before staging anything.
