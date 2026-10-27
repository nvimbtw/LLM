# Recovery checkpoints

A checkpoint is an immutable model UUID with weights, tokenizer, metadata and
training state. The output model UUID groups a run's automatic recovery saves.
`checkpoint_target` controls save frequency; zero disables recovery writes.
Pretrain/SFT save at configured step and epoch boundaries; RL saves completed
iterations. A completed output model remains separate from recovery storage.

## Publication and retention

Each file is written to a sibling temporary file, flushed and fsynced before
atomic replacement. Recovery directories are staged outside model listings,
synced and renamed into place before registration. A failed write keeps the
previous complete generation. Optimizer state contains no duplicate weight
copy: `train_state.pt` resolves its matching sibling `model.pt`, checking
architecture/version. Older self-contained states remain readable.

After registration, retention keeps the latest automatic save and any published,
explicitly kept or referenced generations. Chat and benchmark use pins recovery
models before storing a cross-service reference. Training configuration and
model ancestry are checked locally. Pins are conservative: deleting a reference
does not automatically remove its pin. The model page's **Keep** action also
exempts a generation. Automatic intermediate rows from older writers follow the
same rule; published/referenced models survive consolidation.

The row transaction precedes deletion of retired directories. Startup removes
staging and pending files, re-adopts complete unregistered recovery generations,
and reconciles retention before admitting queued work. If registration fails,
the running job stops after its complete checkpoint lands; continuing to write
unregistered generations would permit unbounded orphan storage. Database and
filesystem publication are not a distributed transaction.

## Recovery versus continuation

`POST /runs/{id}/recover` creates a new job/output UUID with the original stored
configuration and latest checkpoint. It never changes an earlier job's history.
Pretraining exact recovery requires matching configuration, packed-data digest
and schedule. Ordinary `resume_from` still supports continuation onto different
data, resetting position/RNG while preserving compatible weights and optimizer.
Architecture and tokenizer always come from the saved model.

Modular pretraining now connects parent acquisition/admission and child recovery
through the queued handler. Ordinary saved-input submission inherits immutable
definition/source/tokenizer facts after original history pruning or root archive;
it requires administrative actor/request identity. The original named tokenizer
build need not survive. Actual tensor/optimizer loading remains worker-only.
Saved group ownership/order must match; compatible continuation keeps current
optimizer options while restoring moments. Real queue/save/recovery comparisons
remain runnable but blocked by this sandbox's socket setup. In-memory trajectory
and trusted adapter results do not establish that acceptance gate.

SFT recovery requires its original configuration and corpus/mask fingerprint.
The saved cursor is the next **batch**, separate from optimizer step: masked
windows with no assistant targets can be skipped without moving the optimizer.
Epoch and curriculum position, optimizer, schedule position and RNG are restored.
Legacy SFT points lacking exact position are refused for recovery.

The [saved-definition SFT adapter](modular-sft.md) now connects private mask/data
grants, inherited weights, optional original frozen reference and prepared
tokenizer authority through the common worker/publication owner. Final generation
metadata accounts explicitly for skipped targetless batches while preserving the
existing schedule. A failed adoption retains parent preparation for startup;
settled terminal/pruned records retire after package reconciliation. Genuine
supervision and PostgreSQL acceptance remain separate open gates.

RL recovery restores policy, original frozen reference, optimizer, global/task/
sampling RNG and curriculum state. It starts after the last completed iteration.
Old sandbox operations are never replayed against their original gym runs; an
interrupted iteration is discarded and new rollouts use fresh sandboxes. Saved
curriculum/provenance retains placement and validation context. Mid-iteration
backward recovery is intentionally unsupported.

The modular RL handler verifies registered policy/original-reference definitions
without loading tensors, then the child restores exact optimizer ownership,
configuration and metadata-matching iteration position. Recovery restores every
global generator plus separate task/sampling RNG; GPU-mapped RNG tensors return
to their owning CPU generators. Parent durable authority fixes the next task
group seed and remaining iteration count, including inherited curriculum offsets.
Returned training siblings must share the requested seed/question before they
can form a policy-baseline group.

Known gym resources have separate parent-owned issuance/retirement evidence under
`artifacts/gym-effects/<job UUID>/`. Startup only retires known resources after
exclusive process cleanup; it needs no surviving job configuration and never
replays a worker command or uncertain cleanup. Fully settled evidence retires
after callback draining. Unknown or uncertain results remain quota-bound;
unknown creation IDs remain the gym service's expiry responsibility. Actual
supervised/SQL/Rust checks remain distinct from numerical and adapter evidence.

The API-managed runner performs registration and retention. Direct library
consumers supplying their own progress callback must implement the same lifecycle
or use the queued API. Testing of crash points and deterministic trajectories is
required separately; file atomicity does not itself establish numerical identity.
