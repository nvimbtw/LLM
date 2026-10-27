# Modular model runtime protocol v1

This freezes the ISS-49 worker interface. It declares messages and policy;
production worker supervision and consumer obligations are described in
[supervised-runtime.md](supervised-runtime.md). Queue/storage wiring and Rust
adapters remain separately owned slices. The CPU experiment is recorded in
[modular-runtime-feasibility.md](modular-runtime-feasibility.md).

`llm/runtime/protocol.py` owns the Pydantic contracts. Shared examples live under
`contracts/modular-architecture/v1/runtime/`; each case identifies a schema and
may request the `runtime_identity` or `runtime_packet` semantic check. Schema
validation alone cannot validate a live lease, actor authorization or receipt
verdict.

## Ownership and channels

The engine parent owns the durable execution job, actual device admission,
initiating actor, service credentials, immutable input grants, deadlines,
publication and recovery reconciliation. The worker owns its complete model,
optimizer, gradients, sampling RNG and request-scoped cache. Source import,
constructor/load, capture, numerical validation, forward/backward, generation
and measurement all execute in the worker, including probes performed after
successful validation. The parent never imports custom modules or unpickles
custom model objects to inspect or adopt their output.

A trusted guardian owns the worker tree and ordinary framework helper processes.
It never imports custom source. Parent liveness/control and guardian cleanup use
separate channels from worker progress/data, so a full progress queue cannot
prevent cancellation or detection of parent death. The parent creates a fresh
`worker_id` and lease UUID each time; operating-system PIDs are observations,
not sufficient worker identity. Unrelated descriptors are closed and inherited
environment values are allowlisted before any source import.

Every packet is one UTF-8 JSON `RuntimeEnvelope`: `schema_version: 1`,
`worker_id`, `request_id`, monotonically increasing `sequence`, and a payload
tagged by `type`. Sequence is maintained separately for each authenticated
sender/channel during a worker lifetime. Reject stale worker/request identity,
duplicate/replayed/out-of-order sequence, duplicate JSON fields, unknown fields
or tags, and packets exceeding **4096 bytes** before invoking behavior.
`RuntimeEnvelope.from_packet()` applies duplicate/byte/schema validation;
`require_sender()` applies channel role. The supervisor must apply live
identity/sequence checks through `require_context()` before advancing its owned
state. No Python pickle or tensor objects cross IPC.

| Authenticated sender | Payload tags |
| --- | --- |
| Parent | `inspect`, `validate`, `load`, `train`, `generate`, `measure`, `cancel`, `shutdown`, `effect_reply`, `artifact_adopted` |
| Worker | `progress`, `tokens`, `effect_request`, `artifact_ready`, `completed`, `failed` |
| Guardian | `cleanup_verified` |

The parent is the only authority to interpret a guardian cleanup certificate.
A worker cannot certify its own cleanup. A schema-valid certificate from the
wrong channel or for another lease never releases admission;
`CleanupVerified.require_lease()` compares the current lease UUID.

## Resident execution and identity

`inspect` receives a source snapshot grant and inspects declaration metadata
without constructing a model. `validate` receives an immutable fully bound
validation snapshot. Their results are typed validation-result artifacts.
Saving/listing source never sends either command.

`load` receives an exact definition grant and optional saved model package.
`RuntimePins` contains `definition_id`, `bindings_digest`, `profile_digest` and
durable `receipt_id`. The definition is semantic identity; bindings/profile
identify this concrete execution authorization. Binding digest is the canonical
digest of the resolved constructor parameter mapping, matching the receipt's
`parameter_digest`; tokenizer/context compatibility also comes from the pinned
definition and receipt. Profile digest hashes the receipt's `RuntimeProfile`.
The parent verifies the receipt is fully `valid`, current and fully bound before
creating pins, and the worker compares every pin before model reuse. Merely
possessing a UUID or a provisional receipt cannot authorize optimizer execution.

`train` sends a durable job UUID, `pretrain`/`sft`/`rl`, validated immutable
configuration and optional corpus/recovery grants. It starts the **entire
training lifecycle**, including optimizer construction/load, batches/rollouts,
updates, reporting and recovery saves, inside the worker. The configuration
blob is interpreted as the existing typed training configuration for that kind;
it is not an extension allowing arbitrary commands. No tensor-operation RPC is
permitted. Final preflight for a job holding the device lease runs inside its
worker, not as a queued prerequisite waiting for that same lease.

Training has an explicit startup form: `train.pins` may be null only
for a prepared command entering an empty worker. Pretrain/SFT require a corpus;
inherited SFT additionally requires its saved `model_package_v1` grant. Inherited
RL requires the saved policy package and forbids a corpus grant. Its typed
preparation carries the exact saved tokenizer/profile and, on recovery, a separate
original frozen-reference package. Prepared records freeze source, final
tokenizer, profile and domain input facts. The approved
backend must verify inputs, construct/capture/probe and establish final RuntimePins
before optimizer construction. Null pins cannot replace an already loaded model
or bypass final preflight. This avoids inventing a receipt/definition for pending tokenizer
work. Ordinary resident operations still require exact loaded pins.

Fresh pretraining recovery carries a complete `model_package_v1` grant, including
definition/source/tokenizer and matching weight/optimizer state. Resident pinned
training retains the separate `checkpoint_v1` recovery format. The command
validator distinguishes these roles before worker execution; copied model
objects that bypass validation are not wire-contract evidence.

`generate` uses a prompt grant, seed, temperature and finite `max_tokens`.
The worker retains occurrence-specific explicit cache tensors only for that
request and clears them on completion/cancellation/disconnect. Bounded `tokens`
messages contain at most 32 IDs and optional matching finite nonpositive log
probabilities. The worker checks IDs against the bound tokenizer vocabulary.
`measure` follows the same pinned definition and parent device admission.

Only one numerical request executes in a worker at a time. A serving worker
keeps its loaded model between requests; cache state does not survive a request.
The parent retains ownership of the resident cache and admission lifecycle.
Queued numerical work with priority triggers serving-worker eviction and
verified tree/resource cleanup before another worker obtains that device.
Workers never import/reacquire a process-local copy of `llm.model.device`.

The [serving adapter](modular-serving.md) now connects saved packages to these
resident Load/Generate/Measure commands. Final pretrain/SFT profiles include batch
one through the configured batch; older receipts that cover only a larger fixed
batch cannot authorize one-row serving. Generate returns `generation_result_v1`
alongside bounded token-ID events. Parent result callbacks consume complete result
bytes before queue-priority eviction retires private grants. Declared models use
the validated full-prefix path; optional declared caching/context extension still
requires separate evidence before it can be enabled.

## Effects and publication

An `effect_request` contains an operation UUID and an `effect` tagged by
`operation`. Exact identifier fields preserve the existing client contracts:

| Operation | Identity and payload |
| --- | --- |
| `dataset_read` | Dataset UUID; optional bounded query grant. |
| `gym_catalogue` | No resource ID or payload. |
| `gym_create` | Opaque gym ID such as `math`/`general`; required creation-data grant. |
| `gym_read` | Run UUID; optional bounded query grant. |
| `gym_step` / `gym_finish` | Episode UUID; required tool-step/final-reply grant. |
| `gym_abandon` | Episode UUID; closes without grading. |
| `gym_status` | Run UUID; required status-data grant. |

Payload grants have the `effect_data_v1` format. Requests contain no actor, URL,
token, database object or service client. The parent resolves the resource
against the job, authorizes it under the initiating actor and uses the existing
Rust hop. Python services never call one another.

At most **one effect is outstanding**. The parent records write issuance before
calling Rust and keeps its operation identity across uncertain completion.
`effect_reply` distinguishes `completed`, `failed`, `uncertain`; unsuccessful
outcomes require an error code and cannot carry a success payload. Dataset/gym
reads may use their existing bounded retry policy. Already-issued gym writes
are never replayed after timeout, cancellation or restart. An interrupted RL
iteration is discarded; recovery starts new rollouts in fresh sandboxes.

`RuntimeArtifact` contains an immutable grant UUID, SHA-256, exact byte size and
versioned format. It contains no filesystem path. The parent resolves a grant
to an explicitly authorized read-only input or disposable output materialization;
it verifies actual bytes/hash/type and resource budgets before use. Source
snapshots are at most **262144 bytes**. Source/package/config/result formats are
distinct and role confusion is rejected before loading. Input grant lifetimes
end when their owning request/session is drained.

Training preparation uses separate `tokenizer_v1` and `span_table_v1` grants;
spans load as a numeric array with pickle disabled and verified dtype/row bounds.
SFT uses `loss_mask_v1` for aligned binary targets and `sft_probe_v1` for bounded
numeric NPY probes. A separate original-reference package is required for resumed
chat KL. The [SFT adapter](modular-sft.md) describes parent tokenizer authority
and reference memory admission.
`training_progress_v1` carries the existing measured training frame. Its bounded
grant is acknowledged through publication before removal, preventing telemetry
files from accumulating during long jobs. It creates no checkpoint/model row.
The worker also reports actual consumed batches separately from optimizer-step
telemetry so masked batches can refresh liveness without inventing updates.

`artifact_ready` carries publication UUID, complete staged artifact, completed
step and final flag. The worker flushes/fsyncs before announcing; the parent
verifies, publishes and durably registers before replying `artifact_adopted`.
At most **one publication is pending**. A rejection or missing acknowledgement
stops the run; the worker cannot keep writing unregistered checkpoints.
`completed` does not substitute for adoption. A forced stop retains the last
fully published recovery point and discards unfinished work. Existing
[checkpoint publication/reconciliation](checkpoints.md) remains authoritative.

## Deadlines, budgets and cleanup

`RuntimePolicy` bounds pending messages and retained progress to at most 64,
with coalesced progress normally no faster than every 250 ms. Progress queue
overflow requires backpressure/coalescing or explicit failure, never an
unbounded queue. Critical effects/publication/completion are not silently
dropped. Long training can report indefinitely through bounded retention;
64 is not a limit on the number of steps in a training run.

Default import/construction deadlines are 30 seconds each; validation,
generation and measurement deadlines are 120 seconds. Effects and artifact
adoption wait at most 30 seconds. A training watchdog defaults to 300 seconds
between actual completed batches/iterations, including skipped masked batches;
heartbeats or repeated phase labels do not renew it. There is no inherited
30-second deadline for an entire legitimate training run. Profile/deployment
policy may configure explicit larger limits within the declared bounds.
An idle serving worker is evicted after 300 seconds by default.

Cancellation is cooperative first, with **150 ms grace**, then process-tree
termination and a default **two-second reap/verification** period. Observer
disconnect does not cancel a durable job; an interactive disconnect terminates
its request/cache under this supervision. The parent retains the lease until
required execution and cleanup are verified. `completed`/cancel acknowledgement
alone is never a stopped worker. Missing or uncertain cleanup closes admission.
Startup reconciliation must verify old worker ownership/cleanup before starting
a replacement scheduler beside them.

The parent also drains outstanding effect and publication callbacks before
retiring session grants or releasing admission. Reaping the child alone cannot
stop an already-issued HTTP write or storage transaction. Session closure
atomically prevents new callbacks; draining uses each configured effect/adoption
deadline. If it expires, cleanup stays uncertain, the session directory remains
and admission is quarantined. A known finished adoption error still fails its
job without implying a live callback. Uncertain external writes remain
nonretryable.

`RuntimeResources` declares per-process virtual-address limits **separately**
from aggregate owned-tree resident bytes, allocator device bytes, process count,
open descriptors and per-artifact/total artifact bytes. All numerical budgets
are explicit before allocation. Owned processes include compiler/framework
helpers, not only the model PID; at most eight are permitted by this v1 policy.
Their creation must remain under the guardian's tree ownership. Custom source
cannot create arbitrary processes or detach sessions under the supported SDK.

`tree_memory_enforcement` is explicit: a delegated cgroup provides kernel tree
accounting/enforcement, while `preflight_and_monitor` uses conservative estimates
and owned-tree monitoring at most every 100 ms (default 50 ms). The latter may
overshoot between observations and is not a hard RSS guarantee. Per-process
`RLIMIT_AS` does **not** establish aggregate tree memory limits. If helpers or
resource accounting cannot be owned/observed, admission fails closed. Production
must enforce these fields, track cumulative output and revalidate estimates
against the requested architecture/profile before real allocations.

CUDA keeps the [existing allocator-margin policy](memory.md) and worker-side
device allocation caps. These declarations do not prove enforcement or cleanup.
Actual GPU hardware verification was **explicitly deferred by the user**;
production retains CUDA support and runnable GPU checks, with that hardware
result recorded as deferred rather than represented as a CPU test pass.

## Verification

```sh
engine/.venv/bin/python -m pytest --noconftest llm-engine/tests/test_architecture_runtime_contract.py
```

The tests use shared positive/negative fixtures, stale identities, UTF-8 byte
limits, duplicate fields, sender authority and finite numeric constraints.
The original contract checkpoint passed **129 checks**, including **119 shared
fixture cases**. Added training-bootstrap/output cases and current regression
results are tracked in the [delivery ledger](modular-architecture-evidence.md).
They do not launch a production supervisor or claim Rust/GPU integration.
