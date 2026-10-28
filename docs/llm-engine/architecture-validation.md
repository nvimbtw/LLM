# Production architecture validation

`llm.architecture.worker:ValidationBackend` is the child-only backend for runtime
`Inspect` and `Validate`. It consumes an immutable `ValidationSnapshot` artifact,
executes declarations/construction/capture/probes inside the admitted process and
returns a typed `ValidationOutput` as `validation_result_v1`. It has no catalog,
network, scheduling or device-gate clients. The parent owns deadlines, cancellation,
process-tree cleanup, durable rows and artifact publication.

## Input and lifetime

`validation_records.py` defines bounded request/source snapshots. The parent
captures the exact `ValidationRequest`, subject source or YAML, all custom
dependency source bytes and installed builtin references. Hashes, duplicate pins
and source sizes are checked without imports. The worker loads only these bytes;
it never resolves a mutable latest catalog revision. Custom imports retain their
`SourceLoader` owner through construction, capture and probes.

Inspection loads a block declaration without constructor values, returns
`inspection_only` and produces no receipt/graphs. Full validation requires the
actual pinned torch runtime, admitted device, explicit precision and both train
and eval modes. Canonical LM tokens are int64. Recipe context remains authoritative;
a preview cannot override it. Pending tokenizer profiles return provisional
source/parameter metadata with resolution, construction, capture and numerical
stages explicitly skipped. No placeholder vocabulary, final identity or full
graph evidence is fabricated; these reports cannot authorize acceptance or
optimizer execution.

The worker first constructs disposable meta weights to check parameter/buffer
budgets, then constructs fresh probe weights on the admitted device. Construction
and all probes preserve host/device RNG. Probes preserve module modes, existing
gradients, parameters, buffers and inputs. No optimizer step occurs. Completion
clears compiler references, collects cycles and releases unused CUDA allocator
state; the parent still retains admission until the process cleanup certificate.

## Capture and numerical domain

Every enabled full-prefix domain is partitioned into explicit singleton and
non-singleton batch/sequence specializations. A separate strict torch export is
produced for every partition in train and eval mode. Non-singleton axes have
bounded symbolic dimensions; endpoint and interior examples execute against the
captured graph. Actual export range guards, domain bounds, strict mode and
signature are recorded in the graph manifest. A successful single sample never
authorizes an uncaptured shape domain.

The returned receipt profile records the actual strict exports as canonical,
labeled mode/signature/scope/domain/guard records. Empty request guards ask the
worker to discover them; supplied guards must match exactly. Static singleton
specializations remain explicit even without symbolic range guards. Manifest
identity is finalized from this discovered profile before hashing publication
bytes. Package/publication readers verify receipt identity, strict mode/guard
coverage, referenced scope bytes, node/edge identities and child scopes without
executing model arithmetic.

Input/parameter/buffer alias writes are rejected from ATen schemas before
functionalization; eager tensor-version checks and value snapshots are separate
guards. Opaque Python/module calls and unknown structural graph operations fail.
Fake tensor allocations use symbolic upper bounds, and unknown size estimates
fail. Initial validation policy caps persistent tensors and conservative captured
tensor output bytes at 64 MiB, graph nodes at 4096, edges at 16384 and aggregate
graph JSON at 8 MiB. Worker address/RSS/deadline limits apply independently while
constructing or discovering a graph. These conservative bounds do not claim an
exact training-memory estimate or replace training admission.

Real numerical checks compare eager/captured values and gradients with identical
RNG draws. Shared-loop execution is compared with the complete captured unroll.
LM probes use finite next-token loss; singleton/tensor-only floating profiles use
a differentiability probe. Zero gradients are permitted, while a missing path to
an intended trainable parameter fails. Stateless integer-only tensor routing needs
no artificial floating loss. Evaluation additionally checks determinism, changed
suffix causality, prefix equality, batch independence and padding independence.
These are bounded supported-code regression checks, not proofs about arbitrary
Python executions or useful model quality.

Ordinary blocks and reusable `interface: tensor` recipes require all synthetic
tensor inputs and currently validate exact input shapes; ranged tensor profiles
fail explicitly. Tensor recipes resolve and capture their complete pinned child
closure, including multiple outputs and repeated shared subarchitectures, without
requiring a tokenizer or inventing language-model context. They retain both
architecture and operation graphs. Their declared port ranks, dimensions and
dtypes are checked. Causal recipes still require finalized tokenizer bindings.
Cached/prefill/decode signatures are explicitly
rejected until a whole-model cache adapter has its own complete capture/equivalence
implementation. The current builtins use the correct full-prefix baseline; no
unsupported optional capability is advertised by validation.

## Graphs, identities and publication handoff

`GraphArtifacts` keeps canonical compact UTF-8 JSON scopes in a map keyed by the
SHA-256 of the exact bytes. Architecture scopes preserve instance ownership,
named input/output edges, independent-stack links, nested repeat boundaries,
initial/input/output/result carry maps and explicit repeat feedback. Loops remain
hierarchical; the operation captures separately retain actual expanded execution.
Captured operation nodes expose tensor metadata, parameter-owner references,
module occurrence metadata and source locations where capture supplies them.

The manifest has this shape:

```json
{"schema_version":1,"identity_hash":"…","complete":true,"graphs":[
  {"view":"operations","mode":"eval","signature":"full_prefix",
   "scope":"domain-0/part-0","sha256":"…","guards":[],"strict":true,
   "domain":{"batch_min":1,"batch_max":1,"sequence_min":1,"sequence_max":1}}
]}
```

`ValidationReceipt.graph_manifest_hash` is the raw-byte SHA of this canonical
manifest; the manifest and every referenced scope blob must be retained exactly.
ISS-55 stores these same bytes in the package graph inventory. This evidence is
profile-specific and remains separate from semantic model-definition identity.
The dependency digest uses direct root recipe path pins shared with catalog and
package checks; complete transitive dependency bytes are separately frozen.
Parameter digest hashes resolved constructor/public values including defaults.
For copied block source, direct public-builtin pins form the dependency digest.
Structural import inspection handles equivalent Python whitespace without
executing source. Snapshots containing public-builtin imports must carry frozen
implementation pins, which the child verifies before any custom source import.

`read_fragment` reads published scopes only. Cursors bind manifest hash, profile,
view, mode, signature and scope. Tampering, stale identities or cursor replay
fails. Node/edge page bounds remain explicit; reading never constructs a model
or launches capture. Root scopes expose lazy child scopes for partitions and
hierarchy.

A worker `valid` report is a numerical result awaiting parent publication. The
parent must verify worker cleanup and atomically retain the complete report,
receipt and graph inventory before recording acceptance evidence. Failed or
interrupted requests do not retain authorization receipts; cancellation propagates to the
runtime rather than becoming a completed-invalid verdict. Invalid source retains
bounded stage/resource/revision diagnostics and source locations when available.

## Durable queue and acceptance adapters

`api/validation_jobs.py` locks the selected resource, freezes saved bytes and the
complete bounded dependency closure, and queues `model_validation` in the normal
execution tables. Actor/request deduplication is part of that transaction.
Final tokenizer facts are checked against bounded owned artifact bytes and their
complete build marker before queue commit. Full-byte hash, contiguous vocabulary
size and requested reserved IDs are authoritative; browser guesses and invented
derived provenance cannot become final bindings. Tokenizer fact reads require no
model construction. Inspection snapshot grants cap at 256 KiB including JSON overhead; full snapshots
cap at 8 MiB. Retry copies these bytes under fresh request/job identities rather
than consulting the current draft or generating a model output UUID.

`api/validation_handler.py` runs `ValidationBackend` through `WorkerManager`,
translates bounded child progress into typed validation stage events and reads
the complete result while its grant owner is alive. Only after the session exits
with verified cleanup may `validation_artifacts.publish_after_cleanup` stage,
fsync and atomically rename the canonical snapshot/report/graph publication.
Its 16 MiB publication ceiling is separate from the worker graph/result ceilings.
Existing publication bytes are immutable; a conflicting result cannot replace them.

The runner serializes cancellation and terminal completion with a job row lock.
Done plus retained valid receipt commit together after verifying the publication
matches the queued snapshot/result. Cancellation keeps no report in its durable
result. Inspection, provisional and invalid outcomes may complete but retain no
authorization evidence. Typed terminal events follow the terminal transaction.
Registry observer disconnect remains independent from the job cancellation event.

`api/routes/validations.py` supplies admin-only submission, report, bounded graph
and exact-draft acceptance routes. Valid evidence retains its publication after
job pruning. Graph reads verify complete inventory and receipt/query identity
without model arithmetic. Acceptance checks installed builtin compatibility,
current draft/version/hash, dependency digest and retained receipt in its caller
transaction. A copied source's saved compatibility manifest also governs later
recipe validation and package loading.

`GET /validations/{id}/sources/{revision_id}` reads only the published snapshot's
root/custom dependency text, including completed-invalid drafts. Draft source
locations use the same deterministic private revision identity as worker imports.
Reads neither re-inspect source nor follow current catalog heads. Published
builtin roots retain their original readable Python; unavailable builtin dependency
text returns 404 rather than substituting installed implementation bytes.

`api/validation_recovery.py` runs after exclusive worker reconciliation and before
normal orphan failure handling. It removes unpublished staging and examines at
most 100 interrupted running validations. Complete matching parent publications
can be adopted transactionally; cancelling jobs are never adopted. Missing,
corrupt or mismatched publications remain unsuccessful without preventing
reconciliation of another job.

Builtin block startup captures exact readable implementation Python plus its
complete manifest-derived root pin in ordinary validation snapshots. The worker
verifies both before constructing the installed builtin; it never passes an
implementation digest to the custom source loader. A fully valid published
builtin result records evidence and its supplied immutable revision in the same
terminal transaction. All nine builtin roots have numerical fixture coverage;
four builtin recipe drafts use a real published byte-alphabet tokenizer for their
own concrete checks after accepted block dependencies exist. The live startup
run completed all thirteen validations and accepted their exact receipts; full
identities and CPU shape domains are recorded in the evidence ledger.

## Verification

From `llm-engine/`:

```sh
../engine/.venv/bin/python -m pytest tests/test_architecture_validation.py
../engine/.venv/bin/python -m pytest tests/test_validation_queue.py tests/test_run_routes.py tests/test_chain.py
```

The matrix covers real standard/shared/nested/custom captures, singleton/ranged
shape endpoints, malformed/state-mutating/nonfinite models, stochastic or
future/batch-coupled evaluation, port contracts, integer-only routing, allocation
and graph budgets, stale receipts/cursors, hierarchy topology and actual supervised
inspection/full/failure/hung-forward cleanup. CUDA fp32/fp16/bf16 cases remain
runnable and are explicitly skipped on this CPU machine at the user's request.
GPU allocator/precision and live catalog/HTTP/training publication integration
are separate evidence; CPU numerical passes alone do not certify them.

`test_validation_queue.py` exercises signed ASGI handlers, SQLite transactions,
trusted numerical fixture publication, retry, cancellation, restart adoption,
exact acceptance and retained evidence. Its explicit worker seam tests establish
publication ordering only. The separately named supervised HTTP test launches
the actual production worker. Current process/database/browser reports and the
live draft→validation→acceptance/cancellation flows are recorded in the evidence
ledger. Neither fixture/seam checks nor SQLite prove process containment or
PostgreSQL row-lock races. The current full validation module passes 23 CPU cases
with three user-deferred CUDA skips, including reusable tensor recipes.
