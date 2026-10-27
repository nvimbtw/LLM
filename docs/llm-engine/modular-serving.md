# Saved-definition serving and measurement

`api/serving.py` owns one reusable supervised model worker through the runtime
manager attached by `main.lifespan`. Source import, construction, tensor loading,
generation and measurement execute in `orchestrate/inference_worker.py`. The parent
owns admission, immutable grants, bounded token delivery and cancellation. It
does not construct custom models or deserialize their tensors. Legacy Transformer
artifacts retain their existing deliberate adapter.

The parent verifies the complete registered package and exact saved definition,
chooses a retained full-prefix receipt covering batch one and every trained
sequence length, and reserves archive/grant/extracted-package copies plus bounded
request/results before startup. Package hashing, archive copying and grant staging
observe interactive cancellation between finite reads. A request arriving while
another serving request owns the worker receives a busy refusal instead of an
unbounded queue or a second model allocation.

The child rebuilds the same immutable source/recipe and shared Parameters.
Before real construction/loading, strict meta capture charges unique tensor
storage, loading copies, conservative operation workspace, logits and host runtime
headroom against actual admission. Unsupported operation estimates refuse.
Serving loads weights alone; it does not restore optimizer moments, training RNG
or batch cursors from an intermediate checkpoint. Loaded weight dictionaries are
released after assignment. Import registrations and model tensors share one owner.

Final pretrain/SFT validation now includes batch one through the configured batch
size, in both train and eval modes. Earlier draft packages whose receipts cover
only a larger fixed batch are refused explicitly for one-row serving; method
presence or unrelated profile evidence cannot authorize another shape. This
change produces new receipts rather than rewriting old ones or saved definitions.

## Generation and request ownership

`model/generation.py` owns the canonical prompt, grammar/tool budget, token drawing,
UTF-8 rendering and existing summary fields. `model/decoding.py` owns one private
prefix/cache lifetime for serving, continuation and RL rollout. Declared models
use complete-prefix evaluation with explicit positions/masks. Legacy Transformers
retain their cache and request-local positional extension adapter. Optional
declared caches and declared context extension remain disabled until an exact
capability/profile adapter has its required capture/equivalence evidence.

Chat remains available to normal signed service callers. Model authoring/graph
administration keeps its separate scope checks. Declared generation validates
finite sampler settings, at most 65,536 draws plus bounded grammar closures, trained context and at most
32 tool calls before loading. Parent-owned prompt configuration is at most 4 MiB;
private in-memory grants/results are at most 8 MiB. Grammar closure stays inside
the saved context. The existing zero/absent seed behavior is preserved.

The worker emits vocabulary IDs through bounded IPC; the parent decodes them
without model execution. Its 64-chunk queue applies cancellable backpressure.
Complete summary identity, prompt/generated counts, decoded text and parsed tool
calls must agree with the actual stream. A malformed result retires the resident.
Final incomplete UTF-8 bytes flush as text without increasing generated token usage.

Disconnect first signals the request, then drains/joins its owned parent thread
and verifies worker cleanup. A cancelled HTTP waiter alone releases nothing.
Generation's scoped iterator drops prefix/cache/model references on success,
failure and cancellation. Completed results are consumed by the parent callback
before queue-priority eviction can delete their private grants. Idle expiry,
queued work and shutdown use the existing runtime manager/guardian ownership;
uncertain cleanup quarantines admission.

## Measurements and metadata

`model/likelihood.py` scores tokens once using the existing half-window overlap.
`model/continuation.py` preserves unconstrained seeded sampling, stop tokens and
trained-context limits. Both use the LM facts/logits boundary, evaluation behavior
and no-grad, restoring heterogeneous original modes. Scoring policy remains in
bench-engine. Declared measurements bound text to 65,536 encoded tokens, at most
32 prompts and at most 65,536 aggregate sampled tokens. Unsupported legacy layer
ablation returns null; no layer/head shape is invented.

`GET /models/{id}/definition` and `/graph` require an administrative actor/scope.
`architecture/package_metadata.py` reads bounded inventoried metadata and receipt
identities without reading tensors/source or consulting mutable catalog revisions.
The definition summary includes retained `validation_receipts`, allowing graph
queries to select their exact identity after original job pruning. Graph reads
load only the matching manifest and requested scope, verify hashes and retain
existing cursor/profile/scope checks. These reads never capture a new graph.
Receipt bytes cap at 8 MiB per package; graph bytes retain their aggregate engine
ceiling. Generic model listings report saved context/vocabulary and measured
unique counts; incompatible five-field legacy architecture remains null.

`GET /models/{id}/sources/{revision_id}` separately reads one frozen root YAML,
custom dependency or builtin descriptor from the same inventory. It verifies
the raw-byte revision digest and UTF-8/source budget without consulting catalog
rows, importing Python or loading tensors. Builtin descriptors are labeled JSON;
missing implementation bytes never resolve to the currently installed Python.

## Occurrence-aware attention

`GET /models/{id}/attention-points` explicitly discovers declared attention views.
For modular models this loads/reuses the numerical worker; reading a model's
ordinary metadata never does. Candidates require both saved `introspection` and
`attention` declarations and a matching `ModelIntrospection` or named-port
`BlockIntrospection` method. Discovery alone is not numerical capability evidence.
Opaque point IDs bind actual instance/local-point/call paths. Repeated calls to
one shared block have distinct IDs without duplicating parameter ownership.

`POST /models/{id}/attention` accepts `{text, point_id, head}`. Legacy
`{text, layer, head}` remains available for legacy artifacts and direct standard
composites. Generic recipes never acquire invented layer counts. Observations
reuse canonical attention projection/probability arithmetic without toggling
persistent legacy recording flags. Normal logits are also returned internally
during capture so memory planning cannot eliminate the later ordinary model work.

Before returning a point's first result or a larger sequence domain, its worker
charges model snapshots/autocast copies, strictly captures complete singleton and
range paths on meta and the admitted device, compares eager/captured observations
and unchanged ordinary logits, and checks deterministic causal normalized values.
Cancellation is checked between finite phases. Proofs retain only bounded metadata,
bind definition/profile/point/sequence bounds and last for that resident worker.
Weight/version/layout changes invalidate them. They do not authorize cache,
context extension, training or another view. No source or captured graphs appear
in these ordinary-user responses.

Matrix output remains at most `min(512, trained_context)` tokens and 256 heads per
modular point, with one fp32 little-endian base64 head matrix. Typed parent/browser
decoders check exact axes, probability values and complete runtime proof identity.
The frontend discovers on demand, names actual occurrences, clears stale matrices
on selection/text changes and owns abortable reads for the actual session/model.
The engine's interactive HTTP owner signals cancellation and drains its blocking
operation. Actual browser-through-Rust evidence is tracked separately from
adapter checks in the acceptance map.

## Verification boundaries

Numerical tests cover actual trained standard, shared-loop, nested-loop and
custom dropout packages through repeated chat/likelihood/continuation requests,
without restoring optimizer state. They explicitly verify production parent
loading refusal before using a trusted tensor-reader/construction seam, so they
do not certify process containment. Parent tests use an explicitly synthetic
worker seam for grant cleanup, reuse, queue eviction, backpressure/cancellation,
malformed results and signed public chat versus private definition reads.

Real parent/guardian serving cases are in `tests/test_serving_parent.py`.
The earlier sandbox socket restriction was removed by the user's harness
restart. Actual ordinary-user chats now cover legacy, standard, shared-loop and
custom-derived SFT/RL packages through Rust; measured loop occurrences carry
distinct attention proofs, and unsupported custom attention remains absent.
These live checks do not replace forced-failure/disconnect process tests.
Whole-model production caches remain disabled unless their exact capability
contract is supported and validated. CUDA hardware remains user-deferred.
Exact reports and final gate status are in the
[delivery ledger](modular-architecture-evidence.md).
