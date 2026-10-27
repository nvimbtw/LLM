# Version-one builtin blocks and recipes

`llm/architecture/builtins` supplies read-only implementation snapshots for
ISS-53. It has no global registry, catalog writes, HTTP routes, checkpoint
dispatch changes or device admission. `builtin_definitions()` returns fresh
`BlockDefinition` snapshots for the resolver; `recipe_template(name)` returns a
complete version-pinned recipe. The shared SDK/YAML contract fixtures remain
synthetic examples and are not silently reinterpreted as these real revisions.

## Public blocks

Every block uses the [SDK named-port contract](architecture-sdk.md); tensor
dimensions and controls are available through its declaration.

| Block name | Parameters | Named inputs → outputs |
| --- | --- | --- |
| `token_embedding` | `vocab`, `width` | tokens → hidden |
| `normalization` | `width`, `epsilon` | hidden → hidden |
| `causal_attention` | dimensions, heads, context, RoPE base, learned temperature | hidden, positions, key_mask → hidden |
| `feed_forward` | dimensions, FF multiplier | hidden → hidden |
| `add` | none | equal-shaped left, right → hidden |
| `concat` | none | left, right with matching leading axes → hidden, concatenated on channels |
| `language_model_head` | `vocab`, `width` | hidden → logits |
| `transformer_layer` | attention controls plus FF multiplier | hidden, positions, key_mask → hidden |
| `standard_transformer` | layer controls, `layers`, `vocab` | tokens, positions, key_mask → logits |

Control names for attention/layer construction are `dimensions`, `heads`,
`ctx_window`, `rope_base`, `learned_temperature` and `ff_multiplier` where used.
Only these builtins require an even positive per-head dimension with exact head
divisibility. Custom components do not inherit that restriction.

The legacy tensor primitives remain the numerical authority: fused Q|K|V order,
Xavier initialization with separate QKV thirds, biased LayerNorm with epsilon
1e-5, biased GELU FFN, adjacent-pair RoPE, bias-free attention projections,
tied embedding/output weights and optional learned log-temperature are retained.
Explicit positions/masks preserve ragged and causal semantics. Derived RoPE
tables are reconstructible nonpersistent buffers; forwards retain no cache or
diagnostic tensors on the adapters.

The standard composite constructs the existing `Transformer` exactly once. It
preserves constructor RNG consumption, initial parameter values and the legacy
two-pass initialization order. Expanded independent modules initialize in stable
resolved instance-path order; they are deterministic but do not claim the same
seed-to-weight mapping as the historical composite. Copying identical parameters
between expanded and standard models preserves numerical behavior and gradients.

## Identity and compatibility

`compatibility_manifest()` records ABI `slate-transformer-v1`, SDK version and
SHA-256 of the exact component/composite/legacy primitive source files. Builtin
resource UUIDs are stable per ABI/name; revision UUIDs/content hashes depend on
the implementation manifest and declaration. Any changed implementation bytes
produce new revision identities. A builtin reference's `content_hash` is the
canonical manifest-plus-declaration digest, not the hash of a standalone Python
file. This identity is for builtin resolution. It must not be passed to
`SourceLoader`, which requires the exact UTF-8 source hash for custom revisions.
Saved packages must retain/check the manifest;
the current loader does not recreate unavailable historical builtin code.

These identities are explicit inputs for catalog/publication consumers. The
catalog adapter now seeds all nine readable builtin block drafts at startup and
queues one idempotent ordinary CPU validation per installed revision. Startup
neither constructs a model nor fabricates acceptance evidence. Only a valid
published result after verified worker cleanup can atomically accept the builtin
under its manifest-derived revision UUID. Failed/pruned bootstrap checks require
administrator retry/new validation; restart does not silently recreate them.
Four recipe templates are also seeded as read-only drafts. Their ordinary
validation jobs are queued only after every required immutable block revision
has been accepted. Live completion and startup adoption both schedule newly
ready recipe checks in the same caller-owned transaction. Recipe acceptance
still requires actual valid published evidence; dependency readiness alone is
not acceptance.

Builtin draft hashes identify the exact displayed Python file. Immutable builtin
revision hashes identify the complete manifest/declaration, including supporting
files. Their stored source remains the exact displayed implementation file;
validation separately pins the complete builtin root and verifies that readable
source matches the installed implementation. Builtin revisions never enter the
custom `SourceLoader`. Old accepted source/evidence remains readable after an
upgrade; executing or generating a copy of an unavailable historical builtin
fails rather than substituting current code.

`copyable_source(name)` supplies an eligible editable wrapper importing only
PyTorch, the public SDK and public builtin class. Internal implementation files
import the legacy primitive and are inspection artifacts; they are not directly
eligible custom-source copies. A copied wrapper receives a new custom resource,
revision and SHA-256 of its exact source bytes. Its public builtin dependency
must pin the compatibility manifest in the enclosing definition/package.
`GET /blocks/{id}/revisions/{revision_id}/copy-source` returns a typed `BlockCreate`
with eligible wrapper bytes for an exact installed builtin. It creates no resource;
the browser uses normal creation to give the copy its own identity. This admin-only
read is forwarded through Rust. It never reuses the builtin implementation digest
as the custom-source hash. All nine generated wrappers are exercised through `SourceLoader` and
compared numerically with the original builtins. Copies expose full-prefix ports
and deliberately omit optional capabilities until the author delegates/validates
them explicitly. The authoring UI copy workflow itself is not implemented here.

## Recipes and optional capabilities

Startup publishes a real deterministic byte-level BPE tokenizer for the concrete
recipe examples: all 256 bytes, the engine's fixed reserved markers and zero
learned merges. It learns no corpus and makes no dataset claim. Exact bytes give
it a content-derived UUID/full fingerprint. It is indexed as implicit, apart from
the normal explicitly selected training vocabularies. Existing display edits
survive restart.

The example artifact stages and fsyncs both `tokenizer.json` and its complete
`build.json` marker before atomic no-replace rename. Index failure leaves a
verifiable complete orphan for startup adoption; interrupted staging is removed
under exclusive runtime ownership. Conflicting existing bytes fail without
replacement. This actual tokenizer supplies only the one concrete builtin
acceptance profile. A pending training selection retains its own pending build
identity and must finalize/preflight inside its admitted training job; it never
inherits this example vocabulary as a placeholder.

Bootstrap request identities include draft versions. Repeated startup does not
duplicate checks, and restoring previously accepted exact recipe bytes reuses
the earlier revision rather than conflicting with an old request UUID. Old
revisions and evidence remain immutable across upgrades and rollbacks.

The [copyable examples](../../llm-engine/examples/architectures/README.md) provide
standard, expanded independent stack, shared loop and nested shared loop recipes.
Loops retain exact fixed counts for training/inference and complete autograd.
Embedding/head recipes express exact `Parameter` identity through explicit aliases;
they do not create a second trainable transpose. Vocabulary is tokenizer-owned.

`StandardTransformer` provides block-level `token_rows` and `recompute`
capabilities. Token rows use the legacy pre-update table mean, process unique
valid IDs once and report skipped IDs. Both standard and legacy SFT delegate to
the same `Transformer.initialize_token_rows` numerical owner. Recomputation selects exact layer
occurrences and retains gradients/RNG replay using PyTorch checkpointing.
The generic composition does not automatically promote child capabilities to a
whole-model protocol; downstream model adapters must explicitly bind and validate
them. Cache and extended context remain disabled; correct full-prefix inference
is the supported baseline. Standard composites, causal-attention blocks and
Transformer-layer blocks explicitly declare stateless attention inspection.
The [resident serving adapter](modular-serving.md#occurrence-aware-attention)
binds each actual nested/shared call and proves its measurement path before use.
Discovery itself is not numerical evidence. This changed implementation manifest
produces fresh immutable builtin revisions; all five example YAML files reference
the current manifest rather than reinterpreting older accepted revisions.

The custom source example supplies real SiLU-gated FFN arithmetic. Its complete
YAML binds an immutable source revision and builtins and exercises shared repeat.
It is usable through `SourceLoader`, resolution and construction; editing source
requires new revision identities rather than source rewriting by parameter controls.

## Verification

From `llm-engine/`, using the installed checkout environment:

```sh
../engine/.venv/bin/python -m pytest tests/test_architecture_builtins.py tests/test_architecture_core.py tests/test_dataflow_trace.py tests/test_attention.py tests/test_recompute_blocks.py
../engine/.venv/bin/python -m pytest tests/test_builtin_catalog.py tests/test_builtin_recipes.py
```

Tests compare seeded initialization/RNG, state tensors, residual boundaries,
ragged logits and gradients against the existing Transformer. They load the saved
tiny fixture into the standard recipe and compare the unchanged pinned logit
trace, compare loop gradients with explicit unroll, capture each real recipe
strictly, and execute the custom Python/YAML example. The trace is never rerecorded.
CUDA float32/float16/bfloat16 cases remain runnable and skip with the user's
hardware deferral when CUDA is unavailable. CPU evidence does not verify GPU
precision or production worker/API/training integration.

Builtin recipe adapter checks exercise real CPU numerical results with actual
published tokenizer bytes, complete source pins, idempotent acceptance, rollback,
publication/index failures and pending training pins. Their trusted numerical
fixture publication does not establish process containment. The separately
named supervised bootstrap test runs the production parent queue and children.
The live startup run also completed all nine blocks and four recipes, with
matching accepted receipts and retained graphs; exact CPU domains and identities
are in the evidence ledger. CUDA arithmetic and allocator checks remain deferred.
