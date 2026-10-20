# Modular architecture SDK, version 1

Status: **ISS-49 contract**, not an implemented authoring endpoint or SDK runtime.
The decorator/builder implementation belongs to ISS-50. The declarations in
[`spec.py`](../../llm-engine/llm/architecture/spec.py), tensor capabilities in
[`interfaces.py`](../../llm-engine/llm/architecture/interfaces.py), and shared
[SDK fixtures](../../contracts/modular-architecture/v1/sdk/cases.json) freeze the
initial contract. See [YAML composition](architecture-yaml.md) and the
[capture evidence](architecture-capture.md).

## Declaring a block

An author supplies a standard `torch.nn.Module` class decorated with
`@block(BlockDeclaration(...))`. The public import surface is
`llm.architecture.sdk`: `block`, `BlockDeclaration`, `ParameterDeclaration` and
`TensorPort`. These SDK exports are the implementation contract for ISS-50;
that module is not introduced by this contract-only slice.

The [complete CustomScale source](../../contracts/modular-architecture/v1/sdk/custom_scale.py)
and its [metadata](../../contracts/modular-architecture/v1/sdk/positive-custom_scale.json)
show one learned scalar configured from an editable `gain` parameter. Its full
[causal model recipe](../../contracts/modular-architecture/v1/recipes/positive-custom-lm.yaml)
places the block on a residual branch. No source rewriting changes the gain:
the validated value is passed to `CustomScale(gain=...)` at construction.

`BlockDeclaration` has these exact fields:

| Field | Meaning |
| --- | --- |
| `sdk_version` | Integer `1`; booleans are rejected. |
| `name`, `title`, `description` | Stable local identifier and user-facing metadata. |
| `parameters` | Named typed constructor controls. |
| `inputs`, `outputs` | Named tensor ports, each declaring `dtype` and `shape`. |
| `capabilities` | Unique optional capability names; a declaration is not evidence of support. |

Resource/block names, instance IDs and lexical values match
`[a-z][a-z0-9_-]{0,63}`. Constructor/public parameter names and tensor port keys
use `PythonName`: `[a-z][a-z0-9_]{0,63}`, so they can be explicit Python argument
names. Hyphenated kwargs and implicit `**kwargs` adaptation are unsupported.
Parameters declare `type` as
`integer`, `number`, `boolean`, `string` or `enum`; optional fields are
`required`, `default`, `minimum`, `maximum`, `choices`, `title` and `description`.
Required values have no default. Numeric types do not coerce strings/booleans.
Enum alternatives are unique typed scalar values; a default must match an
alternative. Bounds and defaults are validated before constructor invocation.
The [typed-controls fixture](../../contracts/modular-architecture/v1/sdk/positive-typed-controls.json)
shows every control type. Only declared parameters become editor controls.

Each port has a rank of 1–8. A shape dimension is a positive integer or a named
symbol. `batch`/`sequence` come from the validated input domain, `vocabulary`
comes only from the selected tokenizer, and other symbols express consistent
named dimensions inferred/resolved by the engine. Dtype choices are `int64`,
`int32`, `bool`, `floating`, `float32`, `float16` and `bfloat16`; `floating` is a
family resolved to the execution precision. A block-specific validator owns
constraints such as head divisibility; arbitrary custom blocks do not inherit
Transformer or power-of-two restrictions.

`forward` accepts the declared input names as tensor arguments and returns
`dict[str, Tensor]` with exactly the declared output keys, including for a
single output. Extra/missing keys, non-tensors or incompatible shapes/dtypes fail
validation. A plain module returning one Tensor or a positional tuple requires
an explicit named-port wrapper. The SDK has one calling convention.

## Inspection and source policy

Saving, listing and reading a draft never imports source. Bounded inspection
loads its decorator declaration without instantiating the class, so required
constructor inputs are not needed to render controls. Inspection produces
`inspection_only`, never permission to accept a revision or start optimization.
Full validation receives resolved constructor arguments and input constraints.

The decorator returns the class and attaches validated declaration metadata.
Discovery belongs to a revision-local collector, not a global name registry.
Imports use immutable revision/content identity. Two revisions with the same
class/decorator name, including both in one model, must coexist without shared
code or mutable registry state. Builtin source is readable/copyable and read-only;
a copied custom block receives its own resource/revision identity.

Custom code executes with trusted administrator/operator authority inside the
supervised llm-engine worker. This is not an untrusted Python sandbox. Supported
imports are the pinned environment's approved PyTorch dependencies and versioned
public SDK/builtin references. Source cannot import HTTP, storage, database or
service internals, obtain browser credentials, install packages through the
editor, or call another service. The dependency closure is pinned in the model
package and verified before import.

Construct parameters and registered buffers once. Forward may use immutable
configuration, those tensors, declared inputs, explicit carry/cache and controlled
PyTorch randomness. It must preserve parameters/buffers and caller inputs.
Mutable globals/closures, hidden Python counters, unrecorded generators, external
I/O, forward-time parameter creation and Python scalars derived from input or
weights are unsupported. Nonpersistent buffers must be deterministically
reconstructible. A fresh local temporary is allowed to mutate when its storage
does not alias caller inputs or persistent state.

[Strict export alone can freeze ambient Python values](architecture-capture.md#unsupported-source-and-limits-of-export).
Source eligibility, alias/state checks, complete capture and numerical preflight
are separate gates. No opaque custom section counts as a complete graph.

## Model adapter and optional capabilities

Internal blocks use named ports. A built model's language-model adapter exposes
`LanguageModel.facts` and `forward(tokens, *, positions=None, key_mask=None)`.
`LanguageModelFacts` contains final `vocabulary_size` and trained `context_window`.
The canonical input is int64 `[batch, sequence]`; output is floating logits
`[batch, sequence, vocabulary]`. Any additional token dtype needs its own profile
and a deliberate adapter conversion. The adapter supplies logical positions and
an all-true mask when a recipe declares these inputs and the caller omits them.
An approved builtin wrapper may retain its equivalent unmasked fast signature.

Each live output must be causal and independent of padding/other batch rows.
The existing next-token shift and loss masking remain outside arbitrary blocks.
Evaluation must be deterministic and support autograd for RL policy updates;
calling `.eval()` does not disable gradients. Training dropout uses saved/restored
PyTorch RNG. The module owns parameters and train/eval mode through ordinary
PyTorch; protocols do not expose HTTP, storage or worker device gates.

Optional names map to these typed behaviors:

| Capability | Contract |
| --- | --- |
| `cache` | `CachedLanguageModel.prefill` / `decode` return `CachedResult(logits, cache)`. |
| `introspection` | `ModelIntrospection` lists explicit inspection points and measures one supported view. |
| `attention` | An introspection point of kind `attention`, with `[batch, heads, queries, keys]` values. |
| `recompute` | `RecomputableModel` advertises/configures legal execution occurrences between steps. |
| `token_rows` | `TokenRowInitializer` applies its explicit tied/untied initialization policy. |
| `context_extension` | `ContextExtensibleModel` derives request-local position state and evaluates an extended full prefix. |

A supported capability requires evidence for the exact definition/profile.
Unsupported capabilities carry a reason and remain absent from usable controls.
Python structural `isinstance(..., Protocol)` checks establish attribute presence
only. They do not validate numerical behavior or authorize a capability.

`ExecutionOccurrence(instance_path, call_path)` uses zero-based flow/loop indices.
One module instance can appear at several call paths. Its parameters share an
owner while cache/diagnostic/recompute occurrences remain distinct.
`RequestCache` carries a parent-supplied `CacheIdentity(request_id,
definition_digest)` and maps each occurrence to named tensors. `decode` receives
the expected identity explicitly and rejects a mismatch. Prefill creates fresh
state; decode returns new mappings/tensors without mutating caller cache. The
owning inference worker clears request state on completion, disconnect or cancel;
these tensors never cross process IPC.

Internal attention/hidden blocks use `CachedBlock.cached_forward(inputs, state)`
and return `BlockCachedResult(outputs, state)`, both named tensor mappings.
An empty state starts a prefix. The assembled model supplies separate state per
occurrence, owns request identity/admission and never infers cache sharing from
shared Parameters. Inputs/state are read-only and returned state is fresh. This
contract has no token/logit assumptions, unlike the whole-language-model protocol.

Cache support is optional: correct full-prefix generation is the baseline.
Cached logits must match full-prefix logits across all admitted positions,
padding and occurrence profiles. Separate support for cache and context extension
does not authorize their combination. Extended cached signatures need joint
capture/equivalence evidence and an explicit adapter. Extension leaves the trained
context and saved structural identity unchanged.

Recomputation must preserve values, gradients, RNG replay and explicit state.
It is a worker-owned memory policy, never an inference loop-count override.
Diagnostics preserve RNG/mode/persistent state and report unavailable measurements
as absent. Token-row initialization runs under no-grad, treats exact tied
Parameters once and reports initialized/skipped IDs. The standard builtin retains
legacy mean-row initialization. A model without this capability keeps its weights,
reports `skipped_unsupported`, and remains valid for SFT; callers never assume an
`embedding.weight` attribute.

## Versioning, limits and verification

A source revision pins its declaration, source digest, dependency closure and SDK
compatibility. Resolved semantics do not include device/profile graphs; receipts
separately bind runtime, inputs, modes and evidence. Catalog edits/archival and
validation-history pruning cannot replace accepted/enqueued/package bytes.
A changed parameter or tokenizer/profile requires appropriate new full validation.

Initial ceilings are 256 KiB each for source/YAML, 64 parameters, 32 ports,
128 modules/dependencies/expanded calls, nesting 8, 4096 graph nodes, 16384 graph
edges and 8 MiB per serialized graph. Source/storage and graph budgets do not
replace worker time/memory/device admission. Budgets are validated before the
corresponding expansion/allocation; unsupported or oversized drafts remain savable
within the draft byte budget but cannot authorize training.

The SDK fixture index contains declaration and capability positives/negatives;
the source example is syntax-checked only until ISS-50 supplies the SDK runtime.
The shared contract harness distinguishes schema/syntax checks from resolver-owned
negative expectations. CUDA/bf16 evidence is user-deferred and unverified; it must
remain visible in the later runtime/integration results without blocking CPU work.
