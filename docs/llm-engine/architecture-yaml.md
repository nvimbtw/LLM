# Architecture recipes, version 1

Status: **ISS-49 versioned contract**. Parsing/resolution/building belongs to
ISS-50. The engine owns all interpretation; Rust forwards authorized bounded
payloads and the frontend renders engine metadata. See the
[SDK](architecture-sdk.md), [`Recipe` declarations](../../llm-engine/llm/architecture/spec.py)
and [shared cases](../../contracts/modular-architecture/v1/recipes/cases.json).

## Complete examples

Every example is complete YAML with pinned references and explicit named ports:

| Example | Purpose |
| --- | --- |
| [Standard Transformer](../../contracts/modular-architecture/v1/recipes/positive-standard.yaml) | Existing Transformer as a composite, with all numerical architecture settings explicit. |
| [Expanded standard](../../contracts/modular-architecture/v1/recipes/positive-expanded-standard.yaml) | Embedding, independent layer stack, final normalization and exact tied head Parameter. |
| [Looped Transformer](../../contracts/modular-architecture/v1/recipes/positive-looped.yaml) | One shared core instance, repeated a fixed count with explicit carry. |
| [Custom causal model](../../contracts/modular-architecture/v1/recipes/positive-custom-lm.yaml) | Custom learned scaling on a residual branch, with the same token/logit interface. |
| [Branches and nested repeats](../../contracts/modular-architecture/v1/recipes/positive-custom-branch-nested.yaml) | Tensor subarchitecture, multiple outputs, explicit addition/concatenation and lexical loop carry. |
| [Reused subarchitecture](../../contracts/modular-architecture/v1/recipes/positive-reused-subarchitecture.yaml) | Two calls to one pinned subarchitecture instance, sharing its children. |

References resolve against the
[synthetic fixture catalog](../../contracts/modular-architecture/v1/sdk/catalog.json).
They describe compatibility contracts, not already accepted deployed catalog
resources. The custom source hash and subarchitecture recipe hash match their
fixture bytes. Other builtin identities are synthetic. ISS-53 supplies real
builtin revisions and verifies standard/looped numerical behavior.

The standard composite retains legacy initialization order, normalization,
GELU/bias/epsilon, adjacent-pair RoPE, fused QKV ordering, optional attention
temperature and tied embeddings. Primitive decomposition alone does not prove
identical seeded initialization: creating an extra head Parameter can consume RNG
before aliasing. ISS-53 must preserve deliberate construction/initialization order
and the pinned trace, not re-record it to hide a difference.

The required standard builtin policy follows today's exact RNG order:

1. Construct embedding, independent blocks in index order and final LayerNorm,
   retaining their current PyTorch constructor initialization draws. A block
   constructs attention QKV/output projections before its FFN projections.
2. Generate RoPE tables deterministically. Sweep modules in that same order:
   embedding and Linear weights use Xavier uniform; Linear biases become zero.
   LayerNorm uses epsilon `1e-5`, unit weight and zero bias. Attention projection
   biases are absent; both FFN biases remain present. GELU is the existing
   default (exact, not a new approximate activation).
3. After the sweep, reinitialize each fused QKV weight's Q, K and V thirds,
   in block order, with separate square-projection Xavier bounds. Preserve the
   earlier whole-QKV initialization draws even though these values are replaced.
   The fused weight layout is vertically concatenated Q/K/V.
4. The output projection is the exact embedding Parameter transposed in tensor
   computation. It receives no extra initialization or optimizer entry. An
   expanded tied-head wrapper must not add constructor RNG draws or initialize
   the alias target again. Resolve aliases before loading any state.

A shared looped core is constructed/initialized once. Repeated execution does not
repeat initialization. Distinct stacked children initialize independently in
index order. RoPE rotates adjacent `(2i, 2i+1)` channel pairs with base `10000`
unless the saved value overrides it; tables are reconstructible nonpersistent
buffers. Learned temperature, when enabled, is one zero-initialized log value per
head, applied as `exp(log_temperature)` scaling of Q before SDPA. Disabled
means no such Parameter. Each block is pre-LN attention plus residual, then
pre-LN FFN plus residual; final LayerNorm precedes the tied logits projection.
These are versioned builtin semantics, not initialization rules imposed on
unrelated custom blocks. ISS-53 must exercise constructor RNG, initial tensors,
forward/gradients and legacy artifact loading against this contract.

## Document fields and bindings

A recipe declares `schema_version: 1`, `interface`, `parameters`, `inputs`,
`outputs`, `result`, `modules`, `flow`, optional `aliases` and optional
`context_window`. `interface` is `causal_lm` for a trainable language-model adapter
or `tensor` for reusable internal composition. Resource title/description and
revision identity live in the catalog, outside YAML.

`inputs`/`outputs` map names to SDK tensor ports. `result` binds every declared
output to one final lexical value. Causal recipes require `tokens`, `logits` and
one resolved `context_window`; canonical token/logit ranks and dtypes are checked
semantically. The examples also explicitly declare `positions` and `key_mask`.
The LM adapter supplies logical positions/all-true masks when callers omit them;
ordinary tensor composition has no implicit external values or optional ports.

Public parameter names and tensor port keys are underscore-only Python names,
`[a-z][a-z0-9_]{0,63}`. Instance IDs and lexical values may also contain hyphens.
Public parameters use the same typed declarations as blocks. Whole-value
bindings such as `${dimensions}` supply one validated value. There is no eval,
arithmetic, interpolation or arbitrary object expression language. A binding like
`${dimensions * 2}` is invalid. Unknown fields, duplicate keys, unsafe YAML tags,
anchors/aliases and unresolved references are rejected. Source nesting/byte and
effective expanded-work limits apply before construction.

A module declaration has exactly `reference` and `parameters`. Its `reference`
has `kind: block|architecture`, `resource_id`, `revision_id` and `content_hash`.
Those identify immutable revision bytes; display names and a mutable “latest”
lookup never determine execution. Constructor parameters are validated against
that pinned block/subarchitecture's declaration.

The tokenizer owns vocabulary size, fingerprint and reserved token IDs.
`${tokenizer.vocabulary_size}` is the only vocabulary-size binding. Reserved ID
bindings use `${tokenizer.<name>_id}` and are resolved from the selected tokenizer;
architecture controls cannot invent or shift IDs. The `vocabulary` dimension in
an output shape refers to that same authoritative size. A public vocabulary
parameter or incompatible constant in a vocabulary-bearing builtin is rejected.

`context_window` is a positive integer or whole reference to one declared
structural parameter. A training page/profile cannot supply a conflicting second
context. Resume, SFT, RL and serving inherit structural values and loop counts
from the saved package. Context extension is a separately validated optional
position capability, not an arbitrary structural override.

## Calls, lexical values and branches

`call` names one declared instance. `inputs` maps its input port names to visible
lexical values; `outputs` maps output port names to fresh lexical names. A call's
returned named tensor keys must match its pinned declaration.

Each flow body has one lexical scope and single assignment. Recipe external
inputs are initially visible; each call introduces fresh outputs. Inner scopes
may read visible outer values, but cannot overwrite them or publish arbitrary
internal names. An outer flow sees a loop's final value only through its carry
`result`. A value must exist before use. Sibling branch execution order cannot
change values already read by another branch.

Branches are ordinary calls that consume the same visible value. Merging requires
an explicit block, such as `add` or `concat`, with stated shape semantics. There
is no implicit merge or in-place overwrite. Multiple inputs and outputs are
explicit port mappings. Arbitrary cyclic wiring and recursive subarchitecture
references are rejected; feedback is represented only by fixed loops.

## Shared repeat and independent stack

A `repeat` operation has `repeat`, `carry` and `body`. The count is a fixed
positive integer or a whole declared parameter binding. Every resolved count and
the complete expanded call product are bounded. It is not adaptive stopping or
an inference-time setting.

Each named carry declares these four fields:

| Field | Meaning |
| --- | --- |
| `initial` | Existing outer value used before the first iteration. |
| `input` | Fresh name visible inside each iteration. |
| `output` | Inner value to carry to the next iteration. |
| `result` | Fresh outer name bound to the final iteration output. |

Inner carry shapes/dtypes must agree on every iteration. Autograd runs through
all iterations with no implicit detach. Nested loops multiply effective work,
not the number of parameter owners. Calls to the same `core` module ID reuse its
weights at every iteration. Subarchitecture instances likewise own their children
once; repeated calls to that instance share those children.

A `stack` explicitly creates independent instances. It declares `stack`, `name`,
`template`, optional static `inputs` and `carry`. The template is one pinned
module declaration; generated children have stable paths `name/0`, `name/1`, etc.
For stack carry, `input`/`output` are template port names, not free inner lexical
values. Static inputs remain visible to every child; a port cannot be supplied
both statically and through carry. Each child has independent Parameters unless
an explicit compatible alias ties them.

The expanded standard fixture uses stack for ordinary layers. The looped fixture
uses one declared `core` and repeat. Equal iteration/layer counts do not imply
equal parameter counts or interchangeable cache slots.

## Parameter aliases and execution identity

`aliases` contain `{owner, target}` paths to exact compatible Parameters. The
expanded examples tie `embedding.weight` to `head.weight`. Both names reference
one Parameter object before weights load or optimizer construction. The optimizer,
unique parameter count, checkpoint and recovery order include that owner once.

Reject incompatible shape/dtype/device, cycles, overlapping trainable views and
conflicting optimizer policies for one shared owner. A transpose is tensor
computation, not a separately optimized view. Sharing is deliberate; equal tensor
bytes or Python display names do not make two independent instances shared.

Module instance paths identify parameter ownership. Flow/loop occurrence paths
identify actual computation, diagnostics and request caches. Nested iteration
positions are part of the latter. A shared attention block must retain separate
KV state at each occurrence unless a separately specified supported state-sharing
contract explicitly permits otherwise. Cache state never follows parameter
aliasing automatically.

## Resolution, validation and retained evidence

Resolution freezes the recipe revision, complete dependency closure, parameter
values, tokenizer/context facts and aliases before execution. Training controls
such as datasets, optimizer, schedule, batch size, precision and device are outside
the structural recipe. Their relevant profile values bind validation receipts.

Validation proceeds through syntax/declaration, resolution, construction,
complete strict capture and numerical/resource stages. It checks finite
loss/gradients, causal/prefix/padding independence, repeat versus unroll,
train/eval behavior and every enabled serving signature. A zero gradient on one
example is not alone proof of disconnection. An invalid/untraceable draft may be
saved within its byte budget; it cannot begin optimization.

Acceptance requires one fully bound successful example profile. It does not
validate every possible parameter combination. Pending-tokenizer previews are
`provisional`. An already accepted revision may be submitted with pending final
bindings, but final tokenizer binding and exact preflight run inside the admitted
training job before any optimizer step; they are not queued behind its own lease.
Stale receipts cannot authorize changed source, dependencies, parameters or
profiles. Acceptance/enqueue evidence is retained independently of prunable job
history and catalog archival.

Initial bounds are source/YAML 256 KiB each, nesting 8, expanded calls 128,
module/dependency counts 128, graph nodes 4096, graph edges 16384 and serialized
graph bytes 8 MiB. Repeated/subarchitecture effective work is checked before
expansion. These ceilings complement worker memory/deadline/admission limits.

The shared cases distinguish `yaml`, `declaration` and `resolution` failures.
The ISS-49 harness checks actual syntax/schema failures and verifies that semantic
negative fixtures remain well-formed test inputs. Rejecting unresolved values,
carry/alias problems, nested-work overflow, stale catalog pins and tokenizer
mismatches is the resolver's ISS-50 acceptance work; schema success does not
claim those semantic checks have run.
