# Modular architecture capture feasibility

Status: **ISS-48 CPU capture prototype**, 20 October 2026. This is evidence for
[FEAT-4](https://discord.com/channels/1538659898383011961/1562025656110284921),
not the production SDK or an authorization receipt. Runtime supervision is a
separate part of the same gate. No catalog, queue, artifact format or API changes
are introduced here.

## Method and supported domain

`tests/prototypes/capture.py` calls `torch.export.export(..., strict=True)`
explicitly. Tests inspect every operation, reject opaque Python calls, compare
real eager/exported results and gradients, and retain module/source metadata.
ATen operators, including fused scaled-dot-product attention, remain explicit
operations; this does not decompile their kernels. Structural tuple extraction
and symbolic integer arithmetic are permitted graph primitives. Compilation or
successful non-strict tracing is not a substitute.

Each capture states mode, tensor signature and bounds. Concrete captures reject
other shapes. Bounded symbolic profiles cover variable batch/sequence dimensions
without a capture for every generated token:

| Signature | Captured constraints | Evidence |
| --- | --- | --- |
| Full-prefix, unmasked | Batch 1–3; sequence 2–512 | One graph; endpoints and an intermediate length agree with eager. |
| Full-prefix, singleton | Sequence 1 | Independent signature, tested in train/eval and batch 1/3. |
| Masked/ragged cached prefill | Batch 1–3; sequence 2–8 | One graph, every sequence length tested; explicit positions/masks. |
| Masked singleton cached prefill | Batch 1–3; sequence 1 | Separate signature; explicit positions/masks. |
| Masked cached decode | Batch 1–3; query length 1; past 1–7; keys = past + 1 | One graph, every past length tested against complete-prefix logits. |
| Legacy unmasked cached decode | Query length 1; fixed `pos_offset` | Concrete minimum/maximum cache examples match eager/full-prefix. |
| Context extension | Explicit 16-row RoPE override; offset 15 | Captured/eager agree; saved 8-row buffers remain unchanged. |

These are profile-specific proofs. The 8-token profile uses vocabulary 32,
width 16, two heads, two layers and FF multiplier 2. The 512-token experiment
changes only context length. They do not establish an unlimited sequence range,
all possible block implementations or maximum production throughput. A later
receipt must bind the actual model parameters, context, vocabulary, input dtypes,
device, mode and every enabled signature. A model without a validated cache
capability uses full-prefix evaluation. Symbolic batch/sequence constraints and
concrete scalar/configuration specialization must be recorded and enforced.

Two small builtin changes make strict capture possible without changing model
arithmetic:

- Explicit positions keep eager `ValueError` checks, including a new rejection
  of negative indices. During capture, a tensor assertion checks every position
  against the actual RoPE table length. Invalid exported inputs raise
  `RuntimeError`. No input-derived Python scalar is frozen into the graph.
- SDPA receives `True if mask is None and seq > 1 else False`. Passing the
  symbolic expression directly as `is_causal` fails in torch 2.13 with
  `TypeError: ... is_causal must be bool, not SymBool`. The conditional lets
  strict export establish the sequence guard while preserving the same eager
  boolean and fused attention behavior. Singleton sequences have a separate
  capture because their causal flag differs.

The pinned dataflow fixture is unchanged. Evaluation with gradients, legacy
recomputation, ragged generation and prompt bounds remain separate regressions.

## Numerical and state evidence

The runnable matrix covers training/evaluation, minimum/maximum shapes, changed
runtime positions/masks, negative/overflowing positions, wrong shape guards,
causality under suffix edits, prefix equivalence, padding/batch independence,
finite next-token losses/gradients, and eager/captured backward equivalence.

Shared Transformer layers own one parameter set but retain separate execution
nodes and request caches. For example, the same shared QKV projection appears as
`linear` and `linear_4`, with separate `blocks.0`/`blocks.1` execution paths.
Parameter ownership comes from the exported parameter signature and actual
Parameter identities, not operation count or raw tracing object IDs. Both cached
occurrences retain distinct KV tensors. Captured inference agrees with full-prefix
inference for shared and independent layers.

A custom residual block uses explicit carried tensors and PyTorch dropout.
Nested fixed repeats match an explicit four-call unroll, including gradients.
Training/evaluation captures preserve RNG consumption and eager outputs; eval
remains deterministic with autograd enabled. Recomputation preserves dropout RNG
and gradients after weights are saved/reloaded. Probes preserve input carry,
state dictionaries and CPU RNG. The prototype does not execute optimizer updates
or publish model artifacts.

The prototype inspects ATen alias schemas before functionalization and rejects
writes through inputs, parameters, buffers and their views. Fresh local temporary
mask writes remain valid. Production validation must retain this check before
any transformation that hides writes, inspect mutation output signatures where
applicable, and compare persistent state/input snapshots during numerical probes.

## Unsupported source and limits of export

Strict export rejects demonstrated input-dependent and weight-dependent Python
scalar branches. These failures must report the responsible revision/source
location rather than becoming a partial graph.

Strict export **does not establish source eligibility**. A regression deliberately
shows a mutable ambient Python closure being frozen: exported output stays at
the old scale after eager output changes. Hidden Python counters, mutable globals,
unrecorded Python/NumPy generators, I/O, dynamic imports, weight-derived Python
specialization, persistent mutation and caller-input mutation are outside the
supported code contract. A successful export of ineligible code cannot authorize
training. Later source inspection/dependency policy and worker isolation must
reject these patterns; this experiment does not provide an arbitrary-Python
sandbox or claim such a production validator already exists.

Supported specialization is limited to the immutable resolved configuration,
explicit module mode and documented shape/signature guards. Controlled PyTorch
randomness is supported with the saved RNG state. Cache/carry is explicit
request state, independent from parameter sharing. A changed implementation,
profile or runtime needs new evidence.

## Measured envelope and initial policy candidates

The prototype bounds fixed expanded custom calls at **128**, and graph nodes at
**4096**. Excess fixed calls are rejected before constructing the block. Node
bounds apply after capture, so the worker must separately bound construction and
capture time/memory. These are conservative prototype ceilings, not permission
to allocate to an arbitrary tensor shape.

A fresh pinned-runtime CPU run generated:

| Example | Nodes | Graph JSON bytes | Export bytes | Capture + write seconds |
| --- | ---: | ---: | ---: | ---: |
| Independent two-layer Transformer | 129 | 158801 | 291017 | 3.0919 |
| Shared two-occurrence Transformer | 119 | 154555 | 276217 | 1.2965 |
| Custom block, 128 repeated calls | 901 | 1001496 | 1699196 | 3.9437 |

The first row includes cold capture overhead; these are single runs, not
benchmarks. Artifacts include source paths and are private inspection outputs.
The 128-call example remains below one quarter of the node ceiling. Suggested
initial source/recipe and per-graph serialized limits are **256 KiB** and **8 MiB**,
respectively; the next contract slice must enforce those before storing/exporting
and combine them with nesting, real tensor-allocation and worker memory/deadline
limits. No UI may expand all loop occurrences without its own bounded paging.
The prototype has no upload/API boundary and makes no production memory guarantee.

## Reproduction and verification

The usable checkout interpreter was the older directory's virtualenv:
`/home/x_x/Programming/Personal/LLM/engine/.venv/bin/python`. It reports
Python **3.14.8**, PyTorch **2.13.0+cu130**, CUDA runtime **13.0**, CPU execution,
and `torch.cuda.is_available() == False`. A separate CPU-only torch 2.13.0
interpreter also exists, but the recorded suite uses the former interpreter.
No packages were installed or pins changed.

From `llm-engine/`:

```sh
/home/x_x/Programming/Personal/LLM/engine/.venv/bin/python -m pytest tests/test_architecture_capture.py --log-level=WARNING -o junit_family=legacy --junitxml=/tmp/iss48-capture.xml
/home/x_x/Programming/Personal/LLM/engine/.venv/bin/python -m pytest tests/test_dataflow_trace.py tests/test_ragged_batch.py tests/test_recompute_blocks.py tests/test_prompt_bounds.py --log-level=WARNING
/home/x_x/Programming/Personal/LLM/engine/.venv/bin/python -m tests.prototypes.capture_evidence --output /tmp/iss48-capture-artifacts
```

The artifact command writes graph/source/module metadata JSON, portable `.pt2`
exports and `report.json` outside the repository. Tests do not replace the eager
execution path with exported arithmetic. The final capture matrix passed **36 tests in 42.71 seconds**. A focused
mutation/ragged recheck passed **5 tests in 5.55 seconds**, and the pinned
trace/ragged/recomputation/prompt-bound regressions passed **22 tests in
4.75 seconds**. Compilation and `git diff --check` also passed. Runtime
results belong to the companion gate report.

No GPU device or `nvidia-smi` was available. CUDA/bf16, actual GPU peak allocation
and device-assert failure cleanup remain unverified and must stay explicit in the
worker and final integration gates. CPU evidence does not certify GPU behavior.
No frontend/backend/database contracts changed, so those suites are outside this
prototype's verification scope.
