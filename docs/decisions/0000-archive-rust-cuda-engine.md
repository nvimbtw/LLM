# Archive: the Rust CUDA engine

These describe `backend/crates/llm`, the hand-written CUDA implementation the
PyTorch engine replaced. The code is gone; the findings are kept because the
failures they record are the kind a reimplementation walks straight back into.

**Nothing here describes the current system.** For that, read `engine/llm/`.

## The attention backward, and why it was dead

`cuda_attention_backward_batched` was wrong from the first training commit
(`fa50006`) until it was fixed. Every model trained before that fix has a **K
projection frozen at its random initialization** in every layer, and Q/V
gradients computed from the wrong layer's activations. Weights from before the
fix are not salvageable — the attention was never trained. Re-pretrain.

Three defects, all in that one function:

1. **The forward's Q/K/V live in a single shared `attn` scratch**, which each
   later layer's forward overwrites. Backward runs after the whole forward, so
   it read layer `layers-1`'s Q/K/V no matter which layer it was differentiating.
   Only the last layer was ever right. (`P_all` and `X` are per-layer caches,
   which is why those were fine and the bug was not obvious.) The forward now
   writes Q/K/V into **per-layer `Workspace` caches** (`q_all`/`k_all`/`v_all`)
   and the backward reads those. Nothing about Q/K/V may move back into `attn`.
2. **`dQ` and `dK` were f32×f16 `GemmEx` calls.** cuBLAS requires both operands
   share a data type and rejects the combination outright, so those two GEMMs
   never ran: `dK` stayed at zero (hence the frozen K) and `dQ` kept holding
   `dout` from its use as a temp buffer, which is why `wq` got a large *wrong*
   gradient rather than none. The f16 caches fix the types as a side effect.
3. **Nothing checked a single cuBLAS status in the file.** A failed cuBLAS call
   writes nothing and returns a host-side code; training continues and the loss
   curve still goes down. `CUBLAS_CHECK` now wraps every GEMM and aborts. The
   check is free — the status is a host return value and forces no sync.

**The guard is `tests/attention_learns.rs`.** It trains on a random block
repeated twice: the second half is fully determined by the first, but *only*
through attention, and a fresh block each step makes it unmemorizable. Broken
attention plateaus at `ln(vocab)`; working attention reaches `ln(vocab)/2`
(measured: 4.17 vs 2.07 at vocab 64). An ordinary "loss decreases" gate does not
catch this — loss decreases on prose whether or not K is ever trained, which is
how this survived for the life of the project.

**Caching Q/K/V, not recomputing them.** The first fix recomputed Q/K/V in the
backward with three extra batched GEMMs, which cost ~1.4% throughput. The forward
already computes them, so they are cached per layer instead: V lands in `v_all`
straight from its GEMM (already f16, so free), and one elementwise kernel
(`qk_cache_f16_kernel`) emits the f16 `q_all`/`k_all` the backward needs while Q/K
stay f32 for the score GEMM — so no forward numerics change. Cost is
`3 × layers × ctx × d × 2` bytes (~4.7 MB at d256/L6/ctx512), and it bought the
1.4% back: 14.57 ms/step mean over 7 runs, against 14.80 recomputing and 14.60 for
the pre-fix (broken) code.

The caches are **allocated at `max_s` but addressed with runtime-`seq` strides**,
exactly like `head_p_all`. Forward and backward see the same `s` within a step, so
the packing agrees; a short final chunk leaves the allocation's tail untouched.


## Backward scratch must be sized by `max_s`, not by `d`

`ln2_dgrad` (the LN2 backward's `dx`) is an f16 buffer at `max_s * d`, matching
`b_sd[1]`, which is what LN1's backward has always written into. It used to
share `b_dd` with the `wo` gradient — an f32 `d * d` buffer — which fit only
while `s <= 2 * d`. At d 256 that is ctx 512 *exactly*, so every run up to
Shakedown 1 sat on the ceiling and nothing showed; Balanced 1 at ctx 1024 wrote
512 KB into a 256 KB allocation on every layer of every step and its weights are
not usable.

The overrun does not fault and the loss keeps descending — it lands in the
file-static `attn`/`ff` scratch, which is live inside the same layer's backward,
so it silently corrupts that layer's attention gradients. Measured cost at
d256/ctx1024: 0.32 nats of divergence by step 8 against ~1e-5 run-to-run noise,
and the loss stops descending monotonically. That reads as a bad hyperparameter,
which is exactly why it survived.

`Workspace::assert_scratch_fits` now checks every such buffer in **bytes** at
construction, and it is a hard panic rather than a `debug_assert` — a silent
out-of-bounds device write is not something to leave to a release build.
`tests/scratch_bounds.rs` is the regression guard, at `ctx = 4 * d`. **A buffer
a kernel fills with an `[s, d]` block must be allocated at `max_s`**; sharing
one between two element types is what caused this.

**The GPU scratch is global and single-threaded.** `attn`/`ff` are file-static
and the cuBLAS handle is shared, so two `Workspace`s stepping concurrently
corrupt each other. The training loop is single-threaded so this is fine in
production, but **GPU tests must run with `--test-threads=1`** or they fail
spuriously inside cuBLAS.


## FlashAttention

Attention is fused. The `[h, seq, seq]` score matrix is never written to memory:
`flash_fwd_kernel` computes scores, softmax and `P @ V` a tile at a time in
shared memory, keeping a running maximum and denominator in registers, and saves
only a **per-row logsumexp `[h, seq]` f32** for the backward. `head_p_all` — the
largest activation in the trainer — is not allocated at all.

The replacement is surgical. The QKV projections, RoPE, the KV-cache write, the
`dX` GEMMs and the `dW*` GEMMs are untouched cuBLAS; only forward steps 4-6 and
backward steps 1-5 changed, and the flash backward writes `dQ`/`dK`/`dV` in the
same `[h, seq, hd]` f16 layout, so `rope_dqk_kernel` and everything after it does
not know which path produced them.

Measured on the 3060, `tests/bench_step.rs::bench_flash_attention`:

| shape | ms/step | device MB |
|---|---|---|
| d512 h8 L8 ctx1024, cuBLAS | 55.7 | 360 |
| d512 h8 L8 ctx1024, flash | **43.7** (1.28x) | **130** |
| d512 h8 L8 ctx2048, cuBLAS | 128.0 | 1156 |
| d512 h8 L8 ctx2048, flash | **76.4** (1.68x) | **262** |

Both numbers improve superlinearly with the context window, and for the same
reason: the removed tensor is `layers × h × ctx²`. The speedup is bandwidth, not
arithmetic — the old path wrote an f32 score plane, read it back, wrote an f16
`P` and read that three more times, then in the backward wrote `dP`, read it,
wrote `dS` and read it four separate times (scale, mask, the dQ GEMM, the dK
GEMM). That was ~2.5 GB/step at the ctx-1024 shape for a tensor nothing outside
attention looks at. The causal mask also becomes structural: whole `Bc` tiles
above the diagonal are never visited, where the score GEMM computed them in full
and then discarded half.

**`LLM_ATTN_FLASH=0` puts attention back on the cuBLAS path** without a rebuild.
Both are compiled in; that is what makes the A/B tests possible and is the escape
hatch mid-run. `training::flash` is the Rust face of the switch. A `Workspace` or
`InferSession` reads it **once at construction** and allocates a different set of
buffers accordingly, so flipping it under a live one leaves it holding buffers
the kernels will not write — set it first, then build.

- **`head_dim` must be 16, 32, 64 or 128** — the values `d / heads` takes when
  both are powers of two. Anything else falls back rather than being
  approximated, and the fallback is by *shape*, independent of the switch:
  `ensure_attn` sizing its legacy buffers off the switch alone left a d16/h2
  model running the cuBLAS score GEMM against a null pointer.
- **The wmma accumulator layout is not specified by the API**, and everything
  here depends on the Ampere one (`ACC_ROW`/`ACC_COL`). A lane owns exactly two
  rows, which is why the online-softmax state is six registers instead of a
  shared array; the four lanes sharing a row differ only in the low bits of the
  lane id, which is why a row reduction is two `__shfl_xor_sync`s; and the store
  goes through the map elementwise anyway, which is why writing `P` *transposed*
  is free and `dV = PᵀdO` / `dK = dSᵀQ` need no transpose kernel. It is probed
  once at first use and `abort()`s on mismatch, for the same reason
  `CUBLAS_CHECK` aborts.
- **The backward is two passes, not FlashAttention-2's one.** FA-2 grids over
  key blocks and `atomicAdd`s into `dQ`; that is faster and **not reproducible
  run to run**. Two passes (K-blocks for `dK`/`dV`, Q-blocks for `dQ`) recompute
  `Q Kᵀ` and `dO Vᵀ` for ~40% more backward FLOPs on a budget that is not the
  bottleneck, and every output is written once by one block. Determinism is the
  argument: attention has shipped silently wrong here twice, and the test that
  catches that class is an exact A/B.
- **Every `[h, *, hd]` buffer a flash kernel touches is allocated at `max_s`
  rounded up to 64 rows, and zeroed.** Both halves matter. A warp whose 16-row
  slab sits past `seq` still executes its `mma_sync` — a *lane* may never skip
  one — so the load reaches the rounded extent. And the dK/dV pass contracts
  whole slabs of `Q` and `dO` against `P`/`dS`, which are zero past `seq`
  *arithmetically*, not structurally: `0 * inf` is NaN, and the ±65504 clamp on
  the store turns that NaN into a large **finite** number, because `fmaxf`
  returns its non-NaN operand. The gradient then comes out plausible and wrong.
  The invariant is "the tail is finite", not "the tail is zero" — a later shorter
  chunk leaves a longer one's real values there and they multiply out to zero
  perfectly well. Only never-written memory is dangerous.
- **`inv_scale` is computed on the host** and passed to both kernels. The file is
  built `--use_fast_math`, which implies `-prec-sqrt=false -prec-div=false`, so a
  device-side `1/sqrtf(hd)` is a different number — and the old backward *divided*
  by `sqrt(hd)` where the forward multiplied by its reciprocal.
- **Attention entropy moved into the forward.** `cuda_attention_entropy` read
  back the softmax plane that no longer exists, so the flash forward accumulates
  `lse - t/l` instead, where `t` is one extra running sum beside the denominator.
  Expect a small step in that telemetry series at the switchover: the old kernel
  dropped terms below `1e-9` and read `p` back through f16, and this does
  neither. It is a discontinuity, not a regression.
- **`cuda_attention_decode` got its own `[h, seq]` scratch.** It scores one query
  row against the cache and had been borrowing a `seq`-th of the `[h, seq, seq]`
  plane — merely wasteful before, and a null dereference once that plane stopped
  being allocated.

**The guards are `tests/flash_attention_matches_cublas.rs`.** The sharp one is
`the_flash_backward_reproduces_f64_attention_gradients`: `dQ`/`dK`/`dV` against a
naive f64 CPU computation, ~4e-4 relative L2, stable across shapes. It compares
against *ground truth* rather than against the other GPU path, which is the only
way to pin the semantics two implementations happen to share — the max
subtraction, the scale inside the exponent, `col <= row` rather than `col < row`.
`the_attention_gradients_match_the_cublas_path` runs a whole training step and is
deliberately the looser of the two: at a fresh model ‖g‖ falls roughly as 1/√seq
while the absolute difference stays flat, so its *ratio* grows 8x across
sequence lengths with nothing about the agreement changing. Read the absolute
difference there, not the ratio.

`tests/attention_learns.rs` remains the functional guard and passes on both
paths — with the absolute positional table gone it is a pure RoPE-plus-attention
copy task, and it reaches the `ln(vocab)/2` floor either way.


## Batching the `dX` GEMMs was tried and is slower

Backward step 6 (`dX = Σ_i dQ_i wq_iᵀ + dK_i wk_iᵀ + dV_i wv_iᵀ`) is `3 * h`
separate un-batched `cublasGemmEx` calls accumulating into one `dX` with `β=1` —
24 launches per layer at `h=8`, 192 per step. It is the most obvious remaining
un-batched loop in the trainer and it looks like free money. It is not.

A strided-batched GEMM gives every batch element its own output, and these all
want the same one, so batching means writing an `[h, seq, d]` temp and reducing
the planes. That reduction moves ~8 MB per term per layer that the loop never
touches. Measured at d512/h8/L8/ctx1024, three runs each way: **43.0 ms/step for
the loop against 44.0 for the batched form** — the rewrite is 2.3% *slower*, and
it also wants an 8 MB scratch buffer. At d256/h4 the difference flips sign
between runs, i.e. it is noise there.

The lesson generalizes past this one loop: **launch count is not the bottleneck
in the backward**, because the launches overlap with GPU work. Bandwidth is. Any
change here needs a measurement, not a launch-count argument.


## Positional encoding: RoPE

Position is applied by **rotating Q and K inside attention**, not by adding a
table to the token embeddings. `rope_qk_kernel` rotates each adjacent pair
`(2i, 2i+1)` of a head's dimensions by `pos * 10000^(-2i/hd)` in the forward;
`rope_dqk_kernel` applies the inverse to `dQ`/`dK` in the backward. Because a
score is a rotated Q against a rotated K, what survives the dot product depends
on the *difference* of the two positions — relative position, with no
parameters. `Model::init` rejects an odd `head_dim`, which the adjacent pairing
requires.

This replaced a learned absolute table that was **never trained**: it had no
gradient buffer, no Adam master and was never synced back, so every model before
v4 ran on frozen Glorot noise for position. Removing it costs nothing learned and
takes `ctx * d` floats out of every model file.

- **The rotation is applied in three places** and they must agree: the batched
  forward (training *and* inference prefill, which now share
  `cuda_attention_forward_batched`), and `rope_decode_kernel` for a single new
  token against the KV cache. The first two are literally the same launch of
  `rope_qk_kernel` — its f16 cache outputs are nullable, so inference reuses it
  without caching. The decode kernel is separate only because Q and the cached K
  row no longer share an index space, and it calls the same `rope_cos_sin` /
  `rope_rotate` device helpers so the arithmetic still has one definition. See
  "Inference: one device-resident session".
- **Angles are reduced into `[-pi, pi]` before the trig call.** This file is
  compiled `--use_fast_math`, which substitutes intrinsics whose argument
  reduction degrades past about one turn, and the angle reaches ~1e3 radians at
  long context and low pair index — exactly the long-range positions RoPE exists
  to encode.
- **The cached Q/K are the post-rotation ones**, because the backward needs `K'`
  and `Q'` (`dQ' = dS @ K'`, `dK' = dS^T @ Q'`), not the raw projections.
- `Model::max_seq_len` is the context window now. It used to be read off
  `positional_table.len()` at ~14 call sites; read it through `ctx_window()`.

`MODEL_VERSION` is **6** (`ROPE_VERSION` 4, `FINAL_LN_VERSION` 5 and
`TIED_UNEMBED_VERSION` 6 name the three boundaries that change the layout). A v3
file still parses — `read_model` reads
the positional table body and discards it, since the weights after it are at a
fixed offset — but logs a warning: such a model was trained under absolute
positions this code no longer applies *and* with a K projection that never left
random init, so its output is meaningless. Loading is allowed only so model
browsing and metadata keep working on old runs.


## The output projection is tied to the embedding table

From `MODEL_VERSION` 6, `Model::w_output` is **empty** and the embedding table is
both ends of the network: the embedding gathers rows from it, and
`project_logits` multiplies by it transposed (`cuda_matmul_bt` rather than
`cuda_matmul`). Test with `Model::ties_embeddings()`, never by inspecting values
— the same rule as `has_final_ln`, and for the same reason: a pre-v6 model
trained *against* a separate projection, so synthesizing a tie for it would
change what it outputs. `read_model` rejects a file whose version and tie
disagree rather than loading a model that would project through the wrong matrix.

Why: it buys exactly one doubling of the vocabulary for free. See "Vocabulary
layout" for the compression that doubling is worth.

**Tying changes the parameter count, not the gradient the vocabulary tables
receive, and that is deliberate.** A tied model still keeps **two gradient
accumulators** — `g_d_embedding` for the lookup path and `g_dw_output` (holding
`[vocab, d]`, not `[d, vocab]`) for the projection path — so each still gets its
own per-step unit-norm clip, exactly as an untied model's two matrices do. They
are summed by `merge_tied_gradients` at the optimizer step, after the per-step
clip and before the global clip and the Adam step. Summing them earlier, in the
backward, would put both paths under a single clip and quietly change the
effective learning rate on the largest tensor in the model — a second change
riding along with this one, which would confound any vocabulary comparison.

What *is* dropped is the projection's f16 view, f32 master and two Adam moments
(14 of the 18 bytes/param), and one Adam step per optimizer step.

**The guard is `a_tied_model_computes_what_the_untied_one_does`.** It builds an
untied model whose `w_output` is literally the tied model's embedding
transposed, runs both, and requires the same loss *and* that the tied
accumulators equal the untied pair element-for-element. That is what catches a
transposed operand, a `_bt` where a plain matmul belongs, or an accumulation
landing in the wrong buffer — none of which a "gradient is non-zero" assert can
see, and all of which would still descend a loss curve. This is the third
parameter between the residual stream and the logits, and the previous two (the
absolute positional table, the K projection) both shipped wired into the forward
while receiving no gradient.

`both_gradient_paths_reach_the_tied_matrix` is the complementary check, and it
turns on an asymmetry worth knowing: the **lookup gradient is sparse** (only rows
of tokens the chunk contains) while the **projection gradient is dense** (the
softmax pushes down every non-target logit, so every row is written). A token
absent from the chunk therefore isolates the projection path.

`warm_start_output_columns` returns 0 for a tied model. That is correct, not a
silent failure: `warm_start_embeddings` already reseeded both ends by rewriting
one row. Keep calling both.

**`tests/attention_learns.rs` is the guard.** With the absolute table gone, the
copy task is a pure RoPE test — nothing else can supply position, so a broken or
no-op rotation cannot beat `ln(vocab)`. It reaches the `ln(vocab)/2` floor, and
escapes the plateau ~500-1000 steps earlier than the absolute table did.


## The final layer norm

`project_logits` reads a **normalized** hidden state, not the raw residual
stream. The blocks are Pre-LN (`x += attn(ln(x))`, `x += ff(ln(x))`), so each one
normalizes its own *input* and adds into the residual — nothing normalizes the
residual itself, and its scale grows with depth. Without a final norm the logits
grow with it, the softmax sharpens, and every wrong prediction costs
disproportionately more nats.

It was measurable on the two pretrains that predate it: over their epoch
checkpoints, the mean `ln_gamma` of the last block against layer 0 stayed flat at
1.09 for the 6-layer run and ran away to **1.53** (2.05 absolute) for the 8-layer
one, starting at the stage the corpus switched to web text. That is the last
block fighting a residual it cannot rescale, and it is what turned the ~20-nat
newly-introduced-marker spike into the ~200 seen on that run.

- **No new kernels.** `cuda_layer_norm` / `cuda_layer_norm_backward` are
  dimension-generic; the final norm reuses them with its own `[d]` gamma/beta,
  trained like any other 1-D parameter (undecayed, as AdamW convention has it).
- **Its activations are dedicated buffers** (`final_ln_input`, `final_ln_output`,
  `final_ln_dgrad`), not borrowed from `b_sd`. The pre-norm residual has to
  survive the loss stage so the backward can differentiate against it, and the
  block loop reuses that scratch. Do not fold them back in.
- **`Model::has_final_ln` is the switch**, and `ln_final_gamma` being empty is
  its whole representation. A pre-v5 model trained with no final norm, and a
  LayerNorm at gamma=1/beta=0 is *not* the identity — synthesizing one would
  silently change what every existing model outputs. So an old base keeps the
  unnormalized projection it learned against, through loading, training, and
  saving: `write_model` writes the length, so "absent" round-trips through the v5
  layout rather than being re-inferred.
- Cost is ~3 extra launches on a step that already runs dozens: measured
  +0.9 ms/step on the d256/L6/ctx512 shape (13.2 → 14.0 mean of 3), though the
  run-to-run spread is ±0.7 ms, so read it as a few percent rather than a firm
  number.

**The guards are in `optimizer.rs`**: `the_final_layer_norm_receives_a_gradient`
and `a_model_without_a_final_layer_norm_trains_without_one`. The first exists
because this is the *third* parameter added between the residual stream and the
logits, and the previous two (the absolute positional table, the K projection)
both shipped wired into the forward pass while receiving no gradient — in both
cases the loss curve still went down, which is exactly why a "loss decreases"
test does not catch it.


## nvcc failures must fail the build

`build.rs` checks the exit status of **both** `nvcc` and `ar`. It did not, and
`.status()` reports only whether a process *ran*: a CUDA compile error left the
previous `shaders.o` in place, `ar` re-archived it, the crate linked, and every
test afterwards ran against a stale kernel. That is indistinguishable from an
edit having no effect, and it silently invalidates any measurement taken after
it. If you are editing `shaders.cu` and a change appears to do nothing, check
that the object's mtime is newer than the source before believing it.


## Mixed-precision (f32 master weights)

GPU training keeps **all compute in f16** but holds an **f32 master copy of every
*trained* weight** as the source of truth (ISSUE-0052). Each trained parameter has
three parts in `GpuWorkspace`: the f32 master (`m_*`), the f16 view the matmuls
read (`d_*`), and the f32 Adam moments (`v_*` = 1st, `s_*` = 2nd). The invariant:

- **The master is canonical; the f16 `d_*` weight is a derived, disposable view.**
  `adam_kernel` accumulates the step into the f32 master and refreshes the f16 view
  from it in the same pass (fused — no extra kernel launch). `upload_weights` seeds
  the master from the full-precision f32 CPU weights (not the f16 round-trip), and
  `sync_weights_from_gpu` reads the master back on save (so save is lossless).
- Only the weights `gpu_adam_update` steps get a master: 10×layers plus the
  shared output projection and token embedding, plus the final layer norm's
  gamma/beta when the model carries one (see "The final layer norm").
  `d_positional` is never updated, so it has **no** master and stays f16-only.

Why: before this, the master lived in f16, so an Adam step below the f16 ULP near
the weight (~1e-3 at w≈1.0) rounded away entirely — updates vanished at small LR
and diverged in coarse quanta at large LR, capping the usable LR at ~1e-5. The f32
master lets a sub-ULP `Δw` accumulate, lifting the cap (pretrain default is now
1e-4; see `use-training.ts`). Cost: **14 B/param** (f16 `w` 2 B + f32 `m`/`v` 8 B +
f32 master 4 B), ~+40 MB on a 10M-param model; compute throughput is unchanged.


## Inference: one device-resident session

`training::infer_session::InferSession` is the forward-only counterpart of
`Workspace`, and it is now the **only** inference path. `Model::forward_seq` and
its per-stage helpers (`layer_norm`, `self_attention`, `feed_forward`, `unembed`)
are gone; so is `cuda_attention_head`, the per-head kernel only they called.

The old path held nothing on the device between calls. Every layer norm, every
attention head, every feed-forward and the whole `[d, vocab]` output projection
were flattened on the host, converted f32→f16 element by element, `cudaMalloc`'d,
uploaded, launched, and copied straight back into `Vec<Vec<f32>>` — **once per
generated token**, ~1000 launches and ~250 malloc/free pairs each time. Measured
at d256/L6/h4/ctx512 with `LLM_PROFILE_INFER=1`: 66% of a token was host
marshalling and upload, 17% device→host copies, 4% host residual loops, and
**14% was kernels**. That is what low GPU utilization during generation meant.

Every one of the four pathologies commit `6048702` fixed for training was still
present here: per-head host loops, `cudaMalloc` per call, blocking D2H, and
non-persistent weights. Weights are now uploaded once, activations never leave
the device, and the only D2H per call is the logits.

**Measured, at d256/L6/h4/ctx512 on a 3060** (`tests/bench_generate.rs`, A/B run
with both paths present before the old one was deleted):

| prompt | old tok/s | new tok/s | speedup |
|--------|-----------|-----------|---------|
| 32     | 26.0 | 443.9 | 17.1x |
| 128    | 22.5 | 382.8 | 17.0x |
| 512    | 15.4 | 439.0 | 28.5x |

- **`ensure_attn` was reallocating every token.** It grows to the exact size
  requested, and a generating sequence gets one longer each step, so the guard
  missed every time and all 11 buffers — including the six backward-only ones
  inference never reads — were freed and re-malloc'd. `cuda_infer_reserve` sizes
  it once at `ctx` in `InferSession::new`. Worth ~9 ms/token at a short prompt.
- **The old path silently disagreed with the training forward.** It accumulated
  the residual stream in host f32 where training uses `cuda_add_inplace_f16`, and
  it skipped both the post-embedding and post-residual `cuda_clamp_f16` calls.
  A model was served through a different function than it trained under.
  `tests/infer_agrees_with_training_forward.rs` now pins them together and is the
  guard on any future change to either.

### The KV cache

`cuda_attention_decode` handles one new token against the cache: four
strided-batched GEMMs plus `rope_decode_kernel` and `decode_softmax_kernel`.

- **K is cached in f32, V in f16**, because that is what the prefill's GEMMs
  natively produce (`attn.k_f32`, and V straight out of its f16 GEMM). Caching K
  as f16 would make a decode step's score GEMM lower precision than a prefill's,
  and the two have to agree.
- **A prefill fills the cache as a side effect.**
  `cuda_attention_forward_batched` gained a nullable `k_cache_f32` and a
  `kv_stride`; training passes null/0, which leaves every stride at its old
  `seq`-based value. There is no separate "warm the cache" pass.
- **Decode needs no head concat.** `[h, 1, hd]` contiguous *is* the `[1, d]` row
  the wo GEMM wants. The prefill needs `cuda_concat_heads` only because its
  `[h, s, hd]` has to be interleaved by token.
- **`cached_tokens` holds the *windowed* sequence**, and that is what makes a
  moving context window safe without tracking an offset: K and V at position `i`
  are a function of the windowed tokens `0..=i` and of `i` (through RoPE) and
  nothing else, so two windowed sequences sharing a prefix share valid cached
  rows however the window moved. A slid window fails the prefix comparison and
  re-prefills, which is correct and is `a_sliding_context_window_stays_correct`.

**A trap when writing tests against this.** Two `logits_*` calls on the *same*
session no longer run the same code: the second finds the sequence cached and
takes the decode branch. Comparing them therefore compares prefill against
decode, which is a real property but not usually the one being tested, and it is
not bit-exact (see below). Use a fresh `InferSession` per side when the property
under test is anything else — truncation, a row offset, weight packing. Two tests
had to be fixed this way after the cache landed.

**What the cache is worth depends entirely on context length,** and at this
model's size it is not the headline. `prefill_ms` in the bench is a free A/B —
before the cache every token cost a full prefill of the sequence so far:

| prompt | prefill (= old per-token) | decode ms/token | speedup |
|--------|---------------------------|-----------------|---------|
| 128  | 2.0 ms  | 1.8 ms | 1.1x |
| 512  | 2.3-3.2 | 1.7-1.9 | 1.2-1.9x |
| 2048 | 9.8-10.1 | 1.9-2.1 | ~5x |
| 4000 | 34.3-36.6 | 1.8-2.0 | **~19x** |

Decode is flat in context length; prefill is not, and must not be — it contains
the O(s²) attention.

**A decode step at these shapes is launch-bound, not arithmetic-bound**: ~100
kernel launches against ~0.1 ms of FLOPs. `layers` is the only shape dimension
that reliably moves the number, because launches scale with it; rows like
`d 256→128` measuring *slower* are the noise floor, not physics. Do not read the
per-stage profile as FLOP attribution — with `LLM_PROFILE_INFER=1` each stage is
bracketed by a `cudaDeviceSynchronize`, and at ~2 ms/token those syncs cost more
than the stage does. **Fusing the residual+clamp pairs, or CUDA graphs for the
decode step, is where the next multiple is** — not more algorithmic work.

### Two host costs that were larger than they looked

- **`bpe::decode` rebuilt its merge table on every call** — ~7,900 hash inserts
  at vocab 8,192, measured at 292 µs, i.e. 15% of a token, for a table that never
  changes. The streaming loop called it once per generated token to re-decode the
  reply so far. `bpe::Decoder` indexes once, exactly as `bpe::Encoder` already
  did for the other direction; the free `decode` is now a wrapper.
- **The sampler sorted the whole vocabulary** to keep 40 candidates: 106 µs, ~5%
  of a token. `sample::sample` partitions with `select_nth_unstable_by` instead
  (5.9x faster) and its comparator breaks ties on ascending token id so the order
  is *total* — without that, an unstable partition could reorder tied logits and a
  seeded benchmark would stop reproducing its own past runs.

There were also **two samplers that had drifted apart on correctness, not
determinism**. `infer::sample_token` exponentiated raw logits with no
max-subtraction, so a confident model overflowed `exp` to `inf` and the whole
distribution went `NaN`; and `temperature = 0.0` divided by zero, returning the
argmax only by accident (`r -= inf` gives `NaN`, `NaN <= 0.0` is false, so the
loop fell through to its initial value). Its unit test passed on that accident.
There is now one sampler, taking the RNG as a parameter — which is the only thing
serving and benchmarking actually needed to differ about.

### Scores may have moved, and `SCORE_VERSION` was deliberately not bumped

`eval::score` and `kto::reference_logprob` run on the same forward, so this work
can shift a benchmark number with no model having changed. That was a considered
decision, not an oversight.

Measured on real trained weights (Shakedown 1, d256/L6) against the *training*
forward, which is the function a model's own objective implies:

- worst position: **0.098% of the logit range**, total variation between
  next-token distributions **0.003**
- **1 argmax flip in 64 positions**, and at that position the top-2 gap was
  `0.007812` against a noise floor of `0.007812` — one f16 ULP, an exact tie
  where either answer is equally defensible

So metrics built on strict argmax or `>` comparisons (minimal pairs, top-1 at
markers) can move slightly, but only where the model was already indifferent.
For scale, the markup suite reports `± 1.3` points of sampling error on its own.
No pre-change score was available to diff against — `data/artifacts/models` holds
weights but the scores live in Postgres — so **the shift was bounded by
construction rather than measured end to end.** If a suspicious score movement
shows up on a model with history, this is the first thing to suspect.

Post-change reference, for future comparison: Shakedown 1 scores **40.9 / 100**
on the language suite (bits/byte 1.796) and **32.1 ± 1.3 / 100** on markup,
reproducible across runs.

### Lifecycle

`infer::MODEL_CACHE` holds the session, keyed on the model directory, and drops
the previous one before building the next so two models' weights are never both
resident. `infer::release()` frees it, and `jobs::queue` calls that before every
non-chat job: one GPU, one queue, and the `attn`/`ff` scratch the session sizes at
`ctx` is the same file-static arena a training `Workspace` resizes. Chat jobs are
exempt, since releasing right before generating would throw away the cache that
makes consecutive turns cheap.

`InferSession` is `Send` but deliberately **not** `Sync`. A CUDA device pointer
belongs to the process's primary context, so moving a session between threads is
sound; two threads *using* one concurrently is not, and neither is two sessions
running at once, for the file-static-scratch reason above.
