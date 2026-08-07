// gork: transformer training CUDA kernels (f16 mixed precision + cuBLAS tensor cores)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <mma.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// A cuBLAS call that fails does nothing and says nothing: the destination buffer
// keeps whatever it held, the host return code is the only signal, and training
// carries on producing a plausible-looking loss curve. That is exactly how the
// attention backward shipped for the life of this project with dK stuck at zero
// — an f32×f16 GemmEx, which cuBLAS rejects as an unsupported type combination.
//
// The status is a host-side return value, so checking it costs nothing and
// forces no synchronization; the launch stays asynchronous. Aborting is
// deliberate — a wrong gradient is far more expensive than a crash, and it is
// only discoverable hours later in a ruined model.
#define CUBLAS_CHECK(call)                                                     \
  do {                                                                         \
    cublasStatus_t status_ = (call);                                           \
    if (status_ != CUBLAS_STATUS_SUCCESS) {                                    \
      fprintf(stderr, "cuBLAS call failed with status %d at %s:%d\n",          \
              (int)status_, __FILE__, __LINE__);                               \
      abort();                                                                 \
    }                                                                          \
  } while (0)

// ─── cuBLAS handle (lazily initialized) ─────────────────────────────────────
static cublasHandle_t g_cublas = NULL;
static bool g_cublas_init = false;

extern "C" void cublas_init() {
    if (!g_cublas_init) {
        cublasCreate(&g_cublas);
        g_cublas_init = true;
    }
}

// ─── Flash attention tiling constants ───────────────────────────────────────
// Declared up here because `ensure_attn` has to round its [h, seq, hd] buffers
// up to FLASH_BR rows. See the FlashAttention section for what they mean.
//
// 128 threads = 4 warps per block, Br = 64 so each warp owns one 16-row wmma
// M-slab. Bc is 64 except at hd 128, where it drops to 32 to keep the block
// under the 48 KB static shared-memory limit (past that needs an opt-in dynamic
// carveout, and one block per SM).
//
// Every shared tile's leading dimension is padded by FLASH_PAD halves. An
// unpadded 64-half row is exactly 128 B, the full 32-bank width, which conflicts
// every ldmatrix inside load_matrix_sync. 8 halves is also the minimum pad that
// keeps `ldm` a multiple of 16 bytes, which wmma requires — 4 would not.
#define FLASH_PAD 8
#define FLASH_WARPS 4
#define FLASH_THREADS (FLASH_WARPS * 32)
#define FLASH_BR 64
#define FLASH_BC(HD) ((HD) >= 128 ? 32 : 64)

// The per-block row extent every flash kernel reads in 16-row wmma slabs. A warp
// whose whole slab sits past `seq` still executes its mma_sync (a lane may not
// skip one), so every [h, seq, hd] buffer a flash kernel touches is allocated at
// a multiple of this many rows. `Workspace::ensure_layer_caches` rounds by the
// same value, read through here so the two cannot drift.
extern "C" int cuda_flash_row_granularity(void) { return FLASH_BR; }

static int flash_round_rows(int seq) {
    return ((seq + FLASH_BR - 1) / FLASH_BR) * FLASH_BR;
}

// Defined with the flash kernels; `ensure_attn` needs it to decide whether the
// s²-sized legacy buffers are worth allocating at all.
extern "C" int cuda_attn_flash_active(int hd);

// ─── Pre-allocated attention scratch buffers ───────────────────────────────
// Lazily allocated, grown to max(seq, hd, d) seen. Shared across all
// attention functions (forward/backward are sequential, not concurrent).
static struct {
    // Forward: Q/K in f32, scores f32+f16
    float *q_f32, *k_f32;
    // The [h, seq, seq] score plane. Legacy attention only — see `ensure_attn`.
    float *scores_f32;
    __half *scores_f16;
    // A decode step scores one query row against the cache: [h, 1, seq], not
    // [h, seq, seq]. It used to borrow `scores_f32`/`scores_f16` and touch a
    // `seq`-th of them, which was merely wasteful then and is wrong now that
    // those two are not allocated on the flash path.
    float *dec_f32;
    __half *dec_f16;
    // f16 Q/K for the flash forward when the caller keeps no per-layer cache —
    // i.e. an inference prefill, which passes Q_all/K_all null because it has no
    // backward to feed. Training reads its own per-layer caches instead.
    __half *q_f16, *k_f16;
    // Backward: dQ/dK/dV, dP f32, dS f16. Q/K/V are NOT here — they
    // are per-layer `Workspace` caches; see cuda_attention_backward_batched.
    __half *dQ, *dK, *dV;
    float *dP;
    __half *dS;
    // The per-head dout, [h, seq, hd]. It used to share `dQ`, which the legacy
    // backward overwrites only after its last read of dout; the flash backward
    // reads dout and writes dQ in the same kernel, so the two had to separate.
    __half *dO;
    // Flash state: the forward's per-row logsumexp and the backward's
    // rowsum(dO ∘ O), both [h, seq] f32. Between them they replace the entire
    // [h, seq, seq] softmax plane — 8 KB against 16.8 MB at d512/h8/ctx1024.
    float *lse, *dsum;
    int seq, hd, d, h;
    // Whether the s²-sized legacy buffers below are currently allocated.
    bool legacy;
} attn = {NULL};

static void ensure_attn_cleanup(void) {
    auto f = [](auto &p) { if (p) { cudaFree(p); p = NULL; } };
    f(attn.q_f32); f(attn.k_f32);
    f(attn.dec_f32); f(attn.dec_f16);
    f(attn.q_f16); f(attn.k_f16);
    f(attn.scores_f32); f(attn.scores_f16);
    f(attn.dQ); f(attn.dK); f(attn.dV); f(attn.dO);
    f(attn.lse); f(attn.dsum);
    f(attn.dP); f(attn.dS);
    attn.seq = attn.hd = attn.d = attn.h = 0;
}

// ─── Pre-allocated feed-forward scratch ────────────────────────────────────
// Same lazily-grown, grow-only pattern as `attn` above, and for the same
// reason: these buffers used to be cudaMalloc'd and cudaFree'd on every call.
// At 6 layers that was 18 allocation pairs per training step, and cudaMalloc /
// cudaFree implicitly synchronize the device — each one drained the pipeline
// and left the GPU idle while the CPU caught up.
//
// Reuse across calls is safe because both buffers are fully written before they
// are read on every path: `hidden` is a cuBLAS output with beta=0 (a complete
// overwrite) and `activated` is fully cudaMemcpy'd from pre_gelu. No stale data
// can survive into a later call. They are two separate buffers, not one reused
// twice, because the backward holds both live at the same time.
//
// This assumes a single stream with forward and backward strictly sequential —
// the same assumption `attn` already makes. Concurrent streams would need
// per-stream scratch.
static struct {
    __half *hidden;     // [seq * d_ff] — w1 output / dgelu accumulator
    __half *activated;  // [seq * d_ff] — gelu(pre_gelu), backward only
    int seq, d_ff;
} ff = {NULL};

// One f32 accumulator reused across every per-buffer gradient clip. Lazily
// allocated in cuda_clip_norm, freed with the rest of the persistent scratch.
static float *clip_scratch = NULL;

static void ensure_ff_cleanup(void) {
    auto f = [](auto &p) { if (p) { cudaFree(p); p = NULL; } };
    f(ff.hidden); f(ff.activated);
    ff.seq = ff.d_ff = 0;
}

// Grows to the max (seq, d_ff) ever seen and never shrinks, so a later, longer
// sequence reallocates rather than reading past the end of a smaller buffer.
static void ensure_ff(int seq, int d_ff) {
    if (seq <= ff.seq && d_ff <= ff.d_ff && ff.hidden) return;
    ensure_ff_cleanup();
    size_t sz = (size_t)seq * d_ff * sizeof(__half);
    cudaMalloc(&ff.hidden, sz);
    cudaMalloc(&ff.activated, sz);
    ff.seq = seq; ff.d_ff = d_ff;
}

extern "C" void cublas_shutdown() {
    ensure_attn_cleanup();
    ensure_ff_cleanup();
    if (clip_scratch) { cudaFree(clip_scratch); clip_scratch = NULL; }

    if (g_cublas_init) {
        cublasDestroy(g_cublas);
        g_cublas_init = false;
        g_cublas = NULL;
    }
}

static cublasHandle_t get_cublas() {
    if (!g_cublas_init) {
        cublasCreate(&g_cublas);
        g_cublas_init = true;
    }
    return g_cublas;
}

// ─── Gradient-clipping helpers (f32 vectors) ────────────────────────────────
// L2 norm of a gradient buffer (host-mode result, so this is synchronous).
extern "C" float cuda_l2norm(const float *v, int n) {
    float result = 0.0f;
    CUBLAS_CHECK(cublasSnrm2(get_cublas(), n, v, 1, &result));
    return result;
}

// Scale a gradient buffer in place by `s`.
extern "C" void cuda_scale(float *v, float s, int n) {
    CUBLAS_CHECK(cublasSscal(get_cublas(), n, &s, v, 1));
}

#define TENSOR_OP CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define COMPUTE_F32 CUBLAS_COMPUTE_32F


static void ensure_attn(int seq, int hd, int d, int h) {
    // The four s²-sized buffers exist only for the legacy cuBLAS attention. They
    // are the whole reason this arena is the second-largest allocation in the
    // trainer (~100 MB at d512/h8/ctx1024), so the flash path must not merely
    // stop reading them — it must stop allocating them.
    //
    // The condition is the *shape gate*, not the switch: a head_dim the flash
    // kernels are not instantiated for falls back to the cuBLAS path and still
    // needs these, however the switch is set. Reading the switch alone left a
    // d16/h2 model running the legacy score GEMM against a null pointer. Both
    // inputs can also change between calls — the A/B tests flip the switch, and
    // consecutive models differ in shape — so both belong in the guard below.
    const bool want_legacy = !cuda_attn_flash_active(hd);
    if (seq <= attn.seq && hd <= attn.hd && d <= attn.d && h <= attn.h &&
        attn.q_f32 && want_legacy == attn.legacy)
        return;
    ensure_attn_cleanup();

    // Every [h, seq, hd] buffer a flash kernel reads or writes is allocated at a
    // whole number of FLASH_BR-row blocks: a warp whose 16-row slab sits past
    // `seq` still runs its mma_sync, because a lane may not skip one, and that
    // load touches the rounded-up extent. The rounding costs ~64 KB.
    const int rows = flash_round_rows(seq);

    // Every buffer is zeroed, and for the flash path that is load-bearing rather
    // than tidiness. The dK/dV backward contracts whole 16-row slabs of Q and dO
    // against P/dS, and rows past `seq` are held to zero on the P side only —
    // arithmetically, not structurally. `0 * inf` is NaN, and the ±65504 clamp on
    // the store then turns that NaN into a large *finite* number (fmaxf returns
    // its non-NaN operand), so the gradient comes out plausible and wrong rather
    // than obviously broken. It cost an afternoon here.
    //
    // The invariant this establishes is "the tail is finite", not "the tail is
    // zero": a later shorter chunk leaves a longer one's real values in the tail,
    // which multiply out to zero perfectly well. Only never-written memory is
    // dangerous, and that is exactly what this removes.
    auto a = [](auto &p, size_t sz) { cudaMalloc(&p, sz); cudaMemset(p, 0, sz); };
    a(attn.q_f32,      h * seq  * hd * sizeof(float));
    a(attn.k_f32,      h * seq  * hd * sizeof(float));
    a(attn.dec_f32,    h * rows      * sizeof(float));
    a(attn.dec_f16,    h * rows      * sizeof(__half));
    a(attn.q_f16,      h * rows * hd * sizeof(__half));
    a(attn.k_f16,      h * rows * hd * sizeof(__half));
    a(attn.dQ,         h * rows * hd * sizeof(__half));
    a(attn.dK,         h * rows * hd * sizeof(__half));
    a(attn.dV,         h * rows * hd * sizeof(__half));
    a(attn.dO,         h * rows * hd * sizeof(__half));
    a(attn.lse,        h * rows      * sizeof(float));
    a(attn.dsum,       h * rows      * sizeof(float));
    if (want_legacy) {
        a(attn.scores_f32, h * seq * seq * sizeof(float));
        a(attn.scores_f16, h * seq * seq * sizeof(__half));
        a(attn.dP,         h * seq * seq * sizeof(float));
        a(attn.dS,         h * seq * seq * sizeof(__half));
    }
    attn.seq = seq; attn.hd = hd; attn.d = d; attn.h = h;
    attn.legacy = want_legacy;
}

// Pre-size the shared attention scratch for the longest sequence a caller will
// ever hand it, so it is allocated once instead of growing a token at a time.
//
// `ensure_attn` only grows, and it grows to the exact size requested. A
// generating sequence gets one token longer every step, so `seq <= attn.seq`
// failed on every single step and the whole arena — including the six
// backward-only buffers inference never reads — was freed and re-malloc'd per
// token. Measured at ~9 ms/token of allocator churn at a short prompt.
//
// Inference calls this once when it builds its session, with the context window.
// `d_ff` is passed rather than assumed to be 4*d: the feed-forward width is
// configurable, and `cuda_feed_forward` reads it off the weight shape for exactly
// that reason.
extern "C" void cuda_infer_reserve(int seq, int hd, int d, int h, int d_ff) {
    ensure_attn(seq, hd, d, h);
    ensure_ff(seq, d_ff);
}

// ─── Fused scale + causal_mask + softmax (f32→f16, single kernel) ──────────
// Reads f32 scores, writes f16 softmax output. Masks positions col > row to 0.
// One BLOCK per row of the attention matrix, threads striding the columns.
//
// This was one *thread* per row: a 512x512 score matrix was processed by 512
// threads, each walking a whole row. Two problems, both severe here because the
// work is s² while the thread count was only s. Parallelism: 512 threads on a
// device that wants tens of thousands. Memory: adjacent threads read addresses
// `seq_len * 4` bytes apart, so every load was its own 32-byte transaction for
// 4 useful bytes — an 8x read amplification on the single largest tensor in the
// step. This is the dominant cost of a training step, and it gets worse
// quadratically as the context window grows.
//
// The arithmetic is unchanged, including keeping `inv_scale` inside the
// exponent: exp((x - max(x)) * c) == exp(x*c - max(x*c)) for c > 0, so this is
// the same softmax the old kernel computed, bit for bit in the same order of
// operations per element. `sdata` needs `blockDim.x` floats.
__global__ void fused_scale_causal_softmax_kernel(
    const float *__restrict__ scores_f32, __half *__restrict__ scores_f16,
    float inv_scale, int seq_len) {
  int row = blockIdx.x;
  if (row >= seq_len) return;
  extern __shared__ float sdata[];
  int tid = threadIdx.x;

  // blockIdx.y selects the head, so all `h` heads go out in one launch instead
  // of a host-side loop. A 1-D launch leaves gridDim.y == 1 and offset 0, which
  // is exactly the single-head behaviour the inference path wants.
  size_t plane = (size_t)blockIdx.y * seq_len * seq_len;
  const float *row_in = scores_f32 + plane + (size_t)row * seq_len;
  __half *row_out = scores_f16 + plane + (size_t)row * seq_len;

  // Max over the causal prefix [0, row]. Threads with no element in range keep
  // the sentinel, which loses the max reduction.
  float m = -1e38f;
  for (int j = tid; j <= row; j += blockDim.x)
    if (row_in[j] > m) m = row_in[j];
  sdata[tid] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s && sdata[tid + s] > sdata[tid]) sdata[tid] = sdata[tid + s];
    __syncthreads();
  }
  float max_val = sdata[0];
  __syncthreads();

  // exp into the output over the prefix, zeros over the masked tail.
  float sum = 0.0f;
  for (int j = tid; j < seq_len; j += blockDim.x) {
    if (j <= row) {
      float e = expf((row_in[j] - max_val) * inv_scale);
      row_out[j] = __float2half(e);
      sum += e;
    } else {
      row_out[j] = __float2half(0.0f);
    }
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float inv_sum = 1.0f / sdata[0];

  // Normalize in place, reading back through f16 exactly as before.
  for (int j = tid; j <= row; j += blockDim.x)
    row_out[j] = __float2half(__half2float(row_out[j]) * inv_sum);
}

// ─── Fused bias_add + GELU (f16 in-place) ──────────────────────────────────
__global__ void fused_bias_gelu_kernel(__half *x, const __half *__restrict__ bias,
                                        int cols, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(x[i]) + __half2float(bias[i % cols]);
  float c = 0.7978845608f * (v + 0.044715f * v * v * v);
  x[i] = __float2half(0.5f * v * (1.0f + tanhf(c)));
}

// ─── Scale f32 buffer in-place: x[i] *= scale ──────────────────────────────
__global__ void scale_f32_kernel(float *x, float scale, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] *= scale;
}

extern "C" void cuda_scale_f32(float *x, float scale, int n) {
    scale_f32_kernel<<<(n + 255) / 256, 256>>>(x, scale, n);
}

// ─── SGD with momentum: v = mu*v + g; w = f16(f32(w) - lr * v) ────────────
__global__ void sgd_momentum_kernel(const float *__restrict__ g, __half *w,
                                     float *v, float lr, float mu, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float new_v = mu * v[i] + g[i];
  v[i] = new_v;
  w[i] = __float2half(__half2float(w[i]) - lr * new_v);
}

// AdamW update. `w_master` (f32) is the canonical weight; `w_view` (f16) is a
// derived copy the matmuls consume, refreshed from the master in the same pass.
// The two moment estimates (m = 1st, v = 2nd) are f32. `grad_scale` averages the
// accumulated gradient (1/accum_steps); `bc1`, `bc2` are the bias-correction
// denominators (1 - beta^t) computed host-side. Accumulating the step in f32
// (not the old f16 read-back) is the mixed-precision fix that lifts the LR cap
// (ISSUE-0052): a sub-f16-ULP `Δw` now survives instead of rounding away.
//
// `weight_decay` is DECOUPLED (AdamW, not L2-in-gradient): it shrinks the master
// weight toward zero directly, outside the moment estimates, so it doesn't feed
// m/v or interact with the per-parameter adaptive scaling. Pass 0.0 to recover
// plain Adam (the caller does exactly that for biases and LayerNorm params).
__global__ void adam_kernel(const float *__restrict__ g, float *w_master,
                            __half *w_view, float *m, float *v, float grad_scale,
                            float lr, float beta1, float beta2, float eps,
                            float weight_decay, float bc1, float bc2, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float grad = g[i] * grad_scale;
  float new_m = beta1 * m[i] + (1.0f - beta1) * grad;
  float new_v = beta2 * v[i] + (1.0f - beta2) * grad * grad;
  m[i] = new_m;
  v[i] = new_v;
  float m_hat = new_m / bc1;
  float v_hat = new_v / bc2;
  // Decoupled decay: the `weight_decay * w_master[i]` term is added to the Adam
  // step, not to the gradient — reduces to plain Adam when weight_decay == 0.
  float updated =
      w_master[i] - lr * (m_hat / (sqrtf(v_hat) + eps) + weight_decay * w_master[i]);
  w_master[i] = updated;                 // accumulate in f32 (the master)
  w_view[i] = __float2half(updated);     // refresh the f16 view, fused
}

// ─── Copy strided columns into contiguous batch layout ─────────────────────
// src: [rows × src_cols] row-major, copies 'num' strided column slices
// each of width 'dst_cols' starting at stride offset 'src_stride'
// into dst: [num × rows × dst_cols] row-major contiguous
__global__ void extract_strided_cols_kernel(const __half *__restrict__ src,
    __half *__restrict__ dst, int rows, int src_cols, int dst_cols,
    int num, int src_stride) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  for (int hi = 0; hi < num; hi++) {
    const __half *s = src + row * src_cols + hi * src_stride;
    __half *d = dst + (hi * rows + row) * dst_cols;
    #pragma unroll
    for (int c = 0; c < dst_cols; c++)
      d[c] = s[c];
  }
}

extern "C" void cuda_extract_strided_cols(const __half *src, __half *dst,
    int rows, int src_cols, int dst_cols, int num, int src_stride) {
  extract_strided_cols_kernel<<<(rows + 255) / 256, 256>>>(src, dst, rows, src_cols, dst_cols, num, src_stride);
}

// ─── f32 → f16 element-wise conversion ──────────────────────────────────────
__global__ void f32_to_f16_kernel(const float *src, __half *dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] = __float2half(src[i]);
}

// ─── cross-entropy backward ────────────────────────────────────────────────

// `mask` (length seq_len, 1=train / 0=ignore) may be null, meaning "every row is
// active". Masked rows contribute a zero gradient. The gradient is normalized by
// `active` — the number of trained rows — so it matches the loss normalizer
// (SFT trains on a subset of positions; pretraining passes mask=null, active=seq_len).
//
// `scale` multiplies the whole gradient. Pretraining and SFT pass 1.0, which is
// exactly the old behaviour. Preference training (KTO) passes a per-example scalar
// it can only know *after* the forward pass — the derivative of its loss with
// respect to the sequence log-probability — because a KTO gradient is precisely
// this cross-entropy gradient times a scalar, and negative for a rejected reply
// (push the sequence down rather than up). That is the whole reason this parameter
// exists: no separate preference kernel is needed.
// One BLOCK per row (was one thread per row, each scanning all `vocab` logits
// three times with uncoalesced access — 512 threads for a 512x8192 tensor).
// The row max and the exp sum are shared-memory reductions; their order changes,
// so the low bits of dlogits can move. Shared memory: blockDim.x floats.
__global__ void cross_entropy_bw_kernel(const __half *logits,
                                         const float *targets,
                                         const float *mask,
                                         __half *dlogits, int seq_len,
                                         int vocab, int active, float scale) {
  int row = blockIdx.x;
  if (row >= seq_len)
    return;
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  __half *row_dlogits = dlogits + (size_t)row * vocab;

  // Ignored position: no learning signal — zero its whole gradient row.
  if (mask != nullptr && mask[row] == 0.0f) {
    for (int j = tid; j < vocab; j += blockDim.x)
      row_dlogits[j] = __float2half(0.0f);
    return;
  }

  const __half *row_logits = logits + (size_t)row * vocab;
  int target = (int)(targets[row]);

  float m = -INFINITY;
  for (int j = tid; j < vocab; j += blockDim.x)
    m = fmaxf(m, __half2float(row_logits[j]));
  sdata[tid] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
    __syncthreads();
  }
  float max_val = sdata[0];
  __syncthreads();

  float acc = 0.0f;
  for (int j = tid; j < vocab; j += blockDim.x)
    acc += expf(__half2float(row_logits[j]) - max_val);
  sdata[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float inv_sum = 1.0f / sdata[0];

  for (int j = tid; j < vocab; j += blockDim.x) {
    float softmax_j = expf(__half2float(row_logits[j]) - max_val) * inv_sum;
    row_dlogits[j] =
        __float2half(scale * (softmax_j - (j == target ? 1.0f : 0.0f)) / active);
  }
}

extern "C" void cuda_cross_entropy_backward(const __half *logits,
                                             const float *targets,
                                             const float *mask,
                                             __half *dlogits, int seq_len,
                                             int vocab, int active, float scale) {
  int threads = 256;
  cross_entropy_bw_kernel<<<seq_len, threads, threads * sizeof(float)>>>(
      logits, targets, mask, dlogits, seq_len, vocab, active, scale);
}

// ─── cross-entropy loss (GPU, async-friendly via event) ─────────────────

// `mask` (length seq_len, 1=train / 0=ignore) may be null, meaning "every row is
// active". Masked rows contribute 0 to the loss, which is normalized by `active`
// (the trained-row count) rather than seq_len so the reported loss is the mean
// over trained positions only.
// One BLOCK per row — see cross_entropy_bw_kernel for the rationale. Each block
// reduces its own row's max and exp-sum and atomically adds that row's
// contribution, where before one thread scanned a whole 8192-entry row twice.
// Shared memory: blockDim.x floats.
__global__ void cross_entropy_loss_kernel(const __half *logits,
                                           const float *targets,
                                           const float *mask,
                                           float *loss_out, int seq_len,
                                           int vocab, int active) {
  int row = blockIdx.x;
  if (row >= seq_len) return;
  if (mask != nullptr && mask[row] == 0.0f) return;

  extern __shared__ float shared[];
  int tid = threadIdx.x;
  const __half *row_logits = logits + (size_t)row * vocab;

  float m = -INFINITY;
  for (int j = tid; j < vocab; j += blockDim.x)
    m = fmaxf(m, __half2float(row_logits[j]));
  shared[tid] = m;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] = fmaxf(shared[tid], shared[tid + s]);
    __syncthreads();
  }
  float max_val = shared[0];
  __syncthreads();

  float acc = 0.0f;
  for (int j = tid; j < vocab; j += blockDim.x)
    acc += expf(__half2float(row_logits[j]) - max_val);
  shared[tid] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    int target = (int)targets[row];
    float log_sum_exp = max_val + logf(shared[0]);
    float contrib = log_sum_exp - __half2float(row_logits[target]);
    atomicAdd(loss_out, contrib / active);
  }
}

extern "C" float cuda_cross_entropy_loss(const __half *logits,
                                          const float *targets,
                                          const float *mask, int seq_len,
                                          int vocab, int active) {
  float *d_loss;
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemset(d_loss, 0, sizeof(float));
  int threads = 256;
  cross_entropy_loss_kernel<<<seq_len, threads, threads * sizeof(float)>>>(
      logits, targets, mask, d_loss, seq_len, vocab, active);
  float loss;
  cudaMemcpy(&loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_loss);
  return loss;
}

// ─── GELU backward ─────────────────────────────────────────────────────────

__global__ void gelu_bw_kernel(const __half *x, const __half *dy, __half *dx,
                                int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = __half2float(x[i]);
  float c = 0.7978845608f * (v + 0.044715f * v * v * v);
  float t = tanhf(c);
  float dgelu = 0.5f * (1.0f + t) +
                0.5f * v * (1.0f - t * t) * 0.7978845608f *
                    (1.0f + 3.0f * 0.044715f * v * v);
  dx[i] = __float2half(__half2float(dy[i]) * dgelu);
}

extern "C" void cuda_gelu_backward(const __half *x, const __half *dy, __half *dx,
                                    int n) {
  gelu_bw_kernel<<<(n + 255) / 256, 256>>>(x, dy, dx, n);
}

// ─── bias gradient ─────────────────────────────────────────────────────────

__global__ void bias_grad_kernel(const __half *dy, float *db, int rows,
                                  int cols) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;
  float sum = 0.0f;
  for (int i = 0; i < rows; i++)
    sum += __half2float(dy[i * cols + col]);
  db[col] = sum;
}

extern "C" void cuda_bias_grad(const __half *dy, float *db, int rows,
                                int cols) {
  bias_grad_kernel<<<(cols + 255) / 256, 256>>>(dy, db, rows, cols);
}

// ─── layer norm backward ───────────────────────────────────────────────────

// One BLOCK per row.
//
// This was one *thread* per column, and every one of those threads re-derived
// the current row's mean, variance, mean(dy) and mean(dy·x_centered) from
// scratch — so each row's four statistics were computed `cols` times over, by
// only `cols` threads (256 of the ~43,000 the device wants). The redundancy and
// the starvation multiply.
//
// It runs twice per layer, and measurement put it at the top of the profile by
// a wide margin: it is why the backward pass cost 15x its forward counterpart
// in attention and 45x in the feed-forward, against an expected ~2x.
//
// The arithmetic is preserved exactly, including the two-pass variance (rather
// than the less stable E[x²]-E[x]² form) and `1/sqrtf` rather than `rsqrtf`.
// What changes is that each row's statistics are reduced once in shared memory,
// and `dx` is written with adjacent threads on adjacent columns — coalesced
// instead of one 32-byte transaction per useful 2 bytes.
//
// `dgamma`/`dbeta` accumulate via atomicAdd across row-blocks, so the wrapper
// zeroes them first; the old kernel assigned, and the callers rely on that
// (they pass scratch buffers and add into the real accumulators afterwards).
// Summation order across rows is consequently nondeterministic, which can move
// the low bits of the gamma/beta gradients run to run.
//
// Shared memory: 2 * blockDim.x floats (two accumulators per reduction pass).
__global__ void layer_norm_bw_kernel(const __half *x, const __half *dy,
                                      const __half *gamma, __half *dx,
                                      float *dgamma, float *dbeta, int rows,
                                      int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  extern __shared__ float sm[];
  float *s1 = sm;
  float *s2 = sm + blockDim.x;
  int tid = threadIdx.x;

  const __half *row_x = x + (size_t)row * cols;
  const __half *row_dy = dy + (size_t)row * cols;
  __half *row_dx = dx + (size_t)row * cols;

  // Pass 1: sum(x) and sum(dy) -> mean, mean_dy.
  float a = 0.0f, b = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    a += __half2float(row_x[j]);
    b += __half2float(row_dy[j]);
  }
  s1[tid] = a;
  s2[tid] = b;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s1[tid] += s1[tid + s];
      s2[tid] += s2[tid + s];
    }
    __syncthreads();
  }
  float mean = s1[0] / cols;
  float mean_dy = s2[0] / cols;
  __syncthreads();

  // Pass 2: sum((x-mean)²) and sum(dy·(x-mean)) -> var, mean_dy_x.
  a = 0.0f;
  b = 0.0f;
  for (int j = tid; j < cols; j += blockDim.x) {
    float xc = __half2float(row_x[j]) - mean;
    a += xc * xc;
    b += __half2float(row_dy[j]) * xc;
  }
  s1[tid] = a;
  s2[tid] = b;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      s1[tid] += s1[tid + s];
      s2[tid] += s2[tid + s];
    }
    __syncthreads();
  }
  float var = s1[0] / cols;
  float mean_dy_x = s2[0] / cols;

  float inv_sigma = 1.0f / sqrtf(var + eps);

  for (int j = tid; j < cols; j += blockDim.x) {
    float xc = __half2float(row_x[j]) - mean;
    float dyj = __half2float(row_dy[j]);

    atomicAdd(&dgamma[j], dyj * xc * inv_sigma);
    atomicAdd(&dbeta[j], dyj);

    float dx_val = (__half2float(gamma[j]) * inv_sigma) *
                   (dyj - mean_dy - xc * inv_sigma * inv_sigma * mean_dy_x);
    dx_val = fminf(fmaxf(dx_val, -65504.0f), 65504.0f);
    row_dx[j] = __float2half(dx_val);
  }
}

extern "C" void cuda_layer_norm_backward(const __half *x, const __half *dy,
                                          const __half *gamma, __half *dx,
                                          float *dgamma, float *dbeta,
                                          int rows, int cols, float eps) {
  // dgamma/dbeta are accumulated with atomicAdd by the row-blocks, so they must
  // start at zero — the kernel's contract is still "assign", as the callers
  // expect (they hand in scratch and add into the real accumulators after).
  cudaMemsetAsync(dgamma, 0, cols * sizeof(float));
  cudaMemsetAsync(dbeta, 0, cols * sizeof(float));
  int threads = 256;
  layer_norm_bw_kernel<<<rows, threads, 2 * threads * sizeof(float)>>>(
      x, dy, gamma, dx, dgamma, dbeta, rows, cols, eps);
}

// ─── causal mask backward ──────────────────────────────────────────────────

// blockIdx.z selects the head, so the per-head host loop collapses to one
// launch. A 2-D launch leaves gridDim.z == 1 and offset 0 (single-head path).
__global__ void causal_mask_bw_kernel(__half *dS, int seq_len) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len || col >= seq_len)
    return;
  if (col > row)
    dS[(size_t)blockIdx.z * seq_len * seq_len + (size_t)row * seq_len + col] =
        __float2half(0.0f);
}

// ─── cuda helpers ──────────────────────────────────────────────────────────

extern "C" void cuda_malloc(void **ptr, size_t size) { cudaMalloc(ptr, size); }
extern "C" void cuda_free(void *ptr) { cudaFree(ptr); }
extern "C" void cuda_memcpy_h2d(void *dst, const void *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice);
}
extern "C" void cuda_memcpy_d2h(void *dst, const void *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost);
}
extern "C" void cuda_memcpy_d2d(void *dst, const void *src, size_t size) {
  cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice);
}

// element-wise add: dst[i] += src[i] (f32 version for gradient accumulators)
__global__ void add_inplace_f32_kernel(float *dst, const float *src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  dst[i] += src[i];
}

extern "C" void cuda_add_inplace_f32(float *dst, const float *src, int n) {
  add_inplace_f32_kernel<<<(n + 255) / 256, 256>>>(dst, src, n);
}

// element-wise add: dst[i] += src[i] (f16 version for activations)
__global__ void add_inplace_f16_kernel(__half *dst, const __half *src, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  dst[i] = __float2half(__half2float(dst[i]) + __half2float(src[i]));
}

extern "C" void cuda_add_inplace_f16(__half *dst, const __half *src, int n) {
  add_inplace_f16_kernel<<<(n + 255) / 256, 256>>>(dst, src, n);
}

// ─── softmax backward with f32 dP (element-wise, keep) ──────────────────────

// One BLOCK per row (was one thread per row, walking the row serially with
// stride-`cols` uncoalesced access). This runs on the s x s attention matrix,
// so the work is quadratic in context while the old thread count was linear —
// the same shape of defect as the forward softmax. The row dot product is now a
// shared-memory reduction; its summation order changes, which can move the low
// bits of dS. Shared memory: blockDim.x floats.
__global__ void softmax_bw_f32_dP_kernel(const __half *P, const float *dP,
                                          __half *dS, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  // blockIdx.y selects the head — see fused_scale_causal_softmax_kernel.
  size_t plane = (size_t)blockIdx.y * rows * cols;
  const __half *P_row = P + plane + (size_t)row * cols;
  const float *dP_row = dP + plane + (size_t)row * cols;
  __half *dS_row = dS + plane + (size_t)row * cols;

  float partial = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x)
    partial += __half2float(P_row[i]) * dP_row[i];
  sdata[tid] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float dot = sdata[0];

  for (int i = tid; i < cols; i += blockDim.x) {
    float val = __half2float(P_row[i]) * (dP_row[i] - dot);
    val = fminf(fmaxf(val, -65504.0f), 65504.0f);
    dS_row[i] = __float2half(val);
  }
}

// ─── clamp f16 ──────────────────────────────────────────────────────────────

__global__ void clamp_f16_kernel(__half *buf, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(buf[i]);
  v = fminf(fmaxf(v, -65504.0f), 65504.0f);
  buf[i] = __float2half(v);
}

extern "C" void cuda_clamp_f16(__half *buf, int n) {
  clamp_f16_kernel<<<(n + 255) / 256, 256>>>(buf, n);
}

// ─── element-wise kernels (scale, causal_mask, softmax, gelu, bias_add, LN) ─

__global__ void scale_kernel_f16(__half *x, float scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  x[i] = __float2half(__half2float(x[i]) / scale);
}

__global__ void causal_mask_kernel(__half *scores, int seq_len) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len || col >= seq_len)
    return;
  if (col > row)
    scores[row * seq_len + col] = __float2half(-1e4f);
}

// row-wise softmax
__global__ void softmax_kernel(__half *x, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  __half *r = x + row * cols;
  float max_val = __half2float(r[0]);
  for (int i = 1; i < cols; i++)
    max_val = fmaxf(max_val, __half2float(r[i]));
  float sum = 0.0f;
  for (int i = 0; i < cols; i++) {
    float v = expf(__half2float(r[i]) - max_val);
    r[i] = __float2half(v);
    sum += v;
  }
  for (int i = 0; i < cols; i++)
    r[i] = __float2half(__half2float(r[i]) / sum);
}

__global__ void gelu_kernel(__half *x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float v = __half2float(x[i]);
  float c = 0.7978845608f * (v + 0.044715f * v * v * v);
  x[i] = __float2half(0.5f * v * (1.0f + tanhf(c)));
}

// adds bias to each row of x
// x: [rows x cols], bias: [cols]
__global__ void bias_add_kernel(__half *x, const __half *bias, int rows,
                                int cols) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows || col >= cols)
    return;
  x[row * cols + col] = __float2half(__half2float(x[row * cols + col]) + __half2float(bias[col]));
}

// ─── layer norm ─────────────────────────────────────────────────────────────

// One BLOCK per row, threads striding across the columns.
//
// This was one *thread* per row, each looping the whole row serially. Two things
// were wrong with that. Parallelism: a 512-row activation launched 512 threads
// on a device that wants tens of thousands, so ~1% of the GPU was awake.
// Memory: thread `row` walked `in + row*cols + i`, so adjacent threads touched
// addresses `cols` apart and every load became its own 32-byte transaction to
// use 2 bytes of it — a ~16x read amplification on top of the idle SMs.
//
// Block-per-row fixes both: adjacent threads now read adjacent addresses (fully
// coalesced), and the row's mean/variance come from a shared-memory reduction
// instead of a serial loop. `sdata` needs `blockDim.x` floats.
__global__ void layer_norm_kernel(const __half *in, __half *out,
                                  const __half *gamma, const __half *beta,
                                  int rows, int cols, float eps) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  const __half *row_in = in + (size_t)row * cols;
  __half *row_out = out + (size_t)row * cols;

  // Sum for the mean.
  float partial = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x)
    partial += __half2float(row_in[i]);
  sdata[tid] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float mean = sdata[0] / cols;
  __syncthreads();

  // Sum of squared deviations for the variance.
  partial = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x) {
    float diff = __half2float(row_in[i]) - mean;
    partial += diff * diff;
  }
  sdata[tid] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float inv_std = rsqrtf(sdata[0] / cols + eps);

  for (int i = tid; i < cols; i += blockDim.x)
    row_out[i] = __float2half(
        (__half2float(row_in[i]) - mean) * inv_std * __half2float(gamma[i]) + __half2float(beta[i]));
}

extern "C" void cuda_layer_norm(const __half *in, __half *out,
                                const __half *gamma, const __half *beta,
                                int rows, int cols, float eps) {
  // One block per row; 256 threads stride the columns. Shared memory is one
  // float per thread for the two reductions.
  int threads = 256;
  layer_norm_kernel<<<rows, threads, threads * sizeof(float)>>>(
      in, out, gamma, beta, rows, cols, eps);
}

// ─── Scale f32 array element-wise by 1/divisor, copy result to f16 ─────────
__global__ void f32_scale_and_copy_to_f16_kernel(const float *src, __half *dst,
                                                   float divisor, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  dst[i] = __float2half(src[i] / divisor);
}

// ─── RoPE (rotary position embeddings) ──────────────────────────────────────
//
// Position enters the model by rotating Q and K, not by adding a table to the
// token embeddings. Each adjacent pair (2i, 2i+1) of a head's dimensions is
// rotated by `pos * theta^(-2i/hd)`. Because the score is a dot product of a
// rotated Q against a rotated K, what survives is a function of the *difference*
// of their positions — relative position, for free and with no parameters.
//
// Adjacent pairing rather than the half-split (i, i+hd/2): no reshape, and it
// works for any even head_dim. `Model::init` rejects an odd one.
#define ROPE_THETA 10000.0f
#define ROPE_TWO_PI 6.2831853071795864769f

// The (cos, sin) for one (position, pair) slot.
//
// The angle is reduced into [-pi, pi] *before* the trig call. This file is
// compiled with `--use_fast_math`, which substitutes single-precision
// intrinsics, and those have poor argument reduction outside roughly one turn —
// while this angle reaches ~1e3 radians at long context and low pair index,
// which is precisely the long-range position RoPE exists to encode. Reducing
// first makes the result independent of that substitution.
__device__ __forceinline__ void rope_cos_sin(int pos, int pair, int hd,
                                              float *cos_t, float *sin_t) {
  float inv_freq = powf(ROPE_THETA, -2.0f * (float)pair / (float)hd);
  float angle = fmodf((float)pos * inv_freq, ROPE_TWO_PI);
  if (angle > ROPE_TWO_PI * 0.5f) angle -= ROPE_TWO_PI;
  sincosf(angle, sin_t, cos_t);
}

// Rotate one pair in place. `sign` is +1 in the forward and -1 in the backward:
// R is orthogonal, so R^T = R(-theta) is the same rotation with sin negated, and
// sharing this helper is what keeps the two from drifting apart.
__device__ __forceinline__ void rope_rotate(float *x0, float *x1,
                                             float cos_t, float sin_t, float sign) {
  float s = sign * sin_t;
  float a = *x0 * cos_t - *x1 * s;
  float b = *x0 * s + *x1 * cos_t;
  *x0 = a;
  *x1 = b;
}

// Forward: rotate the f32 Q/K in place (the score GEMM reads them), then emit
// the f16 copies the backward reads.
//
// One kernel over both tensors: they share an index space, and a thread doing
// two pairs costs less than a second grid. The f16 outputs are nullable so the
// inference path — which has no backward and so nothing to cache for — reuses
// exactly this rotation rather than a second copy of it.
//
// Grid is one thread per (head, position, pair), so each thread owns the two
// elements it rotates. Layout is [h, seq, hd].
__global__ void rope_qk_kernel(float *__restrict__ q_f32, float *__restrict__ k_f32,
                                __half *__restrict__ q16, __half *__restrict__ k16,
                                int seq, int hd, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = hd / 2;
  if (i >= h * seq * pairs) return;

  int pair = i % pairs;
  int pos = (i / pairs) % seq;
  int head = i / (pairs * seq);

  float cos_t, sin_t;
  rope_cos_sin(pos, pair, hd, &cos_t, &sin_t);

  size_t e0 = (size_t)head * seq * hd + (size_t)pos * hd + 2 * pair;
  size_t e1 = e0 + 1;

  rope_rotate(&q_f32[e0], &q_f32[e1], cos_t, sin_t, 1.0f);
  rope_rotate(&k_f32[e0], &k_f32[e1], cos_t, sin_t, 1.0f);

  if (q16) {
    q16[e0] = __float2half(q_f32[e0]);
    q16[e1] = __float2half(q_f32[e1]);
  }
  if (k16) {
    k16[e0] = __float2half(k_f32[e0]);
    k16[e1] = __float2half(k_f32[e1]);
  }
}

// Backward: undo the rotation on dQ/dK.
//
// The scores backward produces dQ' and dK' — gradients with respect to the
// *rotated* Q and K. Everything downstream (dX, dWq, dWk) wants them with
// respect to the unrotated ones, and Q' = R Q gives dQ = R^T dQ'. Same helper,
// sign flipped. dV is untouched: V is never rotated.
__global__ void rope_dqk_kernel(__half *__restrict__ dq, __half *__restrict__ dk,
                                 int seq, int hd, int h) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = hd / 2;
  if (i >= h * seq * pairs) return;

  int pair = i % pairs;
  int pos = (i / pairs) % seq;
  int head = i / (pairs * seq);

  float cos_t, sin_t;
  rope_cos_sin(pos, pair, hd, &cos_t, &sin_t);

  size_t e0 = (size_t)head * seq * hd + (size_t)pos * hd + 2 * pair;
  size_t e1 = e0 + 1;

  float q0 = __half2float(dq[e0]), q1 = __half2float(dq[e1]);
  rope_rotate(&q0, &q1, cos_t, sin_t, -1.0f);
  dq[e0] = __float2half(q0);
  dq[e1] = __float2half(q1);

  float k0 = __half2float(dk[e0]), k1 = __half2float(dk[e1]);
  rope_rotate(&k0, &k1, cos_t, sin_t, -1.0f);
  dk[e0] = __float2half(k0);
  dk[e1] = __float2half(k1);
}

// ─── KV-cache helpers (inference) ───────────────────────────────────────────

// Copy the post-RoPE f32 K out of the shared scratch into a cache whose row-block
// stride is `kv_stride` positions rather than `seq`.
//
// A plain memcpy will not do: the scratch is packed [h, seq, hd] with the *current*
// sequence length as its stride, while the cache is [h, kv_stride, hd] so that it
// can be appended to as the sequence grows. Same bytes, different pitch.
__global__ void cache_k_kernel(const float *__restrict__ src, float *__restrict__ dst,
                                int seq, int hd, int h, int kv_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= h * seq * hd) return;
  int j = i % hd;
  int pos = (i / hd) % seq;
  int head = i / (hd * seq);
  dst[(size_t)head * kv_stride * hd + (size_t)pos * hd + j] = src[i];
}

// RoPE for a single new position, over Q and the one K row just written into the
// cache.
//
// Separate from `rope_qk_kernel` because the two tensors no longer share an index
// space: Q is [h, 1, hd] packed tight, while the K row lives inside a
// [h, kv_stride, hd] cache. Only the indexing differs — the rotation itself goes
// through the same `rope_cos_sin`/`rope_rotate` device helpers the training forward
// uses, which is the part that has to agree.
__global__ void rope_decode_kernel(float *__restrict__ q_f32, float *__restrict__ k_row,
                                    int hd, int h, int pos, int kv_stride) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  int pairs = hd / 2;
  if (i >= h * pairs) return;

  int pair = i % pairs;
  int head = i / pairs;

  float cos_t, sin_t;
  rope_cos_sin(pos, pair, hd, &cos_t, &sin_t);

  size_t q0 = (size_t)head * hd + 2 * pair;
  rope_rotate(&q_f32[q0], &q_f32[q0 + 1], cos_t, sin_t, 1.0f);

  size_t k0 = (size_t)head * kv_stride * hd + 2 * pair;
  rope_rotate(&k_row[k0], &k_row[k0 + 1], cos_t, sin_t, 1.0f);
}

// Softmax over one query row of `s` cached keys, per head.
//
// No causal mask: a single query at absolute position `pos` attends to every
// cached position 0..pos, which is all `s = pos + 1` of them. The mask in
// `fused_scale_causal_softmax_kernel` exists to stop position i seeing i+1..seq
// within a block of queries, and a one-query block has nothing to hide.
//
// The arithmetic is otherwise identical to that kernel's, deliberately: shared
// memory max reduction over the row, `inv_scale` kept *inside* the exponent, then
// a normalizing pass. exp((x - max) * c) == exp(x*c - max*c) for c > 0, so this is
// the same softmax a prefill computes for the same row — which is what
// `decode_equals_prefill` checks.
__global__ void decode_softmax_kernel(const float *__restrict__ scores_f32,
                                       __half *__restrict__ scores_f16,
                                       float inv_scale, int s) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  size_t plane = (size_t)blockIdx.x * s;
  const float *row_in = scores_f32 + plane;
  __half *row_out = scores_f16 + plane;

  float m = -1e38f;
  for (int j = tid; j < s; j += blockDim.x)
    if (row_in[j] > m) m = row_in[j];
  sdata[tid] = m;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (tid < k && sdata[tid + k] > sdata[tid]) sdata[tid] = sdata[tid + k];
    __syncthreads();
  }
  float max_val = sdata[0];
  __syncthreads();

  float sum = 0.0f;
  for (int j = tid; j < s; j += blockDim.x) {
    float e = expf((row_in[j] - max_val) * inv_scale);
    row_out[j] = __float2half(e);
    sum += e;
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int k = blockDim.x / 2; k > 0; k >>= 1) {
    if (tid < k) sdata[tid] += sdata[tid + k];
    __syncthreads();
  }
  float inv_sum = 1.0f / sdata[0];

  for (int j = tid; j < s; j += blockDim.x)
    row_out[j] = __float2half(__half2float(row_out[j]) * inv_sum);
}

// ─── Single-token attention against a KV cache (inference decode) ───────────
//
// One new token at absolute position `pos`, attending to the `pos` positions
// already in the cache plus itself. Where the prefill re-derives K and V for every
// position on every call — O(s) projections and an O(s²) score matrix per token —
// this projects one row, appends it, and scores it against what is already there.
//
// K is cached in f32 and V in f16 because that is exactly what the prefill's GEMMs
// produce (`attn.k_f32`, and V straight out of its f16 GEMM). Caching K as f16
// would make a decode step's score GEMM lower precision than a prefill's, and the
// two must agree.
extern "C" void cuda_attention_decode(
    const __half *x_row,      // [1, d] the new token's layer input
    const __half *wq_all,     // [h, d, hd]
    const __half *wk_all,
    const __half *wv_all,
    float *k_cache,           // [h, kv_stride, hd] f32, positions < pos filled
    __half *v_cache,          // [h, kv_stride, hd] f16
    __half *out,              // [h, 1, hd] — contiguous, which is also [1, d]
    int d, int hd, int h, int pos, int kv_stride) {

  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  int s = pos + 1;
  size_t cache_plane = (size_t)kv_stride * hd;

  // 1. Q = x @ Wq  → [h, 1, hd] f32 in the shared scratch.
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, 1, d, &alpha,
      wq_all, CUDA_R_16F, hd, (long long)d * hd,
      x_row, CUDA_R_16F, d, 0,
      &beta,
      attn.q_f32, CUDA_R_32F, hd, hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 2. K = x @ Wk  → straight into the cache at row `pos`.
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, 1, d, &alpha,
      wk_all, CUDA_R_16F, hd, (long long)d * hd,
      x_row, CUDA_R_16F, d, 0,
      &beta,
      k_cache + (size_t)pos * hd, CUDA_R_32F, hd, (long long)cache_plane,
      h, COMPUTE_F32, TENSOR_OP));

  // 3. V = x @ Wv  → likewise, and V is never rotated.
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, 1, d, &alpha,
      wv_all, CUDA_R_16F, hd, (long long)d * hd,
      x_row, CUDA_R_16F, d, 0,
      &beta,
      v_cache + (size_t)pos * hd, CUDA_R_16F, hd, (long long)cache_plane,
      h, COMPUTE_F32, TENSOR_OP));

  // 4. Rotate Q and the new K by the *absolute* position. The cached rows keep the
  //    rotation they were written with, which is why a shifted context window
  //    cannot reuse this cache — see `InferSession::logits_last`.
  {
    int n = h * (hd / 2);
    rope_decode_kernel<<<(n + 255) / 256, 256>>>(
        attn.q_f32, k_cache + (size_t)pos * hd, hd, h, pos, kv_stride);
  }

  // 5. scores = Q · K_cache^T  → [h, 1, s] f32.
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      s, 1, hd, &alpha,
      k_cache, CUDA_R_32F, hd, (long long)cache_plane,
      attn.q_f32, CUDA_R_32F, hd, hd,
      &beta,
      attn.dec_f32, CUDA_R_32F, s, s,
      h, COMPUTE_F32, TENSOR_OP));

  // 6. Softmax over the s cached keys, one block per head.
  {
    float inv_scale = 1.0f / sqrtf((float)hd);
    int t = 256;
    decode_softmax_kernel<<<h, t, t * sizeof(float)>>>(
        attn.dec_f32, attn.dec_f16, inv_scale, s);
  }

  // 7. out = P · V_cache  → [h, 1, hd].
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, 1, s, &alpha,
      v_cache, CUDA_R_16F, hd, (long long)cache_plane,
      attn.dec_f16, CUDA_R_16F, s, s,
      &beta,
      out, CUDA_R_16F, hd, hd,
      h, COMPUTE_F32, TENSOR_OP));
}

// ════════════════════════════════════════════════════════════════════════════
// FlashAttention
// ════════════════════════════════════════════════════════════════════════════
//
// Replaces, surgically, three steps of the forward (scores = Q@K^T, the fused
// scale+mask+softmax, out = P@V) and five of the backward (dP, softmax backward,
// scale, mask, dV/dQ/dK). Everything else — the QKV projections, RoPE, the KV
// cache write, the dX GEMMs, the dW GEMMs — is untouched and still cuBLAS.
//
// What it buys is not FLOPs, it is the [h, seq, seq] score matrix never existing
// in HBM. The old path writes an f32 score plane, reads it back, writes an f16
// P, reads P three more times; the backward writes dP f32, reads it, writes dS,
// and reads dS four separate times (scale, mask, the dQ GEMM, the dK GEMM). At
// d512/h8/L8/ctx1024 that is ~2.5 GB of DRAM traffic per step for a tensor no
// one outside this block ever looks at, plus ~230 MB of resident allocation.
// Here the tile lives in shared memory and the only thing saved for the backward
// is a per-row logsumexp, [h, seq] f32.
//
// The causal mask is also structural rather than elementwise: whole Bc tiles
// above the diagonal are never visited, where the old score GEMM computed them
// in full and then threw half of them away.

// ─── The wmma accumulator layout ────────────────────────────────────────────
//
// mma_sync's accumulator fragment distributes a 16x16 f32 tile over the warp's
// 32 lanes, 8 elements each. **That mapping is not specified by the API.** It is
// the one below on Ampere, and everything in this file depends on it: the online
// softmax keeps its per-row state in registers rather than shared memory (a lane
// owns exactly 2 rows), a row reduction is two shuffles (the 4 lanes sharing a
// row differ only in bits 0-1), and writing P transposed costs nothing (the
// store is elementwise through this map anyway, which is what makes dV = P^T dO
// and dK = dS^T Q need no extra kernel).
//
// So it is verified once at startup and aborts on mismatch, for the same reason
// CUBLAS_CHECK aborts: a wrong layout does not fault. It produces finite,
// plausible gradients, the loss still descends, and it is discoverable only
// hours later in a ruined model. This file has shipped that failure twice.
#define ACC_ROW(lane, e) (((lane) >> 2) + 8 * (((e) >> 1) & 1))
#define ACC_COL(lane, e) (((lane) & 3) * 2 + ((e) & 1) + 8 * ((e) >> 2))

// Loads a 16x16 accumulator whose element (r, c) holds r*16+c, then has every
// lane check that each of its 8 elements sits where the macros claim.
__global__ void flash_probe_acc_layout_kernel(int *ok) {
  __shared__ float tile[16 * 16];
  for (int i = threadIdx.x; i < 256; i += blockDim.x)
    tile[i] = (float)((i / 16) * 16 + (i % 16));
  __syncthreads();

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
  wmma::load_matrix_sync(acc, tile, 16, wmma::mem_row_major);

  int lane = threadIdx.x & 31;
  int good = 1;
  if (acc.num_elements != 8) good = 0;
#pragma unroll
  for (int e = 0; e < 8; ++e) {
    int want = ACC_ROW(lane, e) * 16 + ACC_COL(lane, e);
    if ((int)acc.x[e] != want) good = 0;
  }
  if (!good) *ok = 0;
}

// Returns 1 if the layout matches. Callers inside this file abort; the Rust side
// exposes it as a test so a driver or arch change is a red test rather than a
// corrupted run.
extern "C" int cuda_flash_probe_acc_layout(void) {
  int *d_ok = NULL;
  cudaMalloc(&d_ok, sizeof(int));
  int one = 1;
  cudaMemcpy(d_ok, &one, sizeof(int), cudaMemcpyHostToDevice);
  flash_probe_acc_layout_kernel<<<1, 32>>>(d_ok);
  int host = 0;
  cudaMemcpy(&host, d_ok, sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_ok);
  return host;
}

static void flash_assert_acc_layout(void) {
  static bool checked = false;
  if (checked) return;
  checked = true;
  if (!cuda_flash_probe_acc_layout()) {
    fprintf(stderr,
            "wmma accumulator layout does not match ACC_ROW/ACC_COL at %s:%d — "
            "the flash attention kernels would compute silently wrong "
            "gradients. Refusing to run.\n",
            __FILE__, __LINE__);
    abort();
  }
}

// ─── Flash forward ──────────────────────────────────────────────────────────
//
// O = softmax_causal(Q K^T / sqrt(hd)) V, with the score tile never leaving
// shared memory.
//
// The scale stays semantically where the old kernel had it. That kernel computed
// exp((x - rowmax(x)) * c); since exp(x*c - rowmax(x*c)) is the same thing for
// c > 0, this scales immediately after the matmul and runs a textbook online
// softmax over the scaled scores. `inv_scale` is computed on the host and passed
// in for both this and the backward — the file is built --use_fast_math, so
// -prec-sqrt=false makes a device-side 1/sqrtf(hd) a different number.
//
// `lse` receives the per-row logsumexp of the *scaled* scores. It is the entire
// state the backward needs in place of the [h, seq, seq] softmax plane: h*seq
// floats against h*seq*seq halves.
//
// `ent` is the optional per-head attention-entropy accumulator that used to be a
// separate pass over the saved P. With p_j = e^(s_j - m)/l,
//   H = -Σ p log p = (m + log l) - (1/l)·Σ e^(s_j - m)·s_j = lse - t/l
// so one more running accumulator alongside `l`, rescaled by the same alpha,
// replaces reading back the whole plane. Its atomicAdd makes the *diagnostic*
// non-deterministic in the last bits; nothing downstream of the gradient touches
// it.
template <int HD, int BR, int BC>
__global__ __launch_bounds__(FLASH_THREADS) void flash_fwd_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, __half *__restrict__ O,
    float *__restrict__ lse, float *__restrict__ ent, float inv_scale, int S,
    int v_rows) {
  constexpr int NK = HD / 16, NC = BC / 16;
  constexpr int LDK = HD + FLASH_PAD, LDP = BC + FLASH_PAD;

  extern __shared__ __half flash_smem[];
  __half *Ks = flash_smem;
  __half *Vs = Ks + BC * LDK;
  __half *Ps = Vs + BC * LDK;

  const int head = blockIdx.y;
  // Causal work per block grows with its row index, so issue the longest blocks
  // first: with one wave and a bit, the tail is otherwise one long block
  // finishing alone.
  const int ib = gridDim.x - 1 - blockIdx.x;
  const int row0 = ib * BR;
  if (row0 >= S) return;  // uniform over the block, so no lane is stranded

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t plane = (size_t)head * S * HD;
  // V may be pitched wider than `seq`: on an inference prefill it *is* the KV
  // cache, whose rows are `kv_stride` apart so the decode path can append to it.
  // Rows within a plane are still HD apart; only the per-head stride differs.
  const size_t vplane = (size_t)head * v_rows * HD;
  const int qrow = row0 + warp * 16;

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[NK];
#pragma unroll
  for (int k = 0; k < NK; ++k)
    wmma::load_matrix_sync(q_frag[k], Q + plane + (size_t)qrow * HD + k * 16, HD);

  // A lane owns exactly two rows of the accumulator — ACC_ROW's high bit — so
  // the whole online-softmax state is six registers, not a shared array.
  float m_i[2] = {-INFINITY, -INFINITY};
  float l_i[2] = {0.0f, 0.0f};
  float t_i[2] = {0.0f, 0.0f};

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_o[NK];
#pragma unroll
  for (int k = 0; k < NK; ++k) wmma::fill_fragment(acc_o[k], 0.0f);

  // Causal block skipping: a tile whose first column is past this block's last
  // row is entirely masked, so it is never visited rather than computed and
  // discarded. With BR == BC that is exactly half the tile pairs.
  const int jmax = min((row0 + BR - 1) / BC, (S - 1) / BC);

  for (int jb = 0; jb <= jmax; ++jb) {
    const int col0 = jb * BC;

    __syncthreads();
    // Stage K and V, zero past the sequence tail. The zeros matter: the masks
    // below kill the scores, but leaving stale halves here would still feed the
    // backward's dP = dO V^T, which reads V unconditionally.
    for (int t = threadIdx.x; t < BC * (HD / 8); t += blockDim.x) {
      const int r = t / (HD / 8), c8 = (t % (HD / 8)) * 8;
      const int4 zero = {0, 0, 0, 0};
      const bool live = (col0 + r) < S;
      *(int4 *)(Ks + r * LDK + c8) =
          live ? *(const int4 *)(K + plane + (size_t)(col0 + r) * HD + c8) : zero;
      *(int4 *)(Vs + r * LDK + c8) =
          live ? *(const int4 *)(V + vplane + (size_t)(col0 + r) * HD + c8) : zero;
    }
    __syncthreads();

    // S_tile = Q K^T. A col_major matrix_b fragment reads element (k, n) at
    // ptr[n*ldm + k], so loading one from the row-major Ks tile *is* K^T — no
    // transpose kernel, no second staging.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s[NC];
#pragma unroll
    for (int n = 0; n < NC; ++n) wmma::fill_fragment(acc_s[n], 0.0f);
#pragma unroll
    for (int n = 0; n < NC; ++n) {
#pragma unroll
      for (int k = 0; k < NK; ++k) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf;
        wmma::load_matrix_sync(kf, Ks + (n * 16) * LDK + k * 16, LDK);
        wmma::mma_sync(acc_s[n], q_frag[k], kf, acc_s[n]);
      }
    }

    // Scale, mask, and this tile's row max — all in registers.
    //
    // The mask is -INFINITY, never a finite sentinel: a finite one makes
    // (s - m) == 0 for an all-masked row, so exp gives 1 and the row picks up a
    // weight it should not have.
    float mt[2] = {-INFINITY, -INFINITY};
#pragma unroll
    for (int n = 0; n < NC; ++n) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int r = row0 + warp * 16 + ACC_ROW(lane, e);
        const int c = col0 + n * 16 + ACC_COL(lane, e);
        float s = acc_s[n].x[e] * inv_scale;
        if (r >= S || c >= S || c > r) s = -INFINITY;
        acc_s[n].x[e] = s;
        const int h2 = (e >> 1) & 1;
        mt[h2] = fmaxf(mt[h2], s);
      }
    }
    // The four lanes holding one row differ only in bits 0-1 of the lane id.
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      mt[h2] = fmaxf(mt[h2], __shfl_xor_sync(0xffffffffu, mt[h2], 1));
      mt[h2] = fmaxf(mt[h2], __shfl_xor_sync(0xffffffffu, mt[h2], 2));
    }

    // The rescale factor, and the only place inf - inf could arise.
    float alpha[2];
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const float mn = fmaxf(m_i[h2], mt[h2]);
      alpha[h2] = (m_i[h2] == -INFINITY) ? 0.0f
                  : (mn == -INFINITY)    ? 1.0f
                                         : __expf(m_i[h2] - mn);
      m_i[h2] = mn;
    }

    float ls[2] = {0.0f, 0.0f}, ts[2] = {0.0f, 0.0f};
#pragma unroll
    for (int n = 0; n < NC; ++n) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h2 = (e >> 1) & 1;
        const float s = acc_s[n].x[e];
        const float p = (s == -INFINITY) ? 0.0f : __expf(s - m_i[h2]);
        acc_s[n].x[e] = p;
        ls[h2] += p;
        ts[h2] += p * s;  // entropy only; s is finite wherever p != 0
      }
    }
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
#pragma unroll
      for (int b = 1; b <= 2; b <<= 1) {
        ls[h2] += __shfl_xor_sync(0xffffffffu, ls[h2], b);
        ts[h2] += __shfl_xor_sync(0xffffffffu, ts[h2], b);
      }
      l_i[h2] = l_i[h2] * alpha[h2] + ls[h2];
      t_i[h2] = t_i[h2] * alpha[h2] + ts[h2];
    }
    // Rescale the running output in place — no round trip through memory.
#pragma unroll
    for (int k = 0; k < NK; ++k) {
#pragma unroll
      for (int e = 0; e < 8; ++e) acc_o[k].x[e] *= alpha[(e >> 1) & 1];
    }

    // P must become a matrix_a fragment, and the API cannot reinterpret an
    // accumulator as one, so it goes through shared memory. The store is
    // elementwise through the layout map regardless, which is what makes the
    // transposed store in the backward free.
    __half *Pw = Ps + warp * 16 * LDP;
#pragma unroll
    for (int n = 0; n < NC; ++n) {
#pragma unroll
      for (int e = 0; e < 8; ++e)
        Pw[ACC_ROW(lane, e) * LDP + n * 16 + ACC_COL(lane, e)] =
            __float2half(acc_s[n].x[e]);
    }
    __syncwarp();  // Pw is private to this warp

#pragma unroll
    for (int k = 0; k < NK; ++k) {
#pragma unroll
      for (int n = 0; n < NC; ++n) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> vf;
        wmma::load_matrix_sync(pf, Pw + n * 16, LDP);
        wmma::load_matrix_sync(vf, Vs + (n * 16) * LDK + k * 16, LDK);
        wmma::mma_sync(acc_o[k], pf, vf, acc_o[k]);
      }
    }
    __syncwarp();
  }

  // Normalize and store. Rows past `seq` are never written, which is what keeps
  // an all-masked row (l == 0) from reaching memory at all: with causal masking
  // and BR == BC every real row's diagonal tile holds its own diagonal element,
  // so l >= 1 for every row that is stored. A windowed or non-causal variant
  // would break that invariant and would need this reasoned through again.
#pragma unroll
  for (int k = 0; k < NK; ++k) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int h2 = (e >> 1) & 1;
      const float inv = (l_i[h2] > 0.0f) ? 1.0f / l_i[h2] : 0.0f;
      const float v = acc_o[k].x[e] * inv;
      const int r = row0 + warp * 16 + ACC_ROW(lane, e);
      const int c = k * 16 + ACC_COL(lane, e);
      if (r < S)
        O[plane + (size_t)r * HD + c] =
            __float2half(fminf(fmaxf(v, -65504.0f), 65504.0f));
    }
  }

  // An inference prefill has no backward to feed and passes both as null.
  if ((lane & 3) == 0 && (lse || ent)) {
#pragma unroll
    for (int h2 = 0; h2 < 2; ++h2) {
      const int r = row0 + warp * 16 + (lane >> 2) + 8 * h2;
      if (r >= S) continue;
      // logf, not __logf: this feeds an exp in the backward, which amplifies the
      // intrinsic's relative error. It runs once per row.
      const float row_lse = m_i[h2] + logf(l_i[h2]);
      if (lse) lse[(size_t)head * S + r] = row_lse;
      // Row 0 attends to one key: zero entropy, and ln(1) is not a normalizer.
      if (ent && r >= 1)
        atomicAdd(&ent[head], (row_lse - t_i[h2] / l_i[h2]) / logf((float)(r + 1)));
    }
  }
}

// ─── Flash backward ─────────────────────────────────────────────────────────
//
// Three kernels, in place of the old dP → softmax-bw → scale → mask → dV/dQ/dK
// chain. None of them materializes an s²-sized tensor.
//
// The backward is parallelized over **two passes** rather than FlashAttention-2's
// single K/V-block pass. FA-2 makes dK/dV register-resident and pays for it with
// an atomicAdd into dQ — ~4 M atomics per layer at d512/h8/ctx1024, plus an
// h·s·hd f32 scratch that has to be memset every layer of every step, and a
// result that is **not reproducible run to run**. Two passes recompute Q K^T and
// dO V^T, costing ~40% more backward FLOPs on a budget that is not the
// bottleneck once the dP/dS traffic is gone, and every output is written exactly
// once by exactly one block.
//
// Determinism is the deciding argument, not a preference. This file has twice
// shipped a silently wrong gradient that still descended a loss curve (dK frozen
// at zero for the life of the project; the LN2 backward overrunning its scratch),
// and the test that catches that class is an exact A/B against the old path. A
// non-deterministic reduction makes "did this change the gradient?" unanswerable.

// D = rowsum(dO ∘ O), the softmax-backward correction term, [h, seq] f32.
//
// Equivalent to what the old kernel computed: softmax_bw_f32_dP_kernel formed
// dot = Σ_j P_j·dP_j with dP_j = dO·V_j, and Σ_j P_j (dO·V_j) = dO·(Σ_j P_j V_j)
// = dO·O. One warp per row.
__global__ void flash_bwd_preprocess_kernel(const __half *__restrict__ dO,
                                            const __half *__restrict__ O,
                                            float *__restrict__ D, int S,
                                            int HD) {
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int head = blockIdx.y, lane = threadIdx.x & 31;
  if (row >= S) return;
  const size_t off = (size_t)head * S * HD + (size_t)row * HD;
  float s = 0.0f;
  for (int c = lane; c < HD; c += 32)
    s += __half2float(dO[off + c]) * __half2float(O[off + c]);
#pragma unroll
  for (int b = 16; b; b >>= 1) s += __shfl_xor_sync(0xffffffffu, s, b);
  if (lane == 0) D[(size_t)head * S + row] = s;
}

// Pass 2 of 2 — dQ, one block per (query block, head).
//
// Structurally the forward with the online maximum removed: `lse` is already
// known, so P is a single exp with no running rescale.
template <int HD, int BR, int BC>
__global__ __launch_bounds__(FLASH_THREADS) void flash_bwd_dq_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, const __half *__restrict__ dO,
    const float *__restrict__ lse, const float *__restrict__ D,
    __half *__restrict__ dQ, float inv_scale, int S) {
  constexpr int NK = HD / 16, NC = BC / 16;
  constexpr int LDK = HD + FLASH_PAD, LDP = BC + FLASH_PAD;

  extern __shared__ __half flash_smem[];
  __half *Ks = flash_smem;
  __half *Vs = Ks + BC * LDK;
  __half *Ss = Vs + BC * LDK;  // dS, per-warp private 16-row slabs

  const int head = blockIdx.y;
  const int ib = gridDim.x - 1 - blockIdx.x;
  const int row0 = ib * BR;
  if (row0 >= S) return;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t plane = (size_t)head * S * HD;
  const int qrow = row0 + warp * 16;

  // The two rows this lane owns, and their forward statistics. Rows past `seq`
  // read whatever the rounded-up allocation holds; their results are never
  // stored, and `live` below keeps the garbage out of the arithmetic anyway.
  float L_i[2], D_i[2];
#pragma unroll
  for (int h2 = 0; h2 < 2; ++h2) {
    const int r = row0 + warp * 16 + (lane >> 2) + 8 * h2;
    L_i[h2] = (r < S) ? lse[(size_t)head * S + r] : 0.0f;
    D_i[h2] = (r < S) ? D[(size_t)head * S + r] : 0.0f;
  }

  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> q_frag[NK];
  wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> do_frag[NK];
#pragma unroll
  for (int k = 0; k < NK; ++k) {
    wmma::load_matrix_sync(q_frag[k], Q + plane + (size_t)qrow * HD + k * 16, HD);
    wmma::load_matrix_sync(do_frag[k], dO + plane + (size_t)qrow * HD + k * 16, HD);
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dq_acc[NK];
#pragma unroll
  for (int k = 0; k < NK; ++k) wmma::fill_fragment(dq_acc[k], 0.0f);

  const int jmax = min((row0 + BR - 1) / BC, (S - 1) / BC);

  for (int jb = 0; jb <= jmax; ++jb) {
    const int col0 = jb * BC;

    __syncthreads();
    for (int t = threadIdx.x; t < BC * (HD / 8); t += blockDim.x) {
      const int r = t / (HD / 8), c8 = (t % (HD / 8)) * 8;
      const int4 zero = {0, 0, 0, 0};
      const bool live = (col0 + r) < S;
      *(int4 *)(Ks + r * LDK + c8) =
          live ? *(const int4 *)(K + plane + (size_t)(col0 + r) * HD + c8) : zero;
      *(int4 *)(Vs + r * LDK + c8) =
          live ? *(const int4 *)(V + plane + (size_t)(col0 + r) * HD + c8) : zero;
    }
    __syncthreads();

    // S_tile = Q K^T and dP = dO V^T, both from col_major fragments over the
    // row-major staged tiles — the same free-transpose trick as the forward.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_s[NC], acc_dp[NC];
#pragma unroll
    for (int n = 0; n < NC; ++n) {
      wmma::fill_fragment(acc_s[n], 0.0f);
      wmma::fill_fragment(acc_dp[n], 0.0f);
    }
#pragma unroll
    for (int n = 0; n < NC; ++n) {
#pragma unroll
      for (int k = 0; k < NK; ++k) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf, vf;
        wmma::load_matrix_sync(kf, Ks + (n * 16) * LDK + k * 16, LDK);
        wmma::load_matrix_sync(vf, Vs + (n * 16) * LDK + k * 16, LDK);
        wmma::mma_sync(acc_s[n], q_frag[k], kf, acc_s[n]);
        wmma::mma_sync(acc_dp[n], do_frag[k], vf, acc_dp[n]);
      }
    }

    // dS = P ∘ (dP - D), scaled. A masked entry is set to zero *by construction*
    // rather than by multiplying a zero P: dP for an out-of-range column is
    // whatever the zero-filled V tile produced, and 0 * NaN is NaN.
    __half *Sw = Ss + warp * 16 * LDP;
#pragma unroll
    for (int n = 0; n < NC; ++n) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int h2 = (e >> 1) & 1;
        const int r = row0 + warp * 16 + ACC_ROW(lane, e);
        const int c = col0 + n * 16 + ACC_COL(lane, e);
        const bool live = (r < S) && (c < S) && (c <= r);
        float ds = 0.0f;
        if (live) {
          const float p = __expf(acc_s[n].x[e] * inv_scale - L_i[h2]);
          ds = p * (acc_dp[n].x[e] - D_i[h2]) * inv_scale;
          ds = fminf(fmaxf(ds, -65504.0f), 65504.0f);
        }
        Sw[ACC_ROW(lane, e) * LDP + n * 16 + ACC_COL(lane, e)] = __float2half(ds);
      }
    }
    __syncwarp();

    // dQ += dS @ K — K as an ordinary row_major matrix_b this time.
#pragma unroll
    for (int k = 0; k < NK; ++k) {
#pragma unroll
      for (int n = 0; n < NC; ++n) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> sf;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> kf;
        wmma::load_matrix_sync(sf, Sw + n * 16, LDP);
        wmma::load_matrix_sync(kf, Ks + (n * 16) * LDK + k * 16, LDK);
        wmma::mma_sync(dq_acc[k], sf, kf, dq_acc[k]);
      }
    }
    __syncwarp();
  }

#pragma unroll
  for (int k = 0; k < NK; ++k) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = row0 + warp * 16 + ACC_ROW(lane, e);
      const int c = k * 16 + ACC_COL(lane, e);
      if (r < S)
        dQ[plane + (size_t)r * HD + c] = __float2half(
            fminf(fmaxf(dq_acc[k].x[e], -65504.0f), 65504.0f));
    }
  }
}

// Pass 1 of 2 — dK and dV, one block per (key block, head).
//
// Two phases per query block, with a barrier between them, because the two
// matmuls have different M dimensions: phase A partitions the BR query rows
// across the four warps, phase B partitions the BC key rows. P and dS are
// written to shared memory **transposed**, which costs nothing — the store goes
// through the accumulator layout map elementwise either way — and that is what
// makes dV = P^T dO and dK = dS^T Q need no separate transpose.
template <int HD, int BR, int BC>
__global__ __launch_bounds__(FLASH_THREADS) void flash_bwd_dkv_kernel(
    const __half *__restrict__ Q, const __half *__restrict__ K,
    const __half *__restrict__ V, const __half *__restrict__ dO,
    const float *__restrict__ lse, const float *__restrict__ D,
    __half *__restrict__ dK, __half *__restrict__ dV, float inv_scale, int S) {
  constexpr int NK = HD / 16, NC = BC / 16, NR = BR / 16;
  constexpr int LDK = HD + FLASH_PAD, LDT = BR + FLASH_PAD;

  extern __shared__ __half flash_smem[];
  __half *Ks = flash_smem;
  __half *Vs = Ks + BC * LDK;
  __half *Pt = Vs + BC * LDK;    // [BC][BR] — P transposed
  __half *dSt = Pt + BC * LDT;   // [BC][BR] — dS transposed
  float *Ls = (float *)(dSt + BC * LDT);
  float *Ds = Ls + BR;

  const int head = blockIdx.y;
  const int jb = blockIdx.x;
  const int col0 = jb * BC;
  if (col0 >= S) return;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const size_t plane = (size_t)head * S * HD;

  for (int t = threadIdx.x; t < BC * (HD / 8); t += blockDim.x) {
    const int r = t / (HD / 8), c8 = (t % (HD / 8)) * 8;
    const int4 zero = {0, 0, 0, 0};
    const bool live = (col0 + r) < S;
    *(int4 *)(Ks + r * LDK + c8) =
        live ? *(const int4 *)(K + plane + (size_t)(col0 + r) * HD + c8) : zero;
    *(int4 *)(Vs + r * LDK + c8) =
        live ? *(const int4 *)(V + plane + (size_t)(col0 + r) * HD + c8) : zero;
  }

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> dv_acc[NK], dk_acc[NK];
#pragma unroll
  for (int k = 0; k < NK; ++k) {
    wmma::fill_fragment(dv_acc[k], 0.0f);
    wmma::fill_fragment(dk_acc[k], 0.0f);
  }

  // Causal: a query block can contribute only if its last row reaches this key
  // block's first column.
  for (int ib = col0 / BR; ib * BR < S; ++ib) {
    const int row0 = ib * BR;
    const int qrow = row0 + warp * 16;

    __syncthreads();
    for (int t = threadIdx.x; t < BR; t += blockDim.x) {
      const int r = row0 + t;
      Ls[t] = (r < S) ? lse[(size_t)head * S + r] : 0.0f;
      Ds[t] = (r < S) ? D[(size_t)head * S + r] : 0.0f;
    }
    __syncthreads();

    // ── Phase A: P and dS for this (query block, key block) pair ────────────
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_p[NC], acc_dp[NC];
#pragma unroll
    for (int n = 0; n < NC; ++n) {
      wmma::fill_fragment(acc_p[n], 0.0f);
      wmma::fill_fragment(acc_dp[n], 0.0f);
    }
#pragma unroll
    for (int k = 0; k < NK; ++k) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> qf, dof;
      wmma::load_matrix_sync(qf, Q + plane + (size_t)qrow * HD + k * 16, HD);
      wmma::load_matrix_sync(dof, dO + plane + (size_t)qrow * HD + k * 16, HD);
#pragma unroll
      for (int n = 0; n < NC; ++n) {
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> kf, vf;
        wmma::load_matrix_sync(kf, Ks + (n * 16) * LDK + k * 16, LDK);
        wmma::load_matrix_sync(vf, Vs + (n * 16) * LDK + k * 16, LDK);
        wmma::mma_sync(acc_p[n], qf, kf, acc_p[n]);
        wmma::mma_sync(acc_dp[n], dof, vf, acc_dp[n]);
      }
    }

#pragma unroll
    for (int n = 0; n < NC; ++n) {
#pragma unroll
      for (int e = 0; e < 8; ++e) {
        const int rl = warp * 16 + ACC_ROW(lane, e);   // row within the Q block
        const int cl = n * 16 + ACC_COL(lane, e);      // col within the K block
        const int r = row0 + rl, c = col0 + cl;
        const bool live = (r < S) && (c < S) && (c <= r);
        float p = 0.0f, ds = 0.0f;
        if (live) {
          p = __expf(acc_p[n].x[e] * inv_scale - Ls[rl]);
          ds = p * (acc_dp[n].x[e] - Ds[rl]) * inv_scale;
          ds = fminf(fmaxf(ds, -65504.0f), 65504.0f);
        }
        Pt[cl * LDT + rl] = __float2half(p);
        dSt[cl * LDT + rl] = __float2half(ds);
      }
    }
    __syncthreads();

    // ── Phase B: dV += P^T dO and dK += dS^T Q, partitioned by key row ──────
    //
    // Phase A partitions BR query rows over the four warps; this partitions BC
    // key rows. When BC < FLASH_WARPS*16 (hd 128 runs BC=32) the surplus warps
    // have no key slab and sit this phase out. The guard is warp-uniform, which
    // is what makes skipping a mma_sync legal — a *lane* may never skip one.
    if (warp * 16 < BC) {
#pragma unroll
    for (int k = 0; k < NK; ++k) {
#pragma unroll
      for (int t = 0; t < NR; ++t) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> pt, st;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> ob, qb;
        wmma::load_matrix_sync(pt, Pt + (warp * 16) * LDT + t * 16, LDT);
        wmma::load_matrix_sync(st, dSt + (warp * 16) * LDT + t * 16, LDT);
        wmma::load_matrix_sync(ob, dO + plane + (size_t)(row0 + t * 16) * HD + k * 16, HD);
        wmma::load_matrix_sync(qb, Q + plane + (size_t)(row0 + t * 16) * HD + k * 16, HD);
        wmma::mma_sync(dv_acc[k], pt, ob, dv_acc[k]);
        wmma::mma_sync(dk_acc[k], st, qb, dk_acc[k]);
      }
    }
    }
    __syncthreads();
  }

  if (warp * 16 >= BC) return;
#pragma unroll
  for (int k = 0; k < NK; ++k) {
#pragma unroll
    for (int e = 0; e < 8; ++e) {
      const int r = col0 + warp * 16 + ACC_ROW(lane, e);
      const int c = k * 16 + ACC_COL(lane, e);
      if (r < S) {
        dV[plane + (size_t)r * HD + c] = __float2half(
            fminf(fmaxf(dv_acc[k].x[e], -65504.0f), 65504.0f));
        dK[plane + (size_t)r * HD + c] = __float2half(
            fminf(fmaxf(dk_acc[k].x[e], -65504.0f), 65504.0f));
      }
    }
  }
}

// ─── Dispatch ───────────────────────────────────────────────────────────────
//
// The flash path is selected by a process-global toggle *and* a shape gate. The
// old cuBLAS path stays compiled in behind both: it is the reference every A/B
// test compares against, and it is the escape hatch if something surfaces during
// a long run (LLM_ATTN_FLASH=0).
static int g_flash = -1;

extern "C" void cuda_attn_set_flash(int on) { g_flash = on ? 1 : 0; }

// On by default. `LLM_ATTN_FLASH=0` puts attention back on the cuBLAS path
// without a rebuild — the escape hatch if something surfaces mid-run.
static bool flash_enabled(void) {
  if (g_flash < 0) {
    const char *e = getenv("LLM_ATTN_FLASH");
    g_flash = (e && e[0] == '0') ? 0 : 1;
  }
  return g_flash != 0;
}

// head_dim must be one the kernels are instantiated for. These are the values
// d/h actually takes (both are powers of two in every config); anything else
// falls back rather than being approximated.
static bool flash_shape_ok(int hd) {
  return hd == 16 || hd == 32 || hd == 64 || hd == 128;
}

extern "C" int cuda_attn_flash_active(int hd) {
  return (flash_enabled() && flash_shape_ok(hd)) ? 1 : 0;
}

template <int HD>
static void flash_fwd_launch(const __half *Q, const __half *K, const __half *V,
                             __half *O, float *lse, float *ent, float inv_scale,
                             int seq, int h, int v_rows) {
  constexpr int BC = FLASH_BC(HD);
  constexpr size_t smem =
      (2 * BC * (HD + FLASH_PAD) + FLASH_BR * (BC + FLASH_PAD)) * sizeof(__half);
  static_assert(smem <= 48 * 1024,
                "flash forward tile exceeds the 48 KB static shared-memory "
                "limit; it would need an opt-in dynamic carveout");
  dim3 grid((seq + FLASH_BR - 1) / FLASH_BR, h);
  flash_fwd_kernel<HD, FLASH_BR, BC><<<grid, FLASH_THREADS, smem>>>(
      Q, K, V, O, lse, ent, inv_scale, seq, v_rows);
}

static void flash_forward(const __half *Q, const __half *K, const __half *V,
                          __half *O, float *lse, float *ent, float inv_scale,
                          int seq, int hd, int h, int v_rows) {
  flash_assert_acc_layout();
  switch (hd) {
    case 16: flash_fwd_launch<16>(Q, K, V, O, lse, ent, inv_scale, seq, h, v_rows); break;
    case 32: flash_fwd_launch<32>(Q, K, V, O, lse, ent, inv_scale, seq, h, v_rows); break;
    case 64: flash_fwd_launch<64>(Q, K, V, O, lse, ent, inv_scale, seq, h, v_rows); break;
    case 128: flash_fwd_launch<128>(Q, K, V, O, lse, ent, inv_scale, seq, h, v_rows); break;
    default: abort();  // flash_shape_ok gates this
  }
}

template <int HD>
static void flash_bwd_launch(const __half *Q, const __half *K, const __half *V,
                             const __half *dO, const float *lse, const float *D,
                             __half *dQ, __half *dK, __half *dV,
                             float inv_scale, int seq, int h) {
  constexpr int BC = FLASH_BC(HD);
  constexpr size_t smem_dq =
      (2 * BC * (HD + FLASH_PAD) + FLASH_BR * (BC + FLASH_PAD)) * sizeof(__half);
  constexpr size_t smem_dkv =
      (2 * BC * (HD + FLASH_PAD) + 2 * BC * (FLASH_BR + FLASH_PAD)) * sizeof(__half) +
      2 * FLASH_BR * sizeof(float);
  static_assert(smem_dq <= 48 * 1024, "flash dQ tile exceeds 48 KB");
  static_assert(smem_dkv <= 48 * 1024, "flash dK/dV tile exceeds 48 KB");

  const dim3 grid_q((seq + FLASH_BR - 1) / FLASH_BR, h);
  const dim3 grid_kv((seq + BC - 1) / BC, h);
  flash_bwd_dkv_kernel<HD, FLASH_BR, BC><<<grid_kv, FLASH_THREADS, smem_dkv>>>(
      Q, K, V, dO, lse, D, dK, dV, inv_scale, seq);
  flash_bwd_dq_kernel<HD, FLASH_BR, BC><<<grid_q, FLASH_THREADS, smem_dq>>>(
      Q, K, V, dO, lse, D, dQ, inv_scale, seq);
}

static void flash_backward(const __half *Q, const __half *K, const __half *V,
                           const __half *O, const __half *dO, const float *lse,
                           __half *dQ, __half *dK, __half *dV, float inv_scale,
                           int seq, int hd, int h) {
  flash_assert_acc_layout();

  // D = rowsum(dO ∘ O). One warp per row, four rows per block.
  {
    const int rows_per_block = FLASH_THREADS / 32;
    const dim3 grid((seq + rows_per_block - 1) / rows_per_block, h);
    flash_bwd_preprocess_kernel<<<grid, FLASH_THREADS>>>(dO, O, attn.dsum, seq, hd);
  }

  switch (hd) {
    case 16: flash_bwd_launch<16>(Q, K, V, dO, lse, attn.dsum, dQ, dK, dV, inv_scale, seq, h); break;
    case 32: flash_bwd_launch<32>(Q, K, V, dO, lse, attn.dsum, dQ, dK, dV, inv_scale, seq, h); break;
    case 64: flash_bwd_launch<64>(Q, K, V, dO, lse, attn.dsum, dQ, dK, dV, inv_scale, seq, h); break;
    case 128: flash_bwd_launch<128>(Q, K, V, dO, lse, attn.dsum, dQ, dK, dV, inv_scale, seq, h); break;
    default: abort();
  }
}

// The flash forward on its own, over pre-rotated Q/K/V — no projections, no
// RoPE, no cuBLAS. Exists so a test can compare the causal softmax against an
// independent f64 computation: every other check on these kernels is an A/B
// against the old path, and two implementations agreeing does not pin down the
// semantics they agree on. `Q`, `K`, `V` and `O` are `[h, seq, hd]` f16, and the
// caller owns the FLASH_BR row rounding.
extern "C" void cuda_flash_attention_forward(const __half *Q, const __half *K,
                                             const __half *V, __half *O,
                                             float *lse, int seq, int hd, int h) {
  if (!flash_shape_ok(hd)) abort();
  flash_forward(Q, K, V, O, lse, NULL, 1.0f / sqrtf((float)hd), seq, hd, h, seq);
}

// The flash backward on its own, for the same reason as above. `dsum` is the
// only piece of shared scratch it needs, so this sizes it directly rather than
// going through `ensure_attn` and dragging in the whole arena.
extern "C" void cuda_flash_attention_backward(
    const __half *Q, const __half *K, const __half *V, const __half *O,
    const __half *dO, const float *lse, __half *dQ, __half *dK, __half *dV,
    int seq, int hd, int h) {
  if (!flash_shape_ok(hd)) abort();
  const int rows = flash_round_rows(seq);
  ensure_attn(rows, hd, hd, h);
  flash_backward(Q, K, V, O, dO, lse, dQ, dK, dV, 1.0f / sqrtf((float)hd), seq,
                 hd, h);
}

// ─── Batched multi-head attention forward ───────────────────────────────────
// All heads processed in parallel via strided-batched GEMM. The four steps
// are reduced from (5 GEMMs + 3 kernels) × h heads down to 4 GEMMs + h
// element-wise kernel launches.
//
// Layouts (row-major):
//   X:        [seq, d]
//   wq_all:   [h, d, hd] contiguous                stride_d = d*hd
//   wk_all:   [h, d, hd]                            stride_d = d*hd
//   wv_all:   [h, d, hd]                            stride_d = d*hd
//   out_all:  [h, seq, hd]                          stride_c = seq*hd
//   P_all:    [h, seq, seq]                         stride_p = seq*seq
//   dP_all:   [h, seq, seq] (f32)                   stride_p = seq*seq

extern "C" void cuda_attention_forward_batched(
    const __half *X,
    const __half *wq_all, const __half *wk_all, const __half *wv_all,
    __half *out_all,
    // The [h, seq, seq] softmax plane. Written and read only by the legacy path;
    // the flash path never materializes it and the caller may pass null.
    __half *P_all,
    // Flash state the backward reads instead of `P_all`: the per-row logsumexp
    // [h, seq] f32, and an optional [h] per-head attention-entropy accumulator
    // for diagnostics (which the caller must zero first). Both null on an
    // inference prefill, which has no backward and collects no telemetry.
    float *lse_all, float *ent_all,
    // Per-layer Q/K/V caches the backward reads. See `Workspace::q_all` for why
    // these are owned per layer and not by the shared `attn` scratch.
    __half *Q_all, __half *K_all, __half *V_all,
    // Inference only, both ignored when `kv_stride` is 0 (which is what training
    // passes, leaving every stride below exactly as it was).
    //
    // `kv_stride` re-pitches V_all from `seq` positions to `kv_stride` of them so
    // that V lands directly in a KV cache it can be appended to later, and
    // `k_cache_f32`, when non-null, receives the post-RoPE f32 K at the same pitch.
    // A prefill therefore fills the cache as a side effect of running — which is
    // the whole reason the decode path needs no separate "warm the cache" pass.
    float *k_cache_f32, int kv_stride,
    int seq, int d, int hd, int h) {

  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  ensure_attn(seq, hd, d, h);

  const bool flash = cuda_attn_flash_active(hd) != 0;

  // V's row-block pitch, and the cache's. `seq` reproduces the pre-cache behaviour.
  int v_stride = (kv_stride > 0) ? kv_stride : seq;

  // The flash kernel reads Q and K as f16. Training already keeps per-layer f16
  // caches for the backward and hands them in; an inference prefill passes null
  // because it has no backward to feed, so it gets the shared scratch instead.
  __half *q16 = Q_all, *k16 = K_all;
  if (flash) {
    if (!q16) q16 = attn.q_f16;
    if (!k16) k16 = attn.k_f16;
  }

  // 1. Q = X @ Wq (batched f16×f16→f32)  [h, seq, hd]
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, d, &alpha,
      wq_all, CUDA_R_16F, hd, d*hd,
      X, CUDA_R_16F, d, 0,
      &beta,
      attn.q_f32, CUDA_R_32F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 2. K = X @ Wk (batched f16×f16→f32)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, d, &alpha,
      wk_all, CUDA_R_16F, hd, d*hd,
      X, CUDA_R_16F, d, 0,
      &beta,
      attn.k_f32, CUDA_R_32F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 3. V = X @ Wv (batched f16×f16→f16), straight into the per-layer cache —
  //    V is already f16, so caching it costs nothing at all.
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, d, &alpha,
      wv_all, CUDA_R_16F, hd, d*hd,
      X, CUDA_R_16F, d, 0,
      &beta,
      V_all, CUDA_R_16F, hd, v_stride*hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 3b. Apply RoPE to Q/K, then cache them as f16 for the backward.
  //
  //     Must land before the score GEMM: the whole point is that the scores are
  //     computed from rotated Q/K. And the *cached* values are the rotated ones
  //     on purpose — the backward's dQ' = dS @ K' and dK' = dS^T @ Q' need K'
  //     and Q', not the raw projections.
  {
    int n = h * seq * (hd / 2);
    rope_qk_kernel<<<(n + 255) / 256, 256>>>(attn.q_f32, attn.k_f32,
                                              q16, k16, seq, hd, h);
  }

  // The cached K is the *rotated* one, for the same reason the backward's f16 copy
  // is: a score is a rotated Q against a rotated K, so a decode step scoring
  // against raw projections would be applying no position at all to the history.
  if (k_cache_f32) {
    int n = h * seq * hd;
    cache_k_kernel<<<(n + 255) / 256, 256>>>(attn.k_f32, k_cache_f32,
                                             seq, hd, h, kv_stride);
  }

  const float inv_scale = 1.0f / sqrtf((float)hd);

  // 4-6, fused. One kernel, no [h, seq, seq] tensor, half the score tiles never
  // visited because the causal mask is structural here rather than elementwise.
  if (flash) {
    flash_forward(q16, k16, V_all, out_all, lse_all, ent_all, inv_scale, seq, hd,
                  h, v_stride);
    return;
  }

  // 4. Scores = Q @ K^T (batched f32×f32→f32)  [h, seq, seq]
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      seq, seq, hd, &alpha,
      attn.k_f32, CUDA_R_32F, hd, seq*hd,
      attn.q_f32, CUDA_R_32F, hd, seq*hd,
      &beta,
      attn.scores_f32, CUDA_R_32F, seq, seq*seq,
      h, COMPUTE_F32, TENSOR_OP));

  // 5. Per-head: fused scale + causal_mask + softmax (element-wise)
  {
    int t = 256;
    // Write the softmax straight into P_all. It used to land in the shared
    // scratch and then be copied here for the backward to use — h*seq*seq
    // halves per layer, ~12 MB a step at this shape, for nothing. The PV GEMM
    // below now reads P_all too, so the scratch copy has no remaining reader.
    // All heads in one launch via the y grid dimension.
    fused_scale_causal_softmax_kernel<<<dim3(seq, h), t, t * sizeof(float)>>>(
        attn.scores_f32, P_all, inv_scale, seq);
  }

  // 6. Out = softmax(S) @ V (batched f16×f16→f16)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, seq, &alpha,
      V_all, CUDA_R_16F, hd, v_stride*hd,
      P_all, CUDA_R_16F, seq, seq*seq,
      &beta,
      out_all, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP));
}

// ─── Batched multi-head attention backward ──────────────────────────────────
// Reverses the batched forward: 8 strided-batched GEMMs + per-head
// element-wise operations. Accumulates dX from all heads into a single buffer.

extern "C" void cuda_attention_backward_batched(
    const __half *X,                    // [seq, d] — saved LN1 input
    const __half *wq_all,               // [h, d, hd]
    const __half *wk_all,               // [h, d, hd]
    const __half *wv_all,               // [h, d, hd]
    const __half *P_all,                // [h, seq, seq] — legacy path only
    // What the flash path saved instead of `P_all`: the forward's output (which
    // the D = rowsum(dO ∘ O) preprocess needs) and its per-row logsumexp. Both
    // read only when flash is active; the legacy path ignores them.
    const __half *O_all,                // [h, seq, hd] — the forward's output
    const float *lse_all,               // [h, seq] f32
    // The forward's Q/K/V for THIS layer. Per-layer `Workspace` buffers, never
    // the shared `attn` scratch: that scratch is overwritten by every later
    // layer's forward, so reading Q/K/V from it differentiated layer L using
    // layer `layers-1`'s activations. Only the last layer was ever right, and
    // combined with two f32×f16 GEMMs cuBLAS rejects outright it left dK
    // identically zero for the life of the project. See `Workspace::q_all`.
    const __half *Q_all,                // [h, seq, hd]
    const __half *K_all,                // [h, seq, hd]
    const __half *V_all,                // [h, seq, hd]
    const __half *dout_full,            // [seq, d] — dWoX (full post-wo gradient)
    __half *dX,                         // [seq, d] — output gradient (accumulated)
    float *dwq_all,                     // [h, d, hd] f32
    float *dwk_all,                     // [h, d, hd] f32
    float *dwv_all,                     // [h, d, hd] f32
    int seq, int d, int hd, int h) {

  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  ensure_attn(seq, hd, d, h);

  // 0. Extract per-head dout from dout_full [seq, d] → contiguous [h, seq, hd]
  cuda_extract_strided_cols(dout_full, attn.dO, seq, d, hd, h, hd);

  if (cuda_attn_flash_active(hd)) {
    // 1-5, fused. dQ/dK/dV land in exactly the layout the surviving steps below
    // already read, so step 5b and everything after it is untouched by which
    // path produced them.
    const float inv_scale = 1.0f / sqrtf((float)hd);
    flash_backward(Q_all, K_all, V_all, O_all, attn.dO, lse_all, attn.dQ,
                   attn.dK, attn.dV, inv_scale, seq, hd, h);
  } else {

  // 1. dP = dout @ V^T (batched f16×f16^T→f32)  [h, seq, seq]
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      seq, seq, hd, &alpha,
      V_all, CUDA_R_16F, hd, seq*hd,
      attn.dO, CUDA_R_16F, hd, seq*hd,
      &beta,
      attn.dP, CUDA_R_32F, seq, seq*seq,
      h, COMPUTE_F32, TENSOR_OP));

  // 2. Per-head: softmax backward + scale + causal_mask (element-wise on f32)
  {
    float rcp = sqrtf((float)hd);
    // One launch each instead of three per head. The heads are contiguous
    // planes of seq*seq, so the softmax backward and the mask take the head
    // from a grid dimension, and the scale is a flat pass over all of them.
    softmax_bw_f32_dP_kernel<<<dim3(seq, h), 256, 256 * sizeof(float)>>>(
        P_all, attn.dP, attn.dS, seq, seq);
    scale_kernel_f16<<<((h * seq * seq) + 255) / 256, 256>>>(
        attn.dS, rcp, h * seq * seq);
    {
      dim3 t16(16, 16);
      causal_mask_bw_kernel<<<dim3((seq + 15) / 16, (seq + 15) / 16, h), t16>>>(
          attn.dS, seq);
    }
  }

  // 3. dV = P^T @ dout (batched f16×f16→f16)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, seq, seq, &alpha,
      attn.dO, CUDA_R_16F, hd, seq*hd,
      P_all, CUDA_R_16F, seq, seq*seq,
      &beta,
      attn.dV, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 4. dQ = dS @ K (batched f16×f16→f16)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, seq, &alpha,
      K_all, CUDA_R_16F, hd, seq*hd,
      attn.dS, CUDA_R_16F, seq, seq*seq,
      &beta,
      attn.dQ, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 5. dK = dS^T @ Q (batched f16×f16→f16)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, seq, seq, &alpha,
      Q_all, CUDA_R_16F, hd, seq*hd,
      attn.dS, CUDA_R_16F, seq, seq*seq,
      &beta,
      attn.dK, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP));

  }  // end legacy 1-5

  // 5b. Undo RoPE on dQ/dK.
  //
  //     Steps 4 and 5 produced gradients w.r.t. the *rotated* Q/K. Both of the
  //     consumers below want them w.r.t. the unrotated projections — step 6 for
  //     dX, steps 7/8 for dWq/dWk — so one placement here covers all of them.
  {
    int n = h * seq * (hd / 2);
    rope_dqk_kernel<<<(n + 255) / 256, 256>>>(attn.dQ, attn.dK, seq, hd, h);
  }

  // 6. dX = sum_i( dQ_i @ wq_i^T + dK_i @ wk_i^T + dV_i @ wv_i^T )
  //    First head writes (β=0), rest accumulate (β=1)
  //
  // This looks like the obvious remaining un-batched loop — 3*h separate GEMMs,
  // 24 launches per layer at h=8 — and it was rewritten to batch them and
  // measured. **It was slower**, consistently, by ~1 ms/step (2.3%) at
  // d512/h8/L8/ctx1024, over three runs each way.
  //
  // The reason a batch does not fit: a strided-batched GEMM gives every batch
  // element its own output, and these all want the same one, so batching means
  // writing `h` planes of an [h, seq, d] temp and then reducing them. That
  // reduction moves ~8 MB per term per layer that this form never touches, and
  // it costs more than the 18 saved launches save. The launches also overlap
  // with GPU work here; they are not the bottleneck they look like.
  //
  // Leave it alone unless you have a measurement that says otherwise.
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d, seq, hd, &alpha,
      wq_all, CUDA_R_16F, hd,
      attn.dQ, CUDA_R_16F, hd,
      &beta,
      dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP));

  for (int hi = 1; hi < h; hi++) {
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        d, seq, hd, &alpha,
        wq_all + hi * d * hd, CUDA_R_16F, hd,
        attn.dQ + hi * seq * hd, CUDA_R_16F, hd,
        &alpha, dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP));
  }

  // dX += dK @ wk^T
  for (int hi = 0; hi < h; hi++) {
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        d, seq, hd, &alpha,
        wk_all + hi * d * hd, CUDA_R_16F, hd,
        attn.dK + hi * seq * hd, CUDA_R_16F, hd,
        &alpha, dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP));
  }

  // dX += dV @ wv^T
  for (int hi = 0; hi < h; hi++) {
    CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        d, seq, hd, &alpha,
        wv_all + hi * d * hd, CUDA_R_16F, hd,
        attn.dV + hi * seq * hd, CUDA_R_16F, hd,
        &alpha, dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP));
  }

  clamp_f16_kernel<<<(seq * d + 255) / 256, 256>>>(dX, seq * d);

  float one = 1.0f;
  // 7. Weight gradients: dwq = X^T @ dQ (batched f16×f16→f32, accumulate β=1)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, d, seq, &alpha,
      attn.dQ, CUDA_R_16F, hd, seq*hd,
      X, CUDA_R_16F, d, 0,
      &one,
      dwq_all, CUDA_R_32F, hd, d*hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 8. dwk = X^T @ dK (batched, accumulate β=1)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, d, seq, &alpha,
      attn.dK, CUDA_R_16F, hd, seq*hd,
      X, CUDA_R_16F, d, 0,
      &one,
      dwk_all, CUDA_R_32F, hd, d*hd,
      h, COMPUTE_F32, TENSOR_OP));

  // 9. dwv = X^T @ dV (batched, accumulate β=1)
  CUBLAS_CHECK(cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, d, seq, &alpha,
      attn.dV, CUDA_R_16F, hd, seq*hd,
      X, CUDA_R_16F, d, 0,
      &one,
      dwv_all, CUDA_R_32F, hd, d*hd,
      h, COMPUTE_F32, TENSOR_OP));
}

// ─── dW = A^T @ dB (f16×f16→f32, for backward weight gradients) ──────────
// A: [rows × cols_a] f16 row-major, dB: [rows × cols_b] f16 row-major
// dW: [cols_a × cols_b] f32 row-major
extern "C" void cuda_d_weight(const __half *A, const __half *dB, float *dW,
                               int rows, int cols_a, int cols_b) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      cols_b, cols_a, rows, &alpha,
      dB, CUDA_R_16F, cols_b, A, CUDA_R_16F, cols_a, &beta,
      dW, CUDA_R_32F, cols_b, COMPUTE_F32, TENSOR_OP));
}

// ─── cuBLAS wrappers for matmul and matmul_bt ──────────────────────────────

extern "C" void cuda_matmul(const __half *a, const __half *b, __half *result,
                             int rows_a, int cols_a, int cols_b) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      cols_b, rows_a, cols_a, &alpha,
      b, CUDA_R_16F, cols_b, a, CUDA_R_16F, cols_a, &beta,
      result, CUDA_R_16F, cols_b, COMPUTE_F32, TENSOR_OP));
}

extern "C" void cuda_matmul_bt(const __half *a, const __half *b, __half *result,
                                int rows_a, int cols_a, int rows_b) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      rows_b, rows_a, cols_a, &alpha,
      b, CUDA_R_16F, cols_a, a, CUDA_R_16F, cols_a, &beta,
      result, CUDA_R_16F, rows_b, COMPUTE_F32, TENSOR_OP));
}

// ─── Feed-forward forward (inference, fused bias+gelu) ─────────────────────

extern "C" void cuda_feed_forward(const __half *input,
                                   const __half *w1,
                                   const __half *w2,
                                   const __half *b1,
                                   const __half *b2,
                                   __half *out,
                                   int seq_len, int d_model, int d_ff) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;

  ensure_ff(seq_len, d_ff);
  __half *d_hidden = ff.hidden;

  // hidden = input @ w1 (f16×f16→f16)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_ff, seq_len, d_model, &alpha,
      w1, CUDA_R_16F, d_ff, input, CUDA_R_16F, d_model, &beta,
      d_hidden, CUDA_R_16F, d_ff, COMPUTE_F32, TENSOR_OP));

  // Fused bias_add + GELU (single kernel)
  fused_bias_gelu_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(
      d_hidden, b1, d_ff, seq_len * d_ff);

  // out = hidden @ w2 (f16×f16→f16)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_model, seq_len, d_ff, &alpha,
      w2, CUDA_R_16F, d_model, d_hidden, CUDA_R_16F, d_ff, &beta,
      out, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP));

  // Add b2 bias (no GELU after second projection)
  {
    dim3 t16(16, 16);
    bias_add_kernel<<<dim3((d_model + 15) / 16, (seq_len + 15) / 16), t16>>>(
        out, b2, seq_len, d_model);
  }
}

// ─── Feed-forward forward (training, saves pre-gelu for backward) ──────────

extern "C" void
cuda_feed_forward_forward(const __half *input,
                           const __half *w1,
                           const __half *w2,
                           const __half *b1,
                           const __half *b2,
                           __half *out,
                           __half *pre_gelu,
                           int seq_len, int d_model, int d_ff) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;

  ensure_ff(seq_len, d_ff);
  __half *d_hidden = ff.hidden;

  // hidden = input @ w1 (f16×f16→f16)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_ff, seq_len, d_model, &alpha,
      w1, CUDA_R_16F, d_ff, input, CUDA_R_16F, d_model, &beta,
      d_hidden, CUDA_R_16F, d_ff, COMPUTE_F32, TENSOR_OP));

  // Add bias first: pre_gelu = input @ w1 + b1
  {
    dim3 t16(16, 16);
    bias_add_kernel<<<dim3((d_ff + 15) / 16, (seq_len + 15) / 16), t16>>>(
        d_hidden, b1, seq_len, d_ff);
  }

  // Save pre-GELU activations (now WITH bias b1 added)
  cudaMemcpy(pre_gelu, d_hidden, seq_len * d_ff * sizeof(__half),
             cudaMemcpyDeviceToDevice);

  // Run GELU in-place on d_hidden
  gelu_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(
      d_hidden, seq_len * d_ff);

  // out = hidden @ w2 (f16×f16→f16)
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_model, seq_len, d_ff, &alpha,
      w2, CUDA_R_16F, d_model, d_hidden, CUDA_R_16F, d_ff, &beta,
      out, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP));

  {
    dim3 t16(16, 16);
    bias_add_kernel<<<dim3((d_model + 15) / 16, (seq_len + 15) / 16), t16>>>(
        out, b2, seq_len, d_model);
  }
}

// ─── Feed-forward backward (unchanged, but pre_gelu now holds pre-activation) ─

extern "C" void cuda_feed_forward_backward(
    const __half *input,
    const __half *pre_gelu,
    const __half *dout,
    const __half *w1,
    const __half *w2,
    __half *dinput,
    float *dw1,
    float *dw2,
    float *db1,
    float *db2,
    int seq_len, int d_model, int d_ff) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;

  ensure_ff(seq_len, d_ff);
  __half *d_hidden_gelu = ff.hidden;

  // d_hidden_gelu = dout @ w2^T (f16×f16^T→f16) [seq_len x d_ff]
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_ff, seq_len, d_model, &alpha,
      w2, CUDA_R_16F, d_model, dout, CUDA_R_16F, d_model, &beta,
      d_hidden_gelu, CUDA_R_16F, d_ff, COMPUTE_F32, TENSOR_OP));

  // Compute activated = gelu(pre_gelu) dynamically to use for dw2 gradient.
  // Distinct from d_hidden_gelu above: both are live across the dw2 GEMM.
  __half *activated = ff.activated;
  cudaMemcpy(activated, pre_gelu, seq_len * d_ff * sizeof(__half), cudaMemcpyDeviceToDevice);
  gelu_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(activated, seq_len * d_ff);

  // dw2 = activated^T @ dout (f16×f16→f32) [d_ff x d_model]
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      d_model, d_ff, seq_len, &alpha,
      dout, CUDA_R_16F, d_model, activated, CUDA_R_16F, d_ff, &beta,
      dw2, CUDA_R_32F, d_model, COMPUTE_F32, TENSOR_OP));

  cuda_bias_grad(dout, db2, seq_len, d_model);

  // gelu backward on d_hidden_gelu in-place
  gelu_bw_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(pre_gelu, d_hidden_gelu,
                                                          d_hidden_gelu,
                                                          seq_len * d_ff);

  // dw1 = input^T @ d_hidden_gelu (f16×f16→f32) [d_model x d_ff]
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      d_ff, d_model, seq_len, &alpha,
      d_hidden_gelu, CUDA_R_16F, d_ff, input, CUDA_R_16F, d_model, &beta,
      dw1, CUDA_R_32F, d_ff, COMPUTE_F32, TENSOR_OP));

  cuda_bias_grad(d_hidden_gelu, db1, seq_len, d_ff);

  // dinput = d_hidden_gelu @ w1^T (f16×f16^T→f16) [seq_len x d_model]
  CUBLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_model, seq_len, d_ff, &alpha,
      w1, CUDA_R_16F, d_ff, d_hidden_gelu, CUDA_R_16F, d_ff, &beta,
      dinput, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP));
}

// ─── Head concatenation kernel ──────────────────────────────────────────────

__global__ void concat_heads_kernel(__half **head_ptrs, __half *out,
                                    int seq_len, int heads, int head_dim) {
  int hi = blockIdx.z;
  int t = blockIdx.y * blockDim.y + threadIdx.y;
  int j = blockIdx.x * blockDim.x + threadIdx.x;

  if (hi < heads && t < seq_len && j < head_dim) {
    out[t * (heads * head_dim) + hi * head_dim + j] =
        head_ptrs[hi][t * head_dim + j];
  }
}

extern "C" void cuda_concat_heads(__half **head_ptrs, __half *out,
                                  int seq_len, int heads, int head_dim) {
  dim3 block(16, 16, 1);
  dim3 grid((head_dim + 15) / 16, (seq_len + 15) / 16, heads);
  concat_heads_kernel<<<grid, block>>>(head_ptrs, out, seq_len, heads, head_dim);
}

// ─── Async global gradient clip (no host synchronization) ──────────────────
// The clip used to run one `cublasSnrm2` per gradient buffer — 62 of them at 6
// layers — accumulate the squares host-side, then scale each buffer. snrm2 in
// host-pointer mode blocks until the device drains, so one optimizer step cost
// 62 full pipeline stalls, and with accumulation every 4 steps that was ~15
// stalls per training step.
//
// The arithmetic here is identical; it just never leaves the device. Sum of
// squares accumulates across buffers into one f32, a one-thread kernel turns
// that into a norm and a clip scale, and each buffer is scaled by that
// device-resident scalar. Every call is asynchronous, so the queue stays full.
// The host reads the norm only when it actually wants to print it.
//
// `state` is 3 floats: [0] sum-of-squares accumulator, [1] resulting global
// norm, [2] scale to apply. It must start zeroed (`Buffer::alloc_zeroed`);
// `clip_finalize_kernel` re-zeros [0] for the next window.

__global__ void grad_sumsq_kernel(const float *__restrict__ v, int n, float *accum) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 0.0f;
  if (i < n) {
    float x = v[i];
    val = x * x;
  }
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(accum, sdata[0]);
}

extern "C" void cuda_grad_sumsq_accum(const float *v, int n, float *accum) {
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  grad_sumsq_kernel<<<blocks, threads, threads * sizeof(float)>>>(v, n, accum);
}

// Matches the old host-side policy exactly: a non-finite norm scales the whole
// gradient to zero (dropping a NaN/Inf window rather than stepping on it), a
// norm above `max_norm` scales down to exactly `max_norm`, anything else passes
// through untouched.
__global__ void clip_finalize_kernel(float *state, float max_norm) {
  float total = sqrtf(state[0]);
  state[0] = 0.0f;  // reset the accumulator for the next optimizer step
  state[1] = total;
  float scale = 1.0f;
  if (!isfinite(total)) {
    scale = 0.0f;
  } else if (total > max_norm) {
    scale = max_norm / total;
  }
  state[2] = scale;
}

extern "C" void cuda_clip_finalize(float *state, float max_norm) {
  clip_finalize_kernel<<<1, 1>>>(state, max_norm);
}

// Scale by a factor that lives in device memory, so applying the clip needs no
// readback of the norm.
__global__ void scale_by_dev_kernel(float *v, const float *scale, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] *= *scale;
}

extern "C" void cuda_scale_by_dev(float *v, const float *scale, int n) {
  int threads = 256;
  scale_by_dev_kernel<<<(n + threads - 1) / threads, threads>>>(v, scale, n);
}

// ─── Gradient L2-norm clipping ──────────────────────────────────────────────

// Per-buffer gradient clip, split reduce-then-apply so both halves span the
// whole device.
//
// This was a single-block kernel — `<<<1, 256>>>` — reducing an entire gradient
// tensor on one SM out of 28, and `Model::clip_gradients` calls it once per
// gradient buffer, 62 times per step at 6 layers, the largest of them 2.1M
// elements. After the layer-norm backward was fixed this became the single
// biggest remaining stage in the step profile.
//
// Semantics are unchanged: non-finite elements are zeroed in place *before*
// they contribute to the sum, and the buffer is rescaled only when its norm
// exceeds `max_norm`. Because the sanitize pass runs first, the sum is always
// finite and needs no separate non-finite guard.
//
// `scratch` is a single f32 accumulator zeroed by the wrapper. One shared
// scratch is safe across the 62 calls because they are issued on the same
// stream and therefore execute in order.

__global__ void clip_sanitize_sumsq_kernel(float *g, int n, float *scratch) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 0.0f;
  if (i < n) {
    float v = g[i];
    if (isnan(v) || isinf(v)) v = 0.0f;
    g[i] = v;
    val = v * v;
  }
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(scratch, sdata[0]);
}

// Each thread re-derives the scale from the accumulator: a couple of register
// ops, cheaper than a third kernel launch to compute it once.
__global__ void clip_apply_kernel(float *g, int n, const float *scratch,
                                   float max_norm) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float norm = sqrtf(*scratch);
  float scale = (norm > max_norm) ? (max_norm / norm) : 1.0f;
  g[i] *= scale;
}

// Batched per-buffer clip: every gradient tensor clipped in two launches.
//
// `Model::clip_gradients` clips each tensor independently, and calling
// cuda_clip_norm once per tensor meant 61 buffers x 3 launches = ~183 launches
// per step doing almost no work each. At ~12 us of launch cost apiece (WSL2's
// paravirtualized driver is dearer than native) that stage measured 2.26 ms and
// was entirely launch-bound rather than compute-bound.
//
// Here the buffer list lives on the device (pointers + sizes, built once since
// the workspace allocations never move) and the buffer index comes from the y
// grid dimension. The x dimension is a fixed, modest grid that strides through
// whatever the buffer's length happens to be, so one launch covers tensors from
// 256 elements to 2.1M without sizing the grid to the largest.
//
// Semantics match cuda_clip_norm exactly: non-finite elements zeroed in place
// before they are summed, then a rescale only if that buffer's norm exceeds
// max_norm. `norms` is one f32 per buffer, zeroed by the wrapper.

__global__ void clip_batched_sumsq_kernel(float **ptrs, const int *sizes,
                                           float *norms) {
  int b = blockIdx.y;
  float *g = ptrs[b];
  int n = sizes[b];
  extern __shared__ float sdata[];
  int tid = threadIdx.x;

  float sum = 0.0f;
  for (int i = blockIdx.x * blockDim.x + tid; i < n;
       i += blockDim.x * gridDim.x) {
    float v = g[i];
    if (isnan(v) || isinf(v)) v = 0.0f;
    g[i] = v;
    sum += v * v;
  }
  sdata[tid] = sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(&norms[b], sdata[0]);
}

__global__ void clip_batched_apply_kernel(float **ptrs, const int *sizes,
                                           const float *norms, float max_norm) {
  int b = blockIdx.y;
  float *g = ptrs[b];
  int n = sizes[b];
  float norm = sqrtf(norms[b]);
  float scale = (norm > max_norm) ? (max_norm / norm) : 1.0f;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n;
       i += blockDim.x * gridDim.x)
    g[i] *= scale;
}

extern "C" void cuda_clip_norm_batched(float **d_ptrs, const int *d_sizes,
                                        float *d_norms, int n_buffers,
                                        float max_norm) {
  cudaMemsetAsync(d_norms, 0, n_buffers * sizeof(float));
  int threads = 256;
  int gx = 256;  // strided; independent of the largest buffer
  dim3 grid(gx, n_buffers);
  clip_batched_sumsq_kernel<<<grid, threads, threads * sizeof(float)>>>(
      d_ptrs, d_sizes, d_norms);
  clip_batched_apply_kernel<<<grid, threads>>>(d_ptrs, d_sizes, d_norms,
                                                max_norm);
}

extern "C" void cuda_clip_norm(float *g, int n, float max_norm) {
  if (!clip_scratch) cudaMalloc(&clip_scratch, sizeof(float));
  cudaMemsetAsync(clip_scratch, 0, sizeof(float));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  clip_sanitize_sumsq_kernel<<<blocks, threads, threads * sizeof(float)>>>(
      g, n, clip_scratch);
  clip_apply_kernel<<<blocks, threads>>>(g, n, clip_scratch, max_norm);
}

// ─── Debug: check buffer for NaN/Inf ───────────────────────────────────────
__global__ void check_nan_kernel(const __half *buf, int n, int *flag) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float v = __half2float(buf[i]);
  if (!isfinite(v) && flag[0] == 0) {
    flag[0] = 1;
    flag[1] = i;
  }
}

extern "C" int cuda_check_nan(const __half *buf, int n, int *out_pos) {
  int *d_flag;
  cudaMalloc(&d_flag, 2 * sizeof(int));
  cudaMemset(d_flag, 0, 2 * sizeof(int));
  check_nan_kernel<<<(n + 255) / 256, 256>>>(buf, n, d_flag);
  int flag[2];
  cudaMemcpy(flag, d_flag, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_flag);
  if (out_pos) *out_pos = flag[1];
  return flag[0];
}

// f32 variant
__global__ void check_nan_f32_kernel(const float *buf, int n, int *flag) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  if (!isfinite(buf[i]) && flag[0] == 0) {
    flag[0] = 1;
    flag[1] = i;
  }
}

extern "C" int cuda_check_nan_f32(const float *buf, int n, int *out_pos) {
  int *d_flag;
  cudaMalloc(&d_flag, 2 * sizeof(int));
  cudaMemset(d_flag, 0, 2 * sizeof(int));
  check_nan_f32_kernel<<<(n + 255) / 256, 256>>>(buf, n, d_flag);
  int flag[2];
  cudaMemcpy(flag, d_flag, 2 * sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_flag);
  if (out_pos) *out_pos = flag[1];
  return flag[0];
}

// ─── GPU-side SGD with momentum: v = mu*v + g; w = f16(f32(w) - lr*v) ──

extern "C" void cuda_sgd_momentum(const float *g, __half *w, float *v,
                                   float lr, float mu, int n) {
  sgd_momentum_kernel<<<(n + 255) / 256, 256>>>(g, w, v, lr, mu, n);
}

extern "C" void cuda_adam(const float *g, float *w_master, __half *w_view,
                          float *m, float *v, float grad_scale, float lr,
                          float beta1, float beta2, float eps,
                          float weight_decay, float bc1, float bc2, int n) {
  adam_kernel<<<(n + 255) / 256, 256>>>(g, w_master, w_view, m, v, grad_scale,
                                        lr, beta1, beta2, eps, weight_decay,
                                        bc1, bc2, n);
}

// ─── Embedding forward: lookup + positional add ─────────────────────
// A plain gather: position no longer enters here. It used to add a row of the
// learned absolute positional table, which RoPE replaced — see the RoPE section
// at the top of this file.
__global__ void embedding_forward_kernel(
  const int *token_ids,
  const __half *embedding_table,
  __half *output,
  int s, int d, int vocab
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s * d) return;
  int pos = i / d;
  int j = i % d;
  int token = token_ids[pos];
  output[i] = embedding_table[token * d + j];
}

extern "C" void cuda_embedding_forward(
  const int *token_ids,
  const __half *embedding_table,
  __half *output,
  int s, int d, int vocab
) {
  embedding_forward_kernel<<<(s * d + 255) / 256, 256>>>(token_ids, embedding_table, output, s, d, vocab);
}

// ─── Embedding backward: scatter-add d_dx into g_emb ────────────────
__global__ void scatter_add_embedding_grad_kernel(
  const __half *d_dx,
  const int *token_ids,
  float *g_emb,
  int s, int d, int vocab
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s * d) return;
  int pos = i / d;
  int j = i % d;
  int token = token_ids[pos];
  atomicAdd(&g_emb[token * d + j], __half2float(d_dx[i]));
}

extern "C" void cuda_scatter_add_embedding_grad(
  const __half *d_dx,
  const int *token_ids,
  float *g_emb,
  int s, int d, int vocab
) {
  scatter_add_embedding_grad_kernel<<<(s * d + 255) / 256, 256>>>(d_dx, token_ids, g_emb, s, d, vocab);
}

// ─── Training-telemetry reductions (read-only diagnostics) ──────────────────
// These are sampled off the hot path (once per progress report, ~1% of steps)
// to feed the training dashboard: activation RMS, update:weight ratios, and
// attention entropy. They only READ workspace buffers — never mutate training
// state — so they can never perturb the numerics of the run they observe.

// Block-reduced sum of squares of an f16 buffer, returned host-side (synchronous
// via the final cudaMemcpy). The caller turns this into an L2 norm (sqrt) or an
// RMS (sqrt(sumsq / n)); doing the reduction on-device keeps a large activation
// tensor off the host. `out` is a single f32 accumulator, zeroed before launch.
__global__ void sumsq_f16_kernel(const __half *__restrict__ v, int n, float *out) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 0.0f;
  if (i < n) {
    float x = __half2float(v[i]);
    val = x * x;
  }
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

extern "C" float cuda_sumsq_f16(const __half *v, int n) {
  float *d_out;
  cudaMalloc(&d_out, sizeof(float));
  cudaMemset(d_out, 0, sizeof(float));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  sumsq_f16_kernel<<<blocks, threads, threads * sizeof(float)>>>(v, n, d_out);
  float out = 0.0f;
  cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return out;
}

// Sum of squares of the AdamW *step* vector  step_i = lr * (m_hat/(sqrt(v_hat)+eps)),
// using the CURRENT moment buffers (m = 1st moment, v = 2nd moment). This mirrors
// `adam_kernel`'s adaptive term but EXCLUDES the decoupled weight-decay shrink and
// writes nothing — it is the applied-update magnitude the update:weight telemetry
// tracks. `bc1`/`bc2` are the bias-correction denominators (1 - beta^t) the caller
// computes host-side, exactly as the real Adam step does.
__global__ void adam_step_sumsq_kernel(const float *__restrict__ m,
                                       const float *__restrict__ v, float lr,
                                       float eps, float bc1, float bc2, int n,
                                       float *out) {
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float val = 0.0f;
  if (i < n) {
    float m_hat = m[i] / bc1;
    float v_hat = v[i] / bc2;
    float step = lr * (m_hat / (sqrtf(v_hat) + eps));
    val = step * step;
  }
  sdata[tid] = val;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  if (tid == 0) atomicAdd(out, sdata[0]);
}

extern "C" float cuda_adam_step_sumsq(const float *m, const float *v, float lr,
                                      float eps, float bc1, float bc2, int n) {
  float *d_out;
  cudaMalloc(&d_out, sizeof(float));
  cudaMemset(d_out, 0, sizeof(float));
  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  adam_step_sumsq_kernel<<<blocks, threads, threads * sizeof(float)>>>(
      m, v, lr, eps, bc1, bc2, n, d_out);
  float out = 0.0f;
  cudaMemcpy(&out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return out;
}

// Per-head mean attention entropy, normalized to [0,1]. `P` is the saved causal
// softmax [h, seq, seq]: query row r sums over key cols 0..=r. For each row r>=1
// the Shannon entropy -Σ p·ln p is divided by ln(r+1) — the entropy of a uniform
// distribution over its r+1 valid keys — so the value is scale-free across query
// positions (1 = perfectly diffuse attention, →0 = a sharp single-key spike). Row
// 0 has a single key (entropy 0, undefined normalizer) and is skipped. The kernel
// accumulates the normalized per-row entropies into out[head]; the caller divides
// by the row count (seq-1) to get the head's mean. `out` holds h floats, zeroed
// here. One block per head, one thread per query row — seq <= ctx_window, well
// within a block's thread limit for the small contexts this trainer uses.
__global__ void attention_entropy_kernel(const __half *__restrict__ P, int seq,
                                         int h, float *out) {
  int head = blockIdx.x;
  int row = threadIdx.x + 1;  // rows 1..seq-1 (row 0 has a single key)
  if (head >= h || row >= seq) return;
  const __half *prow = P + (size_t)head * seq * seq + (size_t)row * seq;
  float ent = 0.0f;
  for (int c = 0; c <= row; c++) {
    float p = __half2float(prow[c]);
    if (p > 1e-9f) ent -= p * logf(p);
  }
  float norm = logf((float)(row + 1));
  if (norm > 0.0f) atomicAdd(&out[head], ent / norm);
}

extern "C" void cuda_attention_entropy(const __half *P, int seq, int h,
                                       float *out_dev) {
  cudaMemset(out_dev, 0, h * sizeof(float));
  int threads = seq > 1 ? seq - 1 : 1;
  attention_entropy_kernel<<<h, threads>>>(P, seq, h, out_dev);
}
