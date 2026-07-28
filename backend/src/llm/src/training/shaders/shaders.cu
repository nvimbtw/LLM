// gork: transformer training CUDA kernels (f16 mixed precision + cuBLAS tensor cores)
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <math.h>

// ─── cuBLAS handle (lazily initialized) ─────────────────────────────────────
static cublasHandle_t g_cublas = NULL;
static bool g_cublas_init = false;

extern "C" void cublas_init() {
    if (!g_cublas_init) {
        cublasCreate(&g_cublas);
        g_cublas_init = true;
    }
}

// ─── Pre-allocated attention scratch buffers ───────────────────────────────
// Lazily allocated, grown to max(seq, hd, d) seen. Shared across all
// attention functions (forward/backward are sequential, not concurrent).
static struct {
    // Forward: Q/K in f32, V in f16, scores f32+f16
    float *q_f32, *k_f32;
    __half *v_f16;
    float *scores_f32;
    __half *scores_f16;
    // Backward: Q/K/V in f16, dQ/dK/dV, dP f32, dS f16, dX_temp
    __half *Q, *K, *V, *dQ, *dK, *dV;
    float *dP;
    __half *dS, *dX_temp;
    int seq, hd, d, h;
} attn = {NULL};

static void ensure_attn_cleanup(void) {
    auto f = [](auto &p) { if (p) { cudaFree(p); p = NULL; } };
    f(attn.q_f32); f(attn.k_f32); f(attn.v_f16);
    f(attn.scores_f32); f(attn.scores_f16);
    f(attn.Q); f(attn.K); f(attn.V);
    f(attn.dQ); f(attn.dK); f(attn.dV);
    f(attn.dP); f(attn.dS); f(attn.dX_temp);
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
    cublasSnrm2(get_cublas(), n, v, 1, &result);
    return result;
}

// Scale a gradient buffer in place by `s`.
extern "C" void cuda_scale(float *v, float s, int n) {
    cublasSscal(get_cublas(), n, &s, v, 1);
}

#define TENSOR_OP CUBLAS_GEMM_DEFAULT_TENSOR_OP
#define COMPUTE_F32 CUBLAS_COMPUTE_32F

static void ensure_attn(int seq, int hd, int d, int h) {
    if (seq <= attn.seq && hd <= attn.hd && d <= attn.d && h <= attn.h && attn.q_f32) return;
    ensure_attn_cleanup();

    auto a = [](auto &p, size_t sz) { cudaMalloc(&p, sz); };
    a(attn.q_f32,      h * seq * hd * sizeof(float));
    a(attn.k_f32,      h * seq * hd * sizeof(float));
    a(attn.v_f16,      h * seq * hd * sizeof(__half));
    a(attn.scores_f32, h * seq * seq * sizeof(float));
    a(attn.scores_f16, h * seq * seq * sizeof(__half));
    a(attn.Q,          h * seq * hd * sizeof(__half));
    a(attn.K,          h * seq * hd * sizeof(__half));
    a(attn.V,          h * seq * hd * sizeof(__half));
    a(attn.dQ,         h * seq * hd * sizeof(__half));
    a(attn.dK,         h * seq * hd * sizeof(__half));
    a(attn.dV,         h * seq * hd * sizeof(__half));
    a(attn.dP,         h * seq * seq * sizeof(float));
    a(attn.dS,         h * seq * seq * sizeof(__half));
    a(attn.dX_temp,    h * seq * d  * sizeof(__half));
    attn.seq = seq; attn.hd = hd; attn.d = d; attn.h = h;
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

// ─── softmax backward ──────────────────────────────────────────────────────

// One BLOCK per row — see softmax_bw_f32_dP_kernel above for the rationale.
// Shared memory: blockDim.x floats.
__global__ void softmax_bw_kernel(const __half *P, const __half *dP, __half *dS,
                                   int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows)
    return;
  extern __shared__ float sdata[];
  int tid = threadIdx.x;
  const __half *P_row = P + (size_t)row * cols;
  const __half *dP_row = dP + (size_t)row * cols;
  __half *dS_row = dS + (size_t)row * cols;

  float partial = 0.0f;
  for (int i = tid; i < cols; i += blockDim.x)
    partial += __half2float(P_row[i]) * __half2float(dP_row[i]);
  sdata[tid] = partial;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();
  }
  float dot = sdata[0];

  for (int i = tid; i < cols; i += blockDim.x)
    dS_row[i] = __float2half(__half2float(P_row[i]) * (__half2float(dP_row[i]) - dot));
}

extern "C" void cuda_softmax_backward(const __half *P, const __half *dP,
                                       __half *dS, int rows, int cols) {
  int threads = 256;
  softmax_bw_kernel<<<rows, threads, threads * sizeof(float)>>>(P, dP, dS, rows, cols);
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

extern "C" void cuda_causal_mask_backward(__half *dS, int seq_len) {
  dim3 t(16, 16);
  causal_mask_bw_kernel<<<dim3((seq_len + 15) / 16, (seq_len + 15) / 16), t>>>(
      dS, seq_len);
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

// Copy hd columns per row from a strided row-major source (s x src_cols) into
// a contiguous row-major destination (s x hd). Used to extract per-head dout
// slices from the full d_post_attn @ wo result.
__global__ void extract_cols_kernel(const __half *src, __half *dst,
                                     int rows, int src_cols, int dst_cols,
                                     int col_offset) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows) return;
  for (int c = 0; c < dst_cols; c++)
    dst[row * dst_cols + c] = src[row * src_cols + col_offset + c];
}

extern "C" void cuda_extract_cols(const __half *src, __half *dst,
                                   int rows, int src_cols, int dst_cols,
                                   int col_offset) {
  extract_cols_kernel<<<(rows + 255) / 256, 256>>>(src, dst, rows, src_cols, dst_cols, col_offset);
}

// ─── forward declarations for kernels defined below ─────────────────────────
// (needed because backward is defined before some element-wise kernels)
__global__ void softmax_bw_f32_dP_kernel(const __half *P, const float *dP,
                                          __half *dS, int rows, int cols);
__global__ void scale_kernel_f16(__half *x, float scale, int n);
__global__ void clamp_f16_kernel(__half *buf, int n);
__global__ void causal_mask_bw_kernel(__half *dS, int seq_len);

// ─── attention head backward (per-head, uses pre-allocated scratch) ─────────

extern "C" void
cuda_attention_head_backward(const __half *X,
                              const __half *wq,
                              const __half *wk,
                              const __half *wv,
                              const __half *P,
                              const __half *dout,
                              __half *dX,
                              float *dwq,
                              float *dwk,
                              float *dwv,
                              int seq_len, int d_model, int head_dim) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  ensure_attn(seq_len, head_dim, d_model, 1);

  // Recompute Q, K, V = X @ Wq, X @ Wk, X @ Wv (f16×f16→f16)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wq, CUDA_R_16F, head_dim, X, CUDA_R_16F, d_model, &beta,
      attn.Q, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wk, CUDA_R_16F, head_dim, X, CUDA_R_16F, d_model, &beta,
      attn.K, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wv, CUDA_R_16F, head_dim, X, CUDA_R_16F, d_model, &beta,
      attn.V, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  // dP = dout @ V^T (f16×f16^T→f32)
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      seq_len, seq_len, head_dim, &alpha,
      attn.V, CUDA_R_16F, head_dim, dout, CUDA_R_16F, head_dim, &beta,
      attn.dP, CUDA_R_32F, seq_len, COMPUTE_F32, TENSOR_OP);

  // dS = softmax_bw_f32_dP(P, dP)
  softmax_bw_f32_dP_kernel<<<seq_len, 256, 256 * sizeof(float)>>>(P, attn.dP, attn.dS, seq_len, seq_len);

  // dS /= sqrt(head_dim)
  float rcp_scale = sqrtf((float)head_dim);
  {
    int n = seq_len * seq_len;
    scale_kernel_f16<<<(n + 255) / 256, 256>>>(attn.dS, rcp_scale, n);
  }

  // causal mask backward
  {
    dim3 t16(16, 16);
    causal_mask_bw_kernel<<<dim3((seq_len + 15) / 16, (seq_len + 15) / 16), t16>>>(attn.dS, seq_len);
  }

  // dV = P^T @ dout (f16×f16→f16)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      head_dim, seq_len, seq_len, &alpha,
      dout, CUDA_R_16F, head_dim, P, CUDA_R_16F, seq_len, &beta,
      attn.dV, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  // dQ = dS @ K (f16×f16→f16)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, seq_len, &alpha,
      attn.K, CUDA_R_16F, head_dim, attn.dS, CUDA_R_16F, seq_len, &beta,
      attn.dQ, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  // dK = dS^T @ Q (f16×f16→f16, same pattern as dV)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      head_dim, seq_len, seq_len, &alpha,
      attn.Q, CUDA_R_16F, head_dim, attn.dS, CUDA_R_16F, seq_len, &beta,
      attn.dK, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  // dX = dQ @ wq^T + dK @ wk^T + dV @ wv^T  [seq_len x d_model] f16
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_model, seq_len, head_dim, &alpha,
      wq, CUDA_R_16F, head_dim, attn.dQ, CUDA_R_16F, head_dim, &beta,
      dX, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP);

  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_model, seq_len, head_dim, &alpha,
      wk, CUDA_R_16F, head_dim, attn.dK, CUDA_R_16F, head_dim, &beta,
      attn.dX_temp, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP);
  cuda_add_inplace_f16(dX, attn.dX_temp, seq_len * d_model);

  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_model, seq_len, head_dim, &alpha,
      wv, CUDA_R_16F, head_dim, attn.dV, CUDA_R_16F, head_dim, &beta,
      attn.dX_temp, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP);
  cuda_add_inplace_f16(dX, attn.dX_temp, seq_len * d_model);

  clamp_f16_kernel<<<(seq_len * d_model + 255) / 256, 256>>>(dX, seq_len * d_model);

  // dwq = X^T @ dQ  [d_model x head_dim] f16×f16→f32
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      head_dim, d_model, seq_len, &alpha,
      attn.dQ, CUDA_R_16F, head_dim, X, CUDA_R_16F, d_model, &beta,
      dwq, CUDA_R_32F, head_dim, COMPUTE_F32, TENSOR_OP);

  // dwk = X^T @ dK
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      head_dim, d_model, seq_len, &alpha,
      attn.dK, CUDA_R_16F, head_dim, X, CUDA_R_16F, d_model, &beta,
      dwk, CUDA_R_32F, head_dim, COMPUTE_F32, TENSOR_OP);

  // dwv = X^T @ dV
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      head_dim, d_model, seq_len, &alpha,
      attn.dV, CUDA_R_16F, head_dim, X, CUDA_R_16F, d_model, &beta,
      dwv, CUDA_R_32F, head_dim, COMPUTE_F32, TENSOR_OP);
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

// ─── attention head forward (uses pre-allocated scratch + fused softmax) ────

extern "C" void
cuda_attention_head_forward(const __half *embeddings,
                             const __half *wq,
                             const __half *wk,
                             const __half *wv,
                             __half *out,
                             __half *P_out,
                             int seq_len, int d_model, int head_dim) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  ensure_attn(seq_len, head_dim, d_model, 1);

  // Q = X @ Wq (f16×f16→f32)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wq, CUDA_R_16F, head_dim, embeddings, CUDA_R_16F, d_model, &beta,
      attn.q_f32, CUDA_R_32F, head_dim, COMPUTE_F32, TENSOR_OP);

  // K = X @ Wk (f16×f16→f32)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wk, CUDA_R_16F, head_dim, embeddings, CUDA_R_16F, d_model, &beta,
      attn.k_f32, CUDA_R_32F, head_dim, COMPUTE_F32, TENSOR_OP);

  // V = X @ Wv (f16×f16→f16)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wv, CUDA_R_16F, head_dim, embeddings, CUDA_R_16F, d_model, &beta,
      attn.v_f16, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  // Scores = Q @ K^T (f32×f32^T→f32) [seq_len x seq_len]
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      seq_len, seq_len, head_dim, &alpha,
      attn.k_f32, CUDA_R_32F, head_dim, attn.q_f32, CUDA_R_32F, head_dim, &beta,
      attn.scores_f32, CUDA_R_32F, seq_len, COMPUTE_F32, TENSOR_OP);

  // Fused scale (÷√hd) + causal_mask + softmax (f32→f16, single kernel)
  float inv_scale = 1.0f / sqrtf((float)head_dim);
  int threads = 256;
  fused_scale_causal_softmax_kernel<<<seq_len, threads, threads * sizeof(float)>>>(
      attn.scores_f32, attn.scores_f16, inv_scale, seq_len);

  // Copy softmax output to P_out for backward
  cudaMemcpy(P_out, attn.scores_f16, seq_len * seq_len * sizeof(__half),
             cudaMemcpyDeviceToDevice);

  // Out = scores @ V (f16×f16→f16) [seq_len x head_dim]
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, seq_len, &alpha,
      attn.v_f16, CUDA_R_16F, head_dim, attn.scores_f16, CUDA_R_16F, seq_len, &beta,
      out, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);
}

// ─── attention head (inference, no P save, uses scratch + fused softmax) ────

extern "C" void
cuda_attention_head(const __half *embeddings,
                     const __half *wq,
                     const __half *wk,
                     const __half *wv,
                     __half *out,
                     int seq_len, int d_model, int head_dim) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  ensure_attn(seq_len, head_dim, d_model, 1);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wq, CUDA_R_16F, head_dim, embeddings, CUDA_R_16F, d_model, &beta,
      attn.q_f32, CUDA_R_32F, head_dim, COMPUTE_F32, TENSOR_OP);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wk, CUDA_R_16F, head_dim, embeddings, CUDA_R_16F, d_model, &beta,
      attn.k_f32, CUDA_R_32F, head_dim, COMPUTE_F32, TENSOR_OP);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, d_model, &alpha,
      wv, CUDA_R_16F, head_dim, embeddings, CUDA_R_16F, d_model, &beta,
      attn.v_f16, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);

  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      seq_len, seq_len, head_dim, &alpha,
      attn.k_f32, CUDA_R_32F, head_dim, attn.q_f32, CUDA_R_32F, head_dim, &beta,
      attn.scores_f32, CUDA_R_32F, seq_len, COMPUTE_F32, TENSOR_OP);

  float inv_scale = 1.0f / sqrtf((float)head_dim);
  int threads = 256;
  fused_scale_causal_softmax_kernel<<<seq_len, threads, threads * sizeof(float)>>>(
      attn.scores_f32, attn.scores_f16, inv_scale, seq_len);

  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      head_dim, seq_len, seq_len, &alpha,
      attn.v_f16, CUDA_R_16F, head_dim, attn.scores_f16, CUDA_R_16F, seq_len, &beta,
      out, CUDA_R_16F, head_dim, COMPUTE_F32, TENSOR_OP);
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
    __half *out_all, __half *P_all,
    int seq, int d, int hd, int h) {

  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  ensure_attn(seq, hd, d, h);

  // 1. Q = X @ Wq (batched f16×f16→f32)  [h, seq, hd]
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, d, &alpha,
      wq_all, CUDA_R_16F, hd, d*hd,
      X, CUDA_R_16F, d, 0,
      &beta,
      attn.q_f32, CUDA_R_32F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 2. K = X @ Wk (batched f16×f16→f32)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, d, &alpha,
      wk_all, CUDA_R_16F, hd, d*hd,
      X, CUDA_R_16F, d, 0,
      &beta,
      attn.k_f32, CUDA_R_32F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 3. V = X @ Wv (batched f16×f16→f16)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, d, &alpha,
      wv_all, CUDA_R_16F, hd, d*hd,
      X, CUDA_R_16F, d, 0,
      &beta,
      attn.v_f16, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 4. Scores = Q @ K^T (batched f32×f32→f32)  [h, seq, seq]
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      seq, seq, hd, &alpha,
      attn.k_f32, CUDA_R_32F, hd, seq*hd,
      attn.q_f32, CUDA_R_32F, hd, seq*hd,
      &beta,
      attn.scores_f32, CUDA_R_32F, seq, seq*seq,
      h, COMPUTE_F32, TENSOR_OP);

  // 5. Per-head: fused scale + causal_mask + softmax (element-wise)
  {
    float inv_scale = 1.0f / sqrtf((float)hd);
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
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, seq, &alpha,
      attn.v_f16, CUDA_R_16F, hd, seq*hd,
      P_all, CUDA_R_16F, seq, seq*seq,
      &beta,
      out_all, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP);
}

// ─── Batched multi-head attention backward ──────────────────────────────────
// Reverses the batched forward: 8 strided-batched GEMMs + per-head
// element-wise operations. Accumulates dX from all heads into a single buffer.

extern "C" void cuda_attention_backward_batched(
    const __half *X,                    // [seq, d] — saved LN1 input
    const __half *wq_all,               // [h, d, hd]
    const __half *wk_all,               // [h, d, hd]
    const __half *wv_all,               // [h, d, hd]
    const __half *P_all,                // [h, seq, seq] — saved softmax outputs
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
  cuda_extract_strided_cols(dout_full, attn.dQ, seq, d, hd, h, hd);

  // 1. dP = dout @ V^T (batched f16×f16^T→f32)  [h, seq, seq]
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      seq, seq, hd, &alpha,
      attn.v_f16, CUDA_R_16F, hd, seq*hd,   // V (reused from forward, same scratch)
      attn.dQ, CUDA_R_16F, hd, seq*hd,       // dQ used as dout temp buffer
      &beta,
      attn.dP, CUDA_R_32F, seq, seq*seq,
      h, COMPUTE_F32, TENSOR_OP);

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
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, seq, seq, &alpha,
      attn.dQ, CUDA_R_16F, hd, seq*hd,       // dout temp
      P_all, CUDA_R_16F, seq, seq*seq,
      &beta,
      attn.dV, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 4. dQ = dS @ K (batched f32×f16→f16) — reuse q_f32 scratch
  //    k_f32 is f32 from forward; dS is f16
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, seq, &alpha,
      (const float*)attn.k_f32, CUDA_R_32F, hd, seq*hd,
      attn.dS, CUDA_R_16F, seq, seq*seq,
      &beta,
      attn.dQ, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 5. dK = dS^T @ Q (batched f32×f16→f16)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, seq, seq, &alpha,
      (const float*)attn.q_f32, CUDA_R_32F, hd, seq*hd,
      attn.dS, CUDA_R_16F, seq, seq*seq,
      &beta,
      attn.dK, CUDA_R_16F, hd, seq*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 6. dX = sum_i( dQ_i @ wq_i^T + dK_i @ wk_i^T + dV_i @ wv_i^T )
  //    First head writes (β=0), rest accumulate (β=1)
  // dX from dQ @ wq^T
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d, seq, hd, &alpha,
      wq_all, CUDA_R_16F, hd,
      attn.dQ, CUDA_R_16F, hd,
      &beta,
      dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP);

  for (int hi = 1; hi < h; hi++) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        d, seq, hd, &alpha,
        wq_all + hi * d * hd, CUDA_R_16F, hd,
        attn.dQ + hi * seq * hd, CUDA_R_16F, hd,
        &alpha, dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP);
  }

  // dX += dK @ wk^T
  for (int hi = 0; hi < h; hi++) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        d, seq, hd, &alpha,
        wk_all + hi * d * hd, CUDA_R_16F, hd,
        attn.dK + hi * seq * hd, CUDA_R_16F, hd,
        &alpha, dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP);
  }

  // dX += dV @ wv^T
  for (int hi = 0; hi < h; hi++) {
    cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
        d, seq, hd, &alpha,
        wv_all + hi * d * hd, CUDA_R_16F, hd,
        attn.dV + hi * seq * hd, CUDA_R_16F, hd,
        &alpha, dX, CUDA_R_16F, d, COMPUTE_F32, TENSOR_OP);
  }

  clamp_f16_kernel<<<(seq * d + 255) / 256, 256>>>(dX, seq * d);

  float one = 1.0f;
  // 7. Weight gradients: dwq = X^T @ dQ (batched f16×f16→f32, accumulate β=1)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, d, seq, &alpha,
      attn.dQ, CUDA_R_16F, hd, seq*hd,
      X, CUDA_R_16F, d, 0,
      &one,
      dwq_all, CUDA_R_32F, hd, d*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 8. dwk = X^T @ dK (batched, accumulate β=1)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, d, seq, &alpha,
      attn.dK, CUDA_R_16F, hd, seq*hd,
      X, CUDA_R_16F, d, 0,
      &one,
      dwk_all, CUDA_R_32F, hd, d*hd,
      h, COMPUTE_F32, TENSOR_OP);

  // 9. dwv = X^T @ dV (batched, accumulate β=1)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      hd, d, seq, &alpha,
      attn.dV, CUDA_R_16F, hd, seq*hd,
      X, CUDA_R_16F, d, 0,
      &one,
      dwv_all, CUDA_R_32F, hd, d*hd,
      h, COMPUTE_F32, TENSOR_OP);
}

// ─── dW = A^T @ dB (f16×f16→f32, for backward weight gradients) ──────────
// A: [rows × cols_a] f16 row-major, dB: [rows × cols_b] f16 row-major
// dW: [cols_a × cols_b] f32 row-major
extern "C" void cuda_d_weight(const __half *A, const __half *dB, float *dW,
                               int rows, int cols_a, int cols_b) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      cols_b, cols_a, rows, &alpha,
      dB, CUDA_R_16F, cols_b, A, CUDA_R_16F, cols_a, &beta,
      dW, CUDA_R_32F, cols_b, COMPUTE_F32, TENSOR_OP);
}

// ─── cuBLAS wrappers for matmul and matmul_bt ──────────────────────────────

extern "C" void cuda_matmul(const __half *a, const __half *b, __half *result,
                             int rows_a, int cols_a, int cols_b) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      cols_b, rows_a, cols_a, &alpha,
      b, CUDA_R_16F, cols_b, a, CUDA_R_16F, cols_a, &beta,
      result, CUDA_R_16F, cols_b, COMPUTE_F32, TENSOR_OP);
}

extern "C" void cuda_matmul_bt(const __half *a, const __half *b, __half *result,
                                int rows_a, int cols_a, int rows_b) {
  cublasHandle_t handle = get_cublas();
  float alpha = 1.0f, beta = 0.0f;
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      rows_b, rows_a, cols_a, &alpha,
      b, CUDA_R_16F, cols_a, a, CUDA_R_16F, cols_a, &beta,
      result, CUDA_R_16F, rows_b, COMPUTE_F32, TENSOR_OP);
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
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_ff, seq_len, d_model, &alpha,
      w1, CUDA_R_16F, d_ff, input, CUDA_R_16F, d_model, &beta,
      d_hidden, CUDA_R_16F, d_ff, COMPUTE_F32, TENSOR_OP);

  // Fused bias_add + GELU (single kernel)
  fused_bias_gelu_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(
      d_hidden, b1, d_ff, seq_len * d_ff);

  // out = hidden @ w2 (f16×f16→f16)
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_model, seq_len, d_ff, &alpha,
      w2, CUDA_R_16F, d_model, d_hidden, CUDA_R_16F, d_ff, &beta,
      out, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP);

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
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_ff, seq_len, d_model, &alpha,
      w1, CUDA_R_16F, d_ff, input, CUDA_R_16F, d_model, &beta,
      d_hidden, CUDA_R_16F, d_ff, COMPUTE_F32, TENSOR_OP);

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
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      d_model, seq_len, d_ff, &alpha,
      w2, CUDA_R_16F, d_model, d_hidden, CUDA_R_16F, d_ff, &beta,
      out, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP);

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
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_ff, seq_len, d_model, &alpha,
      w2, CUDA_R_16F, d_model, dout, CUDA_R_16F, d_model, &beta,
      d_hidden_gelu, CUDA_R_16F, d_ff, COMPUTE_F32, TENSOR_OP);

  // Compute activated = gelu(pre_gelu) dynamically to use for dw2 gradient.
  // Distinct from d_hidden_gelu above: both are live across the dw2 GEMM.
  __half *activated = ff.activated;
  cudaMemcpy(activated, pre_gelu, seq_len * d_ff * sizeof(__half), cudaMemcpyDeviceToDevice);
  gelu_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(activated, seq_len * d_ff);

  // dw2 = activated^T @ dout (f16×f16→f32) [d_ff x d_model]
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      d_model, d_ff, seq_len, &alpha,
      dout, CUDA_R_16F, d_model, activated, CUDA_R_16F, d_ff, &beta,
      dw2, CUDA_R_32F, d_model, COMPUTE_F32, TENSOR_OP);

  cuda_bias_grad(dout, db2, seq_len, d_model);

  // gelu backward on d_hidden_gelu in-place
  gelu_bw_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(pre_gelu, d_hidden_gelu,
                                                          d_hidden_gelu,
                                                          seq_len * d_ff);

  // dw1 = input^T @ d_hidden_gelu (f16×f16→f32) [d_model x d_ff]
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      d_ff, d_model, seq_len, &alpha,
      d_hidden_gelu, CUDA_R_16F, d_ff, input, CUDA_R_16F, d_model, &beta,
      dw1, CUDA_R_32F, d_ff, COMPUTE_F32, TENSOR_OP);

  cuda_bias_grad(d_hidden_gelu, db1, seq_len, d_ff);

  // dinput = d_hidden_gelu @ w1^T (f16×f16^T→f16) [seq_len x d_model]
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_model, seq_len, d_ff, &alpha,
      w1, CUDA_R_16F, d_ff, d_hidden_gelu, CUDA_R_16F, d_ff, &beta,
      dinput, CUDA_R_16F, d_model, COMPUTE_F32, TENSOR_OP);
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
__global__ void embedding_forward_kernel(
  const int *token_ids,
  const __half *embedding_table,
  const __half *positional_table,
  __half *output,
  int s, int d, int vocab
) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= s * d) return;
  int pos = i / d;
  int j = i % d;
  int token = token_ids[pos];
  output[i] = __hadd(embedding_table[token * d + j], positional_table[pos * d + j]);
}

extern "C" void cuda_embedding_forward(
  const int *token_ids,
  const __half *embedding_table,
  const __half *positional_table,
  __half *output,
  int s, int d, int vocab
) {
  embedding_forward_kernel<<<(s * d + 255) / 256, 256>>>(token_ids, embedding_table, positional_table, output, s, d, vocab);
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
