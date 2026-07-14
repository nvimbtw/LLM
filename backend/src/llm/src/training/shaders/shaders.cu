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

extern "C" void cublas_shutdown() {
    ensure_attn_cleanup();

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
__global__ void fused_scale_causal_softmax_kernel(
    const float *__restrict__ scores_f32, __half *__restrict__ scores_f16,
    float inv_scale, int seq_len) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len) return;

  const float *row_in = scores_f32 + row * seq_len;
  __half *row_out = scores_f16 + row * seq_len;

  float max_val = -1e38f;
  for (int j = 0; j <= row; j++)
    if (row_in[j] > max_val) max_val = row_in[j];

  float sum = 0.0f;
  for (int j = 0; j < seq_len; j++) {
    if (j <= row) {
      float e = expf((row_in[j] - max_val) * inv_scale);
      row_out[j] = __float2half(e);
      sum += e;
    } else {
      row_out[j] = __float2half(0.0f);
    }
  }

  float inv_sum = 1.0f / sum;
  for (int j = 0; j < seq_len; j++)
    if (j <= row)
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

// Adam update. `w_master` (f32) is the canonical weight; `w_view` (f16) is a
// derived copy the matmuls consume, refreshed from the master in the same pass.
// The two moment estimates (m = 1st, v = 2nd) are f32. `grad_scale` averages the
// accumulated gradient (1/accum_steps); `bc1`, `bc2` are the bias-correction
// denominators (1 - beta^t) computed host-side. Accumulating the step in f32
// (not the old f16 read-back) is the mixed-precision fix that lifts the LR cap
// (ISSUE-0052): a sub-f16-ULP `Δw` now survives instead of rounding away.
__global__ void adam_kernel(const float *__restrict__ g, float *w_master,
                            __half *w_view, float *m, float *v, float grad_scale,
                            float lr, float beta1, float beta2, float eps,
                            float bc1, float bc2, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float grad = g[i] * grad_scale;
  float new_m = beta1 * m[i] + (1.0f - beta1) * grad;
  float new_v = beta2 * v[i] + (1.0f - beta2) * grad * grad;
  m[i] = new_m;
  v[i] = new_v;
  float m_hat = new_m / bc1;
  float v_hat = new_v / bc2;
  float updated = w_master[i] - lr * m_hat / (sqrtf(v_hat) + eps);
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
__global__ void cross_entropy_bw_kernel(const __half *logits,
                                         const float *targets,
                                         const float *mask,
                                         __half *dlogits, int seq_len,
                                         int vocab, int active) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len)
    return;

  // Ignored position: no learning signal — zero its whole gradient row.
  if (mask != nullptr && mask[row] == 0.0f) {
    for (int j = 0; j < vocab; j++)
      dlogits[row * vocab + j] = __float2half(0.0f);
    return;
  }

  int target = (int)(targets[row]);

  float max_val = __half2float(logits[row * vocab]);
  for (int j = 1; j < vocab; j++)
    max_val = fmaxf(max_val, __half2float(logits[row * vocab + j]));

  float sum = 0.0f;
  for (int j = 0; j < vocab; j++)
    sum += expf(__half2float(logits[row * vocab + j]) - max_val);

  float inv_sum = 1.0f / sum;
  for (int j = 0; j < vocab; j++) {
    float softmax_j = expf(__half2float(logits[row * vocab + j]) - max_val) * inv_sum;
    dlogits[row * vocab + j] = __float2half((softmax_j - (j == target ? 1.0f : 0.0f)) / active);
  }
}

extern "C" void cuda_cross_entropy_backward(const __half *logits,
                                             const float *targets,
                                             const float *mask,
                                             __half *dlogits, int seq_len,
                                             int vocab, int active) {
  cross_entropy_bw_kernel<<<(seq_len + 255) / 256, 256>>>(
      logits, targets, mask, dlogits, seq_len, vocab, active);
}

// ─── cross-entropy loss (GPU, async-friendly via event) ─────────────────

// `mask` (length seq_len, 1=train / 0=ignore) may be null, meaning "every row is
// active". Masked rows contribute 0 to the loss, which is normalized by `active`
// (the trained-row count) rather than seq_len so the reported loss is the mean
// over trained positions only.
__global__ void cross_entropy_loss_kernel(const __half *logits,
                                           const float *targets,
                                           const float *mask,
                                           float *loss_out, int seq_len,
                                           int vocab, int active) {
  extern __shared__ float shared[];
  int row = blockIdx.x * blockDim.x + threadIdx.x;

  float contrib = 0.0f;
  if (row < seq_len && (mask == nullptr || mask[row] != 0.0f)) {
    int target = (int)targets[row];

    float max_val = __half2float(logits[row * vocab]);
    for (int j = 1; j < vocab; j++)
      max_val = fmaxf(max_val, __half2float(logits[row * vocab + j]));

    float sum = 0.0f;
    for (int j = 0; j < vocab; j++)
      sum += expf(__half2float(logits[row * vocab + j]) - max_val);

    float log_sum_exp = max_val + logf(sum);
    contrib = log_sum_exp - __half2float(logits[row * vocab + target]);
  }

  shared[threadIdx.x] = contrib;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s)
      shared[threadIdx.x] += shared[threadIdx.x + s];
    __syncthreads();
  }

  if (threadIdx.x == 0)
    atomicAdd(loss_out, shared[0] / active);
}

extern "C" float cuda_cross_entropy_loss(const __half *logits,
                                          const float *targets,
                                          const float *mask, int seq_len,
                                          int vocab, int active) {
  float *d_loss;
  cudaMalloc(&d_loss, sizeof(float));
  cudaMemset(d_loss, 0, sizeof(float));
  int threads = 256;
  int blocks = (seq_len + threads - 1) / threads;
  cross_entropy_loss_kernel<<<blocks, threads, threads * sizeof(float)>>>(
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

__global__ void layer_norm_bw_kernel(const __half *x, const __half *dy,
                                      const __half *gamma, __half *dx,
                                      float *dgamma, float *dbeta, int rows,
                                      int cols, float eps) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (col >= cols)
    return;

  float sum_dgamma = 0.0f;
  float sum_dbeta = 0.0f;
  for (int r = 0; r < rows; r++) {
    const __half *row_x = x + r * cols;
    const __half *row_dy = dy + r * cols;
    __half *row_dx = dx + r * cols;

    float mean = 0.0f;
    for (int j = 0; j < cols; j++)
      mean += __half2float(row_x[j]);
    mean /= cols;

    float var = 0.0f;
    for (int j = 0; j < cols; j++) {
      float diff = __half2float(row_x[j]) - mean;
      var += diff * diff;
    }
    var /= cols;

    float sigma = sqrtf(var + eps);
    float inv_sigma = 1.0f / sigma;

    float mean_dy = 0.0f;
    for (int j = 0; j < cols; j++)
      mean_dy += __half2float(row_dy[j]);
    mean_dy /= cols;

    float mean_dy_x = 0.0f;
    for (int j = 0; j < cols; j++) {
      float x_centered = __half2float(row_x[j]) - mean;
      mean_dy_x += __half2float(row_dy[j]) * x_centered;
    }
    mean_dy_x /= cols;

    float x_centered = __half2float(row_x[col]) - mean;
    sum_dgamma += __half2float(row_dy[col]) * x_centered * inv_sigma;
    sum_dbeta += __half2float(row_dy[col]);

    float dx_val =
        (__half2float(gamma[col]) * inv_sigma) *
        (__half2float(row_dy[col]) - mean_dy -
         x_centered * inv_sigma * inv_sigma * mean_dy_x);
    dx_val = fminf(fmaxf(dx_val, -65504.0f), 65504.0f);
    row_dx[col] = __float2half(dx_val);
  }

  dgamma[col] = sum_dgamma;
  dbeta[col] = sum_dbeta;
}

extern "C" void cuda_layer_norm_backward(const __half *x, const __half *dy,
                                          const __half *gamma, __half *dx,
                                          float *dgamma, float *dbeta,
                                          int rows, int cols, float eps) {
  layer_norm_bw_kernel<<<(cols + 255) / 256, 256>>>(x, dy, gamma, dx, dgamma,
                                                      dbeta, rows, cols, eps);
}

// ─── softmax backward ──────────────────────────────────────────────────────

__global__ void softmax_bw_kernel(const __half *P, const __half *dP, __half *dS,
                                   int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const __half *P_row = P + row * cols;
  const __half *dP_row = dP + row * cols;
  __half *dS_row = dS + row * cols;

  float dot = 0.0f;
  for (int i = 0; i < cols; i++)
    dot += __half2float(P_row[i]) * __half2float(dP_row[i]);

  for (int i = 0; i < cols; i++)
    dS_row[i] = __float2half(__half2float(P_row[i]) * (__half2float(dP_row[i]) - dot));
}

extern "C" void cuda_softmax_backward(const __half *P, const __half *dP,
                                       __half *dS, int rows, int cols) {
  softmax_bw_kernel<<<(rows + 255) / 256, 256>>>(P, dP, dS, rows, cols);
}

// ─── causal mask backward ──────────────────────────────────────────────────

__global__ void causal_mask_bw_kernel(__half *dS, int seq_len) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= seq_len || col >= seq_len)
    return;
  if (col > row)
    dS[row * seq_len + col] = __float2half(0.0f);
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
  softmax_bw_f32_dP_kernel<<<(seq_len + 255) / 256, 256>>>(P, attn.dP, attn.dS, seq_len, seq_len);

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

__global__ void softmax_bw_f32_dP_kernel(const __half *P, const float *dP,
                                          __half *dS, int rows, int cols) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const __half *P_row = P + row * cols;
  const float *dP_row = dP + row * cols;
  __half *dS_row = dS + row * cols;

  float dot = 0.0f;
  for (int i = 0; i < cols; i++)
    dot += __half2float(P_row[i]) * dP_row[i];

  for (int i = 0; i < cols; i++) {
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

__global__ void layer_norm_kernel(const __half *in, __half *out,
                                  const __half *gamma, const __half *beta,
                                  int rows, int cols, float eps) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= rows)
    return;
  const __half *row_in = in + row * cols;
  __half *row_out = out + row * cols;

  float mean = 0.0f;
  for (int i = 0; i < cols; i++)
    mean += __half2float(row_in[i]);
  mean /= cols;

  float var = 0.0f;
  for (int i = 0; i < cols; i++) {
    float diff = __half2float(row_in[i]) - mean;
    var += diff * diff;
  }
  var /= cols;

  float inv_std = rsqrtf(var + eps);
  for (int i = 0; i < cols; i++)
    row_out[i] = __float2half(
        (__half2float(row_in[i]) - mean) * inv_std * __half2float(gamma[i]) + __half2float(beta[i]));
}

extern "C" void cuda_layer_norm(const __half *in, __half *out,
                                const __half *gamma, const __half *beta,
                                int rows, int cols, float eps) {
  layer_norm_kernel<<<(rows + 255) / 256, 256>>>(in, out, gamma, beta, rows,
                                                   cols, eps);
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
  fused_scale_causal_softmax_kernel<<<(seq_len + threads - 1) / threads, threads>>>(
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
  fused_scale_causal_softmax_kernel<<<(seq_len + threads - 1) / threads, threads>>>(
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
    int g = (seq + t - 1) / t;
    for (int hi = 0; hi < h; hi++) {
      fused_scale_causal_softmax_kernel<<<g, t>>>(
          attn.scores_f32 + hi * seq * seq,
          attn.scores_f16 + hi * seq * seq,
          inv_scale, seq);
    }
  }

  // Copy P_all from scratch (needed for backward)
  cudaMemcpy(P_all, attn.scores_f16, h * seq * seq * sizeof(__half),
             cudaMemcpyDeviceToDevice);

  // 6. Out = softmax(S) @ V (batched f16×f16→f16)
  cublasGemmStridedBatchedEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
      hd, seq, seq, &alpha,
      attn.v_f16, CUDA_R_16F, hd, seq*hd,
      attn.scores_f16, CUDA_R_16F, seq, seq*seq,
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
    for (int hi = 0; hi < h; hi++) {
      softmax_bw_f32_dP_kernel<<<(seq + 255) / 256, 256>>>(
          P_all + hi * seq * seq,
          attn.dP + hi * seq * seq,
          attn.dS + hi * seq * seq, seq, seq);
      scale_kernel_f16<<<((seq * seq) + 255) / 256, 256>>>(
          attn.dS + hi * seq * seq, rcp, seq * seq);
      {
        dim3 t16(16, 16);
        causal_mask_bw_kernel<<<dim3((seq + 15) / 16, (seq + 15) / 16), t16>>>(
            attn.dS + hi * seq * seq, seq);
      }
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

  __half *d_hidden;
  cudaMalloc(&d_hidden, seq_len * d_ff * sizeof(__half));

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

  cudaFree(d_hidden);
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

  __half *d_hidden;
  cudaMalloc(&d_hidden, seq_len * d_ff * sizeof(__half));

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

  cudaFree(d_hidden);
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

  __half *d_hidden_gelu;
  cudaMalloc(&d_hidden_gelu, seq_len * d_ff * sizeof(__half));

  // d_hidden_gelu = dout @ w2^T (f16×f16^T→f16) [seq_len x d_ff]
  cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N,
      d_ff, seq_len, d_model, &alpha,
      w2, CUDA_R_16F, d_model, dout, CUDA_R_16F, d_model, &beta,
      d_hidden_gelu, CUDA_R_16F, d_ff, COMPUTE_F32, TENSOR_OP);

  // Compute activated = gelu(pre_gelu) dynamically to use for dw2 gradient
  __half *activated;
  cudaMalloc(&activated, seq_len * d_ff * sizeof(__half));
  cudaMemcpy(activated, pre_gelu, seq_len * d_ff * sizeof(__half), cudaMemcpyDeviceToDevice);
  gelu_kernel<<<(seq_len * d_ff + 255) / 256, 256>>>(activated, seq_len * d_ff);

  // dw2 = activated^T @ dout (f16×f16→f32) [d_ff x d_model]
  cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_T,
      d_model, d_ff, seq_len, &alpha,
      dout, CUDA_R_16F, d_model, activated, CUDA_R_16F, d_ff, &beta,
      dw2, CUDA_R_32F, d_model, COMPUTE_F32, TENSOR_OP);

  cudaFree(activated);

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

  cudaFree(d_hidden_gelu);
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

// ─── Gradient L2-norm clipping ──────────────────────────────────────────────

__global__ void clip_norm_kernel(float *g, int n, float max_norm) {
  extern __shared__ float shared[];
  int tid = threadIdx.x;
  int stride = blockDim.x;

  float sum = 0.0f;
  for (int i = tid; i < n; i += stride) {
    float v = g[i];
    if (isnan(v) || isinf(v)) v = 0.0f;
    g[i] = v;
    sum += v * v;
  }

  shared[tid] = sum;
  __syncthreads();

  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      shared[tid] += shared[tid + s];
    __syncthreads();
  }

  if (tid == 0) {
    float norm = sqrtf(shared[0]);
    shared[0] = (norm > max_norm) ? (max_norm / norm) : 1.0f;
  }
  __syncthreads();

  float scale = shared[0];
  for (int i = tid; i < n; i += stride)
    g[i] *= scale;
}

extern "C" void cuda_clip_norm(float *g, int n, float max_norm) {
  int threads = 256;
  clip_norm_kernel<<<1, threads, threads * sizeof(float)>>>(g, n, max_norm);
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
                          float beta1, float beta2, float eps, float bc1,
                          float bc2, int n) {
  adam_kernel<<<(n + 255) / 256, 256>>>(g, w_master, w_view, m, v, grad_scale,
                                        lr, beta1, beta2, eps, bc1, bc2, n);
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
