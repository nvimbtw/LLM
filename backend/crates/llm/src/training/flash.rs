//! The flash attention switch.
//!
//! Attention has two implementations in `shaders.cu`: the original cuBLAS path,
//! which materializes an `[h, seq, seq]` score matrix per layer, and the fused
//! FlashAttention kernels, which never let that matrix leave shared memory. Both
//! are compiled in, and this is what chooses between them.
//!
//! The switch exists for two reasons. The A/B tests need to run the same input
//! through both and compare, which is the only check that can say a kernel this
//! subtle is right — and `LLM_ATTN_FLASH=0` is a one-variable fallback if
//! something surfaces in the middle of a long run, without a rebuild.

use crate::training::device_buffer::{Buffer, DeviceBufferF16};
use std::ffi::c_void;

unsafe extern "C" {
    fn cuda_attn_set_flash(on: i32);
    fn cuda_attn_flash_active(hd: i32) -> i32;
    fn cuda_flash_probe_acc_layout() -> i32;
    fn cuda_flash_row_granularity() -> i32;
    fn cuda_flash_attention_forward(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *mut c_void,
        lse: *mut f32,
        seq: i32,
        hd: i32,
        h: i32,
    );
    fn cuda_flash_attention_backward(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        o: *const c_void,
        d_o: *const c_void,
        lse: *const f32,
        dq: *mut c_void,
        dk: *mut c_void,
        dv: *mut c_void,
        seq: i32,
        hd: i32,
        h: i32,
    );
}

/// Q, K and V padded to the flash row granularity and uploaded as f16.
struct Padded {
    q: DeviceBufferF16,
    k: DeviceBufferF16,
    v: DeviceBufferF16,
    live: usize,
    padded: usize,
    rows: usize,
}

fn pad_qkv(q: &[f32], k: &[f32], v: &[f32], seq: usize, hd: usize, h: usize) -> Padded {
    let live = h * seq * hd;
    assert_eq!(q.len(), live);
    assert_eq!(k.len(), live);
    assert_eq!(v.len(), live);
    // The kernels read whole 16-row wmma slabs, so the tail past `seq` has to
    // exist and — more importantly — be finite. See `ensure_attn` in shaders.cu.
    let rows = rows_for(seq);
    let padded = h * rows * hd;
    let up = |src: &[f32]| {
        let mut buf = vec![0.0f32; padded];
        buf[..live].copy_from_slice(src);
        DeviceBufferF16::from_f32_slice(&buf)
    };
    Padded { q: up(q), k: up(k), v: up(v), live, padded, rows }
}

/// Turn the flash path on or off for the rest of the process.
///
/// **A live `Workspace` or `InferSession` does not follow this.** Both read it
/// once at construction and allocate a different set of buffers accordingly —
/// `head_p_all` versus `lse_all`, and so on — so flipping the switch under one
/// leaves it holding buffers the kernels will not write. Set it first, then
/// build.
pub fn set_enabled(on: bool) {
    // SAFETY: writes one process-global int in shaders.cu; no pointers involved.
    unsafe { cuda_attn_set_flash(if on { 1 } else { 0 }) }
}

/// Whether attention will actually take the flash path at this `head_dim`.
///
/// False either because the switch is off, or because the shape is not one the
/// kernels are instantiated for (`head_dim` must be 16, 32, 64 or 128 — the
/// values `d / heads` takes when both are powers of two). An unsupported shape
/// falls back rather than being approximated.
pub fn active_for(head_dim: usize) -> bool {
    // SAFETY: a plain integer in, a flag out.
    unsafe { cuda_attn_flash_active(head_dim as i32) != 0 }
}

/// Rows the per-head `[h, *, hd]` caches must be allocated at, given `max_s`.
///
/// The kernels read and write in whole 16-row wmma slabs, four to a block, and a
/// warp whose slab sits entirely past `seq` still runs its `mma_sync` — a lane
/// may never skip one. So the caches have to cover `max_s` rounded up to the
/// block height, which is read out of `shaders.cu` rather than restated here.
pub fn rows_for(max_s: usize) -> usize {
    // SAFETY: no arguments, returns a compile-time constant.
    let g = unsafe { cuda_flash_row_granularity() } as usize;
    max_s.div_ceil(g) * g
}

/// Whether this device lays out a wmma accumulator fragment the way the kernels
/// assume.
///
/// The mapping from a fragment's 8 per-lane elements to (row, column) is not
/// specified by the wmma API. Every flash kernel depends on the Ampere one: the
/// online-softmax state lives in registers because a lane owns exactly two rows,
/// a row reduction is two shuffles because the four lanes sharing a row differ
/// only in the low bits of the lane id, and the transposed store that makes
/// `dV = P^T dO` free goes through the same map.
///
/// The kernels check this themselves and `abort()` on mismatch, because a wrong
/// layout does not fault — it yields finite, plausible gradients that still
/// descend a loss curve, and is discoverable only hours later in a ruined model.
/// This exposes the same probe as a test, so a driver or architecture change is
/// a red test rather than a wasted night.
pub fn accumulator_layout_matches() -> bool {
    // SAFETY: allocates, launches a one-warp kernel and frees, all internally.
    unsafe { cuda_flash_probe_acc_layout() != 0 }
}

/// Causal `softmax(Q Kᵀ / √hd) V` over `[h, seq, hd]` host slices, through the
/// flash kernel alone — no projections, no RoPE, no cuBLAS anywhere.
///
/// Exists for one test: comparing the kernel against an independent f64 causal
/// softmax. Every other check on these kernels is an A/B against the cuBLAS path,
/// which can only say the two agree, not what they agree *on* — whether the
/// maximum is subtracted, whether the scale sits inside the exponent, whether a
/// query attends to itself. Not a serving path; it allocates and frees per call.
pub fn attention_for_test(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    head_dim: usize,
    heads: usize,
) -> Vec<f32> {
    assert!(
        active_for(head_dim),
        "head_dim {head_dim} has no flash instantiation, or the switch is off"
    );
    let p = pad_qkv(q, k, v, seq, head_dim, heads);
    let out = DeviceBufferF16::alloc_zeroed(p.padded);
    let lse = Buffer::alloc_zeroed(heads * p.rows);

    // SAFETY: all four buffers are `heads * rows_for(seq) * head_dim` f16, which
    // covers every slab the kernel touches at this shape; `lse` is one f32 per
    // padded row per head.
    unsafe {
        cuda_flash_attention_forward(
            p.q.ptr as *const c_void,
            p.k.ptr as *const c_void,
            p.v.ptr as *const c_void,
            out.ptr as *mut c_void,
            lse.ptr,
            seq as i32,
            head_dim as i32,
            heads as i32,
        );
    }
    out.to_f32_vec()[..p.live].to_vec()
}

/// The gradients of causal attention w.r.t. Q, K and V, through the flash
/// kernels alone. Companion to [`attention_for_test`], and returned in that
/// order.
///
/// It runs the real forward first rather than taking `O` and `lse` as arguments:
/// the backward reads the forward's own saved statistics, and feeding it
/// externally computed ones would test a pairing that never occurs.
pub fn attention_backward_for_test(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_out: &[f32],
    seq: usize,
    head_dim: usize,
    heads: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    assert!(
        active_for(head_dim),
        "head_dim {head_dim} has no flash instantiation, or the switch is off"
    );
    let p = pad_qkv(q, k, v, seq, head_dim, heads);
    assert_eq!(d_out.len(), p.live);
    let mut d_out_padded = vec![0.0f32; p.padded];
    d_out_padded[..p.live].copy_from_slice(d_out);
    let d_o = DeviceBufferF16::from_f32_slice(&d_out_padded);

    let out = DeviceBufferF16::alloc_zeroed(p.padded);
    let lse = Buffer::alloc_zeroed(heads * p.rows);
    let dq = DeviceBufferF16::alloc_zeroed(p.padded);
    let dk = DeviceBufferF16::alloc_zeroed(p.padded);
    let dv = DeviceBufferF16::alloc_zeroed(p.padded);

    // SAFETY: every f16 buffer is `heads * rows_for(seq) * head_dim` elements and
    // both f32 buffers are one per padded row per head, which covers every slab
    // the kernels touch at this shape.
    unsafe {
        cuda_flash_attention_forward(
            p.q.ptr as *const c_void,
            p.k.ptr as *const c_void,
            p.v.ptr as *const c_void,
            out.ptr as *mut c_void,
            lse.ptr,
            seq as i32,
            head_dim as i32,
            heads as i32,
        );
        cuda_flash_attention_backward(
            p.q.ptr as *const c_void,
            p.k.ptr as *const c_void,
            p.v.ptr as *const c_void,
            out.ptr as *const c_void,
            d_o.ptr as *const c_void,
            lse.ptr,
            dq.ptr as *mut c_void,
            dk.ptr as *mut c_void,
            dv.ptr as *mut c_void,
            seq as i32,
            head_dim as i32,
            heads as i32,
        );
    }
    (
        dq.to_f32_vec()[..p.live].to_vec(),
        dk.to_f32_vec()[..p.live].to_vec(),
        dv.to_f32_vec()[..p.live].to_vec(),
    )
}
