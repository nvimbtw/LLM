struct Params {
    len: u32,
    lr: f32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> weights: array<f32>;

// Path 1: Standard Floating Point Gradients
@group(0) @binding(2) var<storage, read> grads_f32: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= params.len) {
        return;
    }
    weights[idx] = weights[idx] - params.lr * grads_f32[idx];
}

// Path 2: Fixed Point Gradients (for Embedding)
@group(0) @binding(2) var<storage, read> grads_i32: array<i32>;

@compute @workgroup_size(64)
fn update_fixed_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= params.len) {
        return;
    }
    // Convert fixed point back to float (1e6 scaling)
    let float_grad = f32(grads_i32[idx]) / 1000000.0;
    weights[idx] = weights[idx] - params.lr * float_grad;
}
