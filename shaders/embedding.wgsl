struct Params {
    vocab_size: u32,
    dimensions: u32,
    seq_len: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tokens: array<u32>;
@group(0) @binding(2) var<storage, read> embedding_table: array<f32>;
@group(0) @binding(3) var<storage, read> positional_table: array<f32>;
@group(0) @binding(4) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn forward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    let t = idx / params.dimensions;
    let d = idx % params.dimensions;

    if (t >= params.seq_len) {
        return;
    }

    let token_id = tokens[t];
    let emb_val = embedding_table[token_id * params.dimensions + d];
    let pos_val = positional_table[t * params.dimensions + d];

    output[t * params.dimensions + d] = emb_val + pos_val;
}

@group(0) @binding(4) var<storage, read> d_input: array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_embedding: array<atomic<i32>>;
@group(0) @binding(6) var<storage, read_write> grad_positional: array<f32>;

// Helper to add float to atomic i32 (fixed point)
fn atomicAddFloat(ptr_idx: u32, val: f32) {
    // Clamp the float gradient value to prevent i32 overflow after scaling
    let clamped_val = clamp(val, -2000.0, 2000.0); // Assuming typical gradients are within this range
    let scaled_val = i32(clamped_val * 1000000.0);
    atomicAdd(&grad_embedding[ptr_idx], scaled_val);
}

@compute @workgroup_size(64)
fn backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    let t = idx / params.dimensions;
    let d = idx % params.dimensions;

    if (t >= params.seq_len) {
        return;
    }

    let token_id = tokens[t];
    let grad_val = d_input[t * params.dimensions + d];

    // Use atomic add to avoid race conditions when the same token appears multiple times in a sequence.
    // Since WGSL doesn't have atomicAdd for floats, we use fixed-point with i32.
    atomicAddFloat(token_id * params.dimensions + d, grad_val);
    
    // Positional gradients don't have race conditions as 't' is unique per thread.
    grad_positional[t * params.dimensions + d] += grad_val;
}
