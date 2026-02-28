struct Dimensions {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> input1: array<f32>;
@group(0) @binding(2) var<storage, read> input2: array<f32>;
@group(0) @binding(3) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn softmax_backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= dims.rows) {
        return;
    }

    var dot_product: f32 = 0.0;
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        let idx = row * dims.cols + j;
        dot_product = dot_product + input1[idx] * input2[idx];
    }

    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        let idx = row * dims.cols + j;
        output[idx] = input1[idx] * (input2[idx] - dot_product);
    }
}

@compute @workgroup_size(64)
fn relu_backward_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= dims.rows * dims.cols) {
        return;
    }
    if (input1[idx] <= 0.0) {
        output[idx] = 0.0;
    }
}

@compute @workgroup_size(16, 16)
fn scale_mask_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= dims.rows || j >= dims.cols) {
        return;
    }

    let idx = i * dims.cols + j;
    // Scale (Hardcoded 128 dimensions sqrt = 11.31)
    // We should ideally pass this in dims.
    output[idx] = output[idx] / 11.3137; 

    // Causal Mask
    if (j > i) {
        output[idx] = -1e9;
    }
}
