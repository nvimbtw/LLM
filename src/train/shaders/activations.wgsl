struct Dimensions {
    rows: u32,
    cols: u32,
    padding1: u32,
    padding2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn relu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    if (idx >= dims.rows * dims.cols) {
        return;
    }
    if (data[idx] < 0.0) {
        data[idx] = 0.0;
    }
}

@compute @workgroup_size(64)
fn layer_norm_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= dims.rows) {
        return;
    }

    // 1. Calculate Mean
    var sum: f32 = 0.0;
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        sum = sum + data[row * dims.cols + j];
    }
    let mean = sum / f32(dims.cols);

    // 2. Calculate Variance
    var var_sum: f32 = 0.0;
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        let diff = data[row * dims.cols + j] - mean;
        var_sum = var_sum + diff * diff;
    }
    let variance = var_sum / f32(dims.cols);
    let inv_std = 1.0 / sqrt(variance + 1e-5);

    // 3. Normalize
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        let val = data[row * dims.cols + j];
        data[row * dims.cols + j] = (val - mean) * inv_std;
    }
}

@compute @workgroup_size(64)
fn softmax_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= dims.rows) {
        return;
    }

    // 1. Find Max for stability
    var max_val: f32 = -1e20; // -infinity
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        let val = data[row * dims.cols + j];
        if (val > max_val) {
            max_val = val;
        }
    }

    // 2. Calculate Exponentials and Sum
    var sum_exp: f32 = 0.0;
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        let val = exp(data[row * dims.cols + j] - max_val);
        data[row * dims.cols + j] = val; // Store exp temporarily
        sum_exp = sum_exp + val;
    }

    // 3. Normalize
    for (var j: u32 = 0u; j < dims.cols; j = j + 1u) {
        data[row * dims.cols + j] = data[row * dims.cols + j] / sum_exp;
    }
}
