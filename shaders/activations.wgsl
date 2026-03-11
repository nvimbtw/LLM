struct Dimensions {
    rows: u32,
    cols: u32,
    padding1: u32,
    padding2: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute @workgroup_size(64)
fn relu_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= dims.rows * dims.cols) {
        return;
    }
    data[idx] = max(0.0, data[idx]);
}

@compute @workgroup_size(256)
fn softmax_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = workgroup_id.x;
    if (row >= dims.rows) {
        return;
    }

    let tid = local_id.x;
    let cols = dims.cols;

    // 1. Find Max for stability (Parallel Reduction)
    var local_max: f32 = -1e38;
    for (var j = tid; j < cols; j += 256u) {
        local_max = max(local_max, data[row * cols + j]);
    }
    shared_max[tid] = local_max;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
        }
        workgroupBarrier();
    }
    let row_max = shared_max[0];

    // 2. Calculate Exponentials and Sum (Parallel Reduction)
    var local_sum: f32 = 0.0;
    for (var j = tid; j < cols; j += 256u) {
        let val = exp(data[row * cols + j] - row_max);
        data[row * cols + j] = val; // Store exp temporarily
        local_sum += val;
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }
    let row_sum = shared_sum[0];

    // 3. Normalize
    if (row_sum > 0.0) {
        for (var j = tid; j < cols; j += 256u) {
            data[row * cols + j] /= row_sum;
        }
    }
}

@compute @workgroup_size(256)
fn layer_norm_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let row = workgroup_id.x;
    if (row >= dims.rows) {
        return;
    }

    let tid = local_id.x;
    let cols = dims.cols;

    // 1. Calculate Mean
    var local_sum: f32 = 0.0;
    for (var j = tid; j < cols; j += 256u) {
        local_sum += data[row * cols + j];
    }
    shared_sum[tid] = local_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }
    let mean = shared_sum[0] / f32(cols);

    // 2. Calculate Variance
    var local_var_sum: f32 = 0.0;
    for (var j = tid; j < cols; j += 256u) {
        let diff = data[row * cols + j] - mean;
        local_var_sum += diff * diff;
    }
    shared_sum[tid] = local_var_sum; // Reuse shared_sum
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            shared_sum[tid] += shared_sum[tid + s];
        }
        workgroupBarrier();
    }
    let variance = shared_sum[0] / f32(cols);
    let inv_std = 1.0 / sqrt(variance + 1e-5);

    // 3. Normalize
    for (var j = tid; j < cols; j += 256u) {
        data[row * cols + j] = (data[row * cols + j] - mean) * inv_std;
    }
}
