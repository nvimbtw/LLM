struct PermuteParams {
    s: u32,
    h: u32,
    d: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: PermuteParams;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(64)
fn permute_021_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.s * params.h * params.d;
    if (idx >= total) {
        return;
    }

    // Input layout: [S, H, D]
    let s = idx / (params.h * params.d);
    let rem = idx % (params.h * params.d);
    let h = rem / params.d;
    let d = rem % params.d;

    // Output layout: [H, S, D]
    let out_idx = h * (params.s * params.d) + s * params.d + d;
    output[out_idx] = input[idx];
}

@compute @workgroup_size(64)
fn permute_102_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.x;
    let total = params.s * params.h * params.d;
    if (idx >= total) {
        return;
    }

    // Input layout: [H, S, D]
    let h = idx / (params.s * params.d);
    let rem = idx % (params.s * params.d);
    let s = rem / params.d;
    let d = rem % params.d;

    // Output layout: [S, H, D]
    let out_idx = s * (params.h * params.d) + h * params.d + d;
    output[out_idx] = input[idx];
}
