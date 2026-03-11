struct Dimensions {
    rows: u32,
    cols: u32,
    batch_size: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;
    let b = global_id.z;

    if (i >= dims.rows || j >= dims.cols || b >= dims.batch_size) {
        return;
    }

    let batch_size_flat = dims.rows * dims.cols;
    let input_idx = b * batch_size_flat + i * dims.cols + j;
    let output_idx = b * batch_size_flat + j * dims.rows + i;

    output[output_idx] = input[input_idx];
}
