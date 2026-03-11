struct Dimensions {
    rows: u32,
    cols: u32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> input: array<f32>;
@group(0) @binding(2) var<storage, read_write> output: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    let j = global_id.y;

    if (i >= dims.rows || j >= dims.cols) {
        return;
    }

    output[j * dims.rows + i] = input[i * dims.cols + j];
}
