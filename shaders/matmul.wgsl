struct MatrixDimensions {
    a_rows: u32,
    a_cols: u32,
    b_cols: u32,
    batch_size: u32,
}

@group(0) @binding(0) var<uniform> dims: MatrixDimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;
    let batch = global_id.z;

    if (row >= dims.a_rows || col >= dims.b_cols || batch >= dims.batch_size) {
        return;
    }

    let batch_offset_a = batch * dims.a_rows * dims.a_cols;
    let batch_offset_b = batch * dims.a_cols * dims.b_cols;
    let batch_offset_c = batch * dims.a_rows * dims.b_cols;

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.a_cols; k = k + 1u) {
        let idx_a = batch_offset_a + row * dims.a_cols + k;
        let idx_b = batch_offset_b + k * dims.b_cols + col;
        sum = sum + a[idx_a] * b[idx_b];
    }

    c[batch_offset_c + row * dims.b_cols + col] = sum;
}
