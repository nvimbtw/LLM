struct MatrixDimensions {
    a_rows: u32,
    a_cols: u32,
    b_cols: u32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> dims: MatrixDimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> c: array<f32>;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= dims.a_rows || col >= dims.b_cols) {
        return;
    }

    var sum: f32 = 0.0;
    for (var k: u32 = 0u; k < dims.a_cols; k = k + 1u) {
        sum = sum + a[row * dims.a_cols + k] * b[k * dims.b_cols + col];
    }

    c[row * dims.b_cols + col] = sum;
}
