struct Dimensions {
    len: u32,
    padding1: u32,
    padding2: u32,
    padding3: u32,
}

struct UpdateParams {
    len: u32,
    scale: f32,
}

@group(0) @binding(0) var<uniform> dims: Dimensions;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> out: array<f32>;

@compute @workgroup_size(64)
fn add_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= dims.len) {
        return;
    }
    out[idx] = a[idx] + b[idx];
}

@group(0) @binding(1) var<storage, read_write> assign_a: array<f32>;
@group(0) @binding(2) var<storage, read> assign_b: array<f32>;

@compute @workgroup_size(64)
fn add_assign_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= dims.len) {
        return;
    }
    assign_a[idx] = assign_a[idx] + assign_b[idx];
}

@group(0) @binding(0) var<uniform> params: UpdateParams;
@group(0) @binding(1) var<storage, read_write> data: array<f32>;

@compute @workgroup_size(64)
fn scale_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= params.len) {
        return;
    }
    data[idx] = data[idx] * params.scale;
}
