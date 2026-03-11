struct Params {
    len: u32,
    lr: f32,
}

struct AdamParams {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    correction1: f32,
    correction2: f32,
    padding: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(0) var<uniform> adam_params: AdamParams; // Aliased binding 0, will use different pipeline

@group(0) @binding(1) var<storage, read_write> weights: array<f32>;
@group(0) @binding(2) var<storage, read> grads_f32: array<f32>;
@group(0) @binding(3) var<storage, read> grads_i32: array<i32>;
@group(0) @binding(4) var<storage, read_write> m: array<f32>;
@group(0) @binding(5) var<storage, read_write> v: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= params.len) {
        return;
    }
    weights[idx] = weights[idx] - params.lr * grads_f32[idx];
}

@compute @workgroup_size(64)
fn update_fixed_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= params.len) {
        return;
    }
    let float_grad = f32(grads_i32[idx]) / 1000000.0;
    weights[idx] = weights[idx] - params.lr * float_grad;
}

@compute @workgroup_size(64)
fn adam_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let idx = global_id.y * 65536u + global_id.x;
    if (idx >= adam_params.len) {
        return;
    }

    let g = grads_f32[idx];
    let m_t = adam_params.beta1 * m[idx] + (1.0 - adam_params.beta1) * g;
    let v_t = adam_params.beta2 * v[idx] + (1.0 - adam_params.beta2) * g * g;

    m[idx] = m_t;
    v[idx] = v_t;

    let m_hat = m_t / adam_params.correction1;
    let v_hat = v_t / adam_params.correction2;

    weights[idx] = weights[idx] - adam_params.lr * m_hat / (sqrt(v_hat) + adam_params.epsilon);
}
