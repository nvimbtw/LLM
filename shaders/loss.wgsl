struct LossParams {
    batch_size: u32,
    vocab_size: u32,
    seq_len: u32,
    padding: u32,
}

@group(0) @binding(0)
var<uniform> params: LossParams;

@group(0) @binding(1)
var<storage, read> logits: array<f32>;

@group(0) @binding(2)
var<storage, read> target_token_ids: array<u32>;

@group(0) @binding(3)
var<storage, read_write> loss_output: array<f32>;

@group(0) @binding(4)
var<storage, read_write> grad_logits: array<f32>;

var<workgroup> shared_max: array<f32, 256>;
var<workgroup> shared_sum: array<f32, 256>;

@compute
@workgroup_size(256)
fn cross_entropy_forward_main(
    @builtin(workgroup_id) workgroup_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>
) {
    let batch_idx = workgroup_id.x;
    if (batch_idx >= params.batch_size) {
        return;
    }

    let tid = local_id.x;
    let vocab_size = params.vocab_size;
    let start_idx = batch_idx * vocab_size;
    let target_token_id = target_token_ids[batch_idx];

    // 1. Parallel Max Reduction
    var local_max: f32 = -1e38;
    for (var i = tid; i < vocab_size; i += 256u) {
        local_max = max(local_max, logits[start_idx + i]);
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

    // 2. Parallel Sum Reduction (exp)
    var local_sum: f32 = 0.0;
    for (var i = tid; i < vocab_size; i += 256u) {
        let val = exp(logits[start_idx + i] - row_max);
        grad_logits[start_idx + i] = val; // Store temporarily
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

    // 3. Normalize and compute Grad
    if (row_sum > 0.0) {
        for (var i = tid; i < vocab_size; i += 256u) {
            let prob = grad_logits[start_idx + i] / row_sum;
            var grad = prob;
            if (i == target_token_id) {
                grad -= 1.0;
            }
            grad_logits[start_idx + i] = grad;
        }

        // Only one thread computes the loss for this sample
        if (tid == 0u) {
            if (target_token_id < vocab_size) {
                let prob_target = exp(logits[start_idx + target_token_id] - row_max) / row_sum;
                loss_output[batch_idx] = -log(max(prob_target, 1e-10));
            } else {
                loss_output[batch_idx] = 0.0;
            }
        }
    } else {
        if (tid == 0u) {
            loss_output[batch_idx] = 0.0;
        }
    }
}
