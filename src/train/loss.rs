pub fn cross_entropy_loss(logits: &[f32], target_token_id: usize) -> (f32, Vec<f32>) {
    if target_token_id >= logits.len() {
        panic!(
            "Target token id {} is out of bounds for logits of length {}",
            target_token_id,
            logits.len()
        );
    }
    let mut max_logit = f32::NEG_INFINITY;
    for &l in logits {
        if l > max_logit {
            max_logit = l;
        }
    }

    let mut sum_exp = 0.0;
    let mut probs = vec![0.0; logits.len()];
    for i in 0..logits.len() {
        probs[i] = (logits[i] - max_logit).exp();
        sum_exp += probs[i];
    }

    for i in 0..logits.len() {
        probs[i] /= sum_exp;
    }

    let loss = -probs[target_token_id].ln();

    let mut grad_logits = probs;
    grad_logits[target_token_id] -= 1.0;

    (loss, grad_logits)
}
