use rand::prelude::*;
use crate::io::*;
use crate::train::backend::{WgpuBackend, GpuTensor};
use std::sync::Arc;

pub fn init_transformer(backend: &WgpuBackend) -> (Transformer, GpuTensor, GpuTensor, usize) {
    let vocab = read_vocab("data/pairs.bin").expect("Failed to read pairs");
    let vocab_size = vocab.len() + 256;
    let dimensions = 128;
    let context_window = 1024;

    let embedding_table_raw = load_matrix("data/embedding_table.bin")
        .unwrap_or_else(|_| new_table(vocab_size, dimensions));
    let positional_table_raw = load_matrix("data/positional_table.bin")
        .unwrap_or_else(|_| new_table(context_window, dimensions));

    let embedding_table = GpuTensor::from_cpu(backend, &embedding_table_raw);
    let positional_table = GpuTensor::from_cpu(backend, &positional_table_raw);

    let mut transformer = Transformer::new(backend, dimensions);
    
    if let Ok(m) = load_matrix("data/w_q.bin") { transformer.w_q = GpuTensor::from_cpu(backend, &m); }
    if let Ok(m) = load_matrix("data/w_k.bin") { transformer.w_k = GpuTensor::from_cpu(backend, &m); }
    if let Ok(m) = load_matrix("data/w_v.bin") { transformer.w_v = GpuTensor::from_cpu(backend, &m); }
    if let Ok(m) = load_matrix("data/w_o.bin") { transformer.w_o = GpuTensor::from_cpu(backend, &m); }
    if let Ok(m) = load_matrix("data/w_ff1.bin") { transformer.w_ff1 = GpuTensor::from_cpu(backend, &m); }
    if let Ok(m) = load_matrix("data/w_ff2.bin") { transformer.w_ff2 = GpuTensor::from_cpu(backend, &m); }

    (transformer, embedding_table, positional_table, dimensions)
}

pub fn new_table(length: usize, dimensions: usize) -> Vec<Vec<f32>> {
    let mut rng = rand::rng();
    let mut output = vec![vec![0.0f32; dimensions]; length];
    let limit = (6.0 / (length + dimensions) as f32).sqrt();
    for i in 0..length {
        for j in 0..dimensions {
            output[i][j] = rng.random_range(-limit..limit);
        }
    }
    output
}

pub struct Transformer {
    pub w_q: GpuTensor,
    pub w_k: GpuTensor,
    pub w_v: GpuTensor,
    pub w_o: GpuTensor,
    pub w_ff1: GpuTensor,
    pub w_ff2: GpuTensor,
    pub grad_w_q: GpuTensor,
    pub grad_w_k: GpuTensor,
    pub grad_w_v: GpuTensor,
    pub grad_w_o: GpuTensor,
    pub grad_w_ff1: GpuTensor,
    pub grad_w_ff2: GpuTensor,
}

pub struct ForwardState {
    pub tokens: Arc<wgpu::Buffer>,
    pub input: GpuTensor,
    pub q: GpuTensor,
    pub k: GpuTensor,
    pub v: GpuTensor,
    pub probs: GpuTensor,
    pub attn_output: GpuTensor,
    pub ff1_output: GpuTensor,
    pub relu_output: GpuTensor,
}

impl Transformer {
    pub fn new(backend: &WgpuBackend, dimensions: usize) -> Self {
        let ff_hidden = dimensions * 4;
        let w_q_raw = new_table(dimensions, dimensions);
        let w_k_raw = new_table(dimensions, dimensions);
        let w_v_raw = new_table(dimensions, dimensions);
        let w_o_raw = new_table(dimensions, dimensions);
        let w_ff1_raw = new_table(dimensions, ff_hidden);
        let w_ff2_raw = new_table(ff_hidden, dimensions);

        let grad_zero = vec![vec![0.0f32; dimensions]; dimensions];
        let grad_ff1_zero = vec![vec![0.0f32; ff_hidden]; dimensions];
        let grad_ff2_zero = vec![vec![0.0f32; dimensions]; ff_hidden];

        Transformer {
            w_q: GpuTensor::from_cpu(backend, &w_q_raw),
            w_k: GpuTensor::from_cpu(backend, &w_k_raw),
            w_v: GpuTensor::from_cpu(backend, &w_v_raw),
            w_o: GpuTensor::from_cpu(backend, &w_o_raw),
            w_ff1: GpuTensor::from_cpu(backend, &w_ff1_raw),
            w_ff2: GpuTensor::from_cpu(backend, &w_ff2_raw),
            grad_w_q: GpuTensor::from_cpu(backend, &grad_zero),
            grad_w_k: GpuTensor::from_cpu(backend, &grad_zero),
            grad_w_v: GpuTensor::from_cpu(backend, &grad_zero),
            grad_w_o: GpuTensor::from_cpu(backend, &grad_zero),
            grad_w_ff1: GpuTensor::from_cpu(backend, &grad_ff1_zero),
            grad_w_ff2: GpuTensor::from_cpu(backend, &grad_ff2_zero),
        }
    }

    pub fn forward(&self, backend: &WgpuBackend, tokens: Arc<wgpu::Buffer>, embedding: &GpuTensor, positional: &GpuTensor, seq_len: usize, _dimensions: usize) -> (GpuTensor, ForwardState) {
        let input = backend.run_embedding_forward(&tokens, embedding, positional, seq_len);
        
        let q = backend.run_matmul(&input, &self.w_q);
        let k = backend.run_matmul(&input, &self.w_k);
        let v = backend.run_matmul(&input, &self.w_v);

        let k_t = backend.run_transpose(&k);
        let mut scores = backend.run_matmul(&q, &k_t);
        backend.run_scale_mask(&mut scores);
        
        let mut probs = scores;
        backend.run_softmax(&mut probs);

        let attn = backend.run_matmul(&probs, &v);
        let attn_output_linear = backend.run_matmul(&attn, &self.w_o);
        let attn_output = backend.run_add(&input, &attn_output_linear);

        let ff1_output = backend.run_matmul(&attn_output, &self.w_ff1);
        let mut relu_output = ff1_output.clone_on_gpu(backend);
        backend.run_relu(&mut relu_output);
        
        let ffn_output_linear = backend.run_matmul(&relu_output, &self.w_ff2);
        let output = backend.run_add(&attn_output, &ffn_output_linear);

        (
            output,
            ForwardState {
                tokens, input, q, k, v, probs, attn_output, ff1_output, relu_output,
            },
        )
    }

    pub fn backward(
        &mut self,
        backend: &WgpuBackend,
        d_output: &GpuTensor,
        state: &ForwardState,
        grad_emb: &mut GpuTensor,
        grad_pos: &mut GpuTensor,
        _dimensions: usize,
    ) -> GpuTensor {
        let relu_output_t = backend.run_transpose(&state.relu_output);
        let d_w_ff2 = backend.run_matmul(&relu_output_t, d_output);
        backend.run_add_to_grad(&mut self.grad_w_ff2, &d_w_ff2);

        let w_ff2_t = backend.run_transpose(&self.w_ff2);
        let d_relu_output = backend.run_matmul(d_output, &w_ff2_t);

        let mut d_ff1_output = d_relu_output.clone_on_gpu(backend);
        backend.run_relu_backward(&state.ff1_output, &mut d_ff1_output);

        let attn_output_t = backend.run_transpose(&state.attn_output);
        let d_w_ff1 = backend.run_matmul(&attn_output_t, &d_ff1_output);
        backend.run_add_to_grad(&mut self.grad_w_ff1, &d_w_ff1);

        let w_ff1_t = backend.run_transpose(&self.w_ff1);
        let d_attn_output_from_ffn = backend.run_matmul(&d_ff1_output, &w_ff1_t);
        let d_attn_output = backend.run_add(d_output, &d_attn_output_from_ffn);

        let w_o_t = backend.run_transpose(&self.w_o);
        let d_attn = backend.run_matmul(&d_attn_output, &w_o_t);

        let attn = backend.run_matmul(&state.probs, &state.v);
        let attn_t = backend.run_transpose(&attn);
        let d_w_o = backend.run_matmul(&attn_t, &d_attn_output);
        backend.run_add_to_grad(&mut self.grad_w_o, &d_w_o);

        let v_t = backend.run_transpose(&state.v);
        let d_probs = backend.run_matmul(&d_attn, &v_t);

        let probs_t = backend.run_transpose(&state.probs);
        let d_v = backend.run_matmul(&probs_t, &d_attn);

        let d_scores = backend.run_softmax_backward(&state.probs, &d_probs);

        let d_q = backend.run_matmul(&d_scores, &state.k);
        let d_scores_t = backend.run_transpose(&d_scores);
        let d_k = backend.run_matmul(&d_scores_t, &state.q);

        let input_t = backend.run_transpose(&state.input);
        let d_w_q = backend.run_matmul(&input_t, &d_q);
        backend.run_add_to_grad(&mut self.grad_w_q, &d_w_q);

        let d_w_k = backend.run_matmul(&input_t, &d_k);
        backend.run_add_to_grad(&mut self.grad_w_k, &d_w_k);

        let d_w_v = backend.run_matmul(&input_t, &d_v);
        backend.run_add_to_grad(&mut self.grad_w_v, &d_w_v);

        let w_q_t = backend.run_transpose(&self.w_q);
        let w_k_t = backend.run_transpose(&self.w_k);
        let w_v_t = backend.run_transpose(&self.w_v);

        let d_input_q = backend.run_matmul(&d_q, &w_q_t);
        let d_input_k = backend.run_matmul(&d_k, &w_k_t);
        let d_input_v = backend.run_matmul(&d_v, &w_v_t);

        let mut d_input = backend.run_add(&d_attn_output, &d_input_q);
        d_input = backend.run_add(&d_input, &d_input_k);
        d_input = backend.run_add(&d_input, &d_input_v);

        backend.run_embedding_backward(&state.tokens, &d_input, grad_emb, grad_pos);

        d_input
    }

    pub fn zero_grad(&mut self, backend: &WgpuBackend) {
        self.grad_w_q.zero(backend);
        self.grad_w_k.zero(backend);
        self.grad_w_v.zero(backend);
        self.grad_w_o.zero(backend);
        self.grad_w_ff1.zero(backend);
        self.grad_w_ff2.zero(backend);
    }

    pub fn update_weights(&mut self, backend: &WgpuBackend, learning_rate: f32) {
        backend.run_update(&mut self.w_q, &self.grad_w_q, learning_rate);
        backend.run_update(&mut self.w_k, &self.grad_w_k, learning_rate);
        backend.run_update(&mut self.w_v, &self.grad_w_v, learning_rate);
        backend.run_update(&mut self.w_o, &self.grad_w_o, learning_rate);
        backend.run_update(&mut self.w_ff1, &self.grad_w_ff1, learning_rate);
        backend.run_update(&mut self.w_ff2, &self.grad_w_ff2, learning_rate);
    }
}
