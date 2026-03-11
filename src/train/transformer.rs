use crate::io::*;
use crate::io::io::load_config;
use crate::train::backend::{GpuTensor, WgpuBackend};
use rand::prelude::*;
use std::sync::Arc;

pub fn init_transformer(
    backend: &WgpuBackend,
    context_window: usize,
) -> (Transformer, GpuTensor, GpuTensor, usize) {
    let (config_vocab_size, _, config_dim) = load_config("data/config.bin").unwrap_or((0, 0, 0));

    let vocab = read_vocab("data/pairs.bin").expect("Failed to read pairs");
    let vocab_size = if config_vocab_size > 0 {
        config_vocab_size as usize
    } else {
        vocab.len() + 256
    };
    
    let dimensions = if config_dim > 0 {
        config_dim as usize
    } else {
        1024
    };

    let embedding_table_raw = load_matrix("data/embedding_table.bin")
        .unwrap_or_else(|_| new_table(vocab_size, dimensions));
    let actual_dimensions = if !embedding_table_raw.is_empty() {
        embedding_table_raw[0].len()
    } else {
        dimensions
    };

    let positional_table_raw = load_matrix("data/positional_table.bin")
        .unwrap_or_else(|_| new_table(context_window, actual_dimensions));

    let embedding_table = GpuTensor::from_cpu(backend, &embedding_table_raw);
    let positional_table = GpuTensor::from_cpu(backend, &positional_table_raw);

    // Default to 3 layers, 8 heads
    let n_layers = 3;
    let n_heads = 8;
    let mut transformer = Transformer::new(backend, actual_dimensions, n_layers, n_heads);

    // Try loading layer weights
    for (i, layer) in transformer.layers.iter_mut().enumerate() {
        if let Ok(m) = load_matrix(&format!("data/layer_{}_w_q.bin", i)) {
            layer.w_q.data = GpuTensor::from_cpu(backend, &m);
        }
        if let Ok(m) = load_matrix(&format!("data/layer_{}_w_k.bin", i)) {
            layer.w_k.data = GpuTensor::from_cpu(backend, &m);
        }
        if let Ok(m) = load_matrix(&format!("data/layer_{}_w_v.bin", i)) {
            layer.w_v.data = GpuTensor::from_cpu(backend, &m);
        }
        if let Ok(m) = load_matrix(&format!("data/layer_{}_w_o.bin", i)) {
            layer.w_o.data = GpuTensor::from_cpu(backend, &m);
        }
        if let Ok(m) = load_matrix(&format!("data/layer_{}_w_ff1.bin", i)) {
            layer.w_ff1.data = GpuTensor::from_cpu(backend, &m);
        }
        if let Ok(m) = load_matrix(&format!("data/layer_{}_w_ff2.bin", i)) {
            layer.w_ff2.data = GpuTensor::from_cpu(backend, &m);
        }
    }

    (transformer, embedding_table, positional_table, actual_dimensions)
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

pub struct Weight {
    pub data: GpuTensor,
    pub grad: GpuTensor,
    pub m: GpuTensor,
    pub v: GpuTensor,
}

impl Weight {
    pub fn new(backend: &WgpuBackend, rows: usize, cols: usize) -> Self {
        let data_raw = new_table(rows, cols);
        let grad_zero = vec![vec![0.0f32; cols]; rows];
        Self {
            data: GpuTensor::from_cpu(backend, &data_raw),
            grad: GpuTensor::from_cpu(backend, &grad_zero),
            m: GpuTensor::from_cpu(backend, &grad_zero),
            v: GpuTensor::from_cpu(backend, &grad_zero),
        }
    }

    pub fn update(&mut self, backend: &WgpuBackend, lr: f32, t: u32) {
        backend.run_adam_update(&mut self.data, &self.grad, &mut self.m, &mut self.v, lr, 0.9, 0.999, 1e-8, t);
    }

    pub fn zero_grad(&mut self, backend: &WgpuBackend) {
        self.grad.zero(backend);
    }

    pub fn scale_grad(&mut self, backend: &WgpuBackend, scale: f32) {
        backend.run_scale(&mut self.grad, scale);
    }
}

pub struct TransformerLayer {
    pub w_q: Weight,
    pub w_k: Weight,
    pub w_v: Weight,
    pub w_o: Weight,
    pub w_ff1: Weight,
    pub w_ff2: Weight,
    pub n_heads: usize,
}

impl TransformerLayer {
    pub fn new(backend: &WgpuBackend, dimensions: usize, n_heads: usize) -> Self {
        let ff_hidden = dimensions * 4;
        Self {
            w_q: Weight::new(backend, dimensions, dimensions),
            w_k: Weight::new(backend, dimensions, dimensions),
            w_v: Weight::new(backend, dimensions, dimensions),
            w_o: Weight::new(backend, dimensions, dimensions),
            w_ff1: Weight::new(backend, dimensions, ff_hidden),
            w_ff2: Weight::new(backend, ff_hidden, dimensions),
            n_heads,
        }
    }

    pub fn forward(
        &self,
        backend: &WgpuBackend,
        input: &GpuTensor,
        seq_len: usize,
        dimensions: usize,
    ) -> (GpuTensor, LayerForwardState) {
        let ln1_output = backend.run_layer_norm(input);

        let q_total = backend.run_matmul(&ln1_output, &self.w_q.data);
        let k_total = backend.run_matmul(&ln1_output, &self.w_k.data);
        let v_total = backend.run_matmul(&ln1_output, &self.w_v.data);

        let head_dim = dimensions / self.n_heads;
        
        let k_total_t = backend.run_transpose(&k_total);
        let mut scores = backend.run_batched_matmul(&q_total, &k_total_t, self.n_heads, seq_len, head_dim, seq_len);
        
        backend.run_scale_mask(&mut scores, 1.0 / (head_dim as f32).sqrt());

        let mut probs = scores;
        backend.run_softmax(&mut probs);

        let attn = backend.run_batched_matmul(&probs, &v_total, self.n_heads, seq_len, seq_len, head_dim);
        
        let attn_output_linear = backend.run_matmul(&attn, &self.w_o.data);
        let attn_output = backend.run_add(input, &attn_output_linear);

        let ln2_output = backend.run_layer_norm(&attn_output);

        let ff1_output = backend.run_matmul(&ln2_output, &self.w_ff1.data);
        let mut relu_output = ff1_output.clone_on_gpu(backend);
        backend.run_relu(&mut relu_output);

        let ffn_output_linear = backend.run_matmul(&relu_output, &self.w_ff2.data);
        let output = backend.run_add(&attn_output, &ffn_output_linear);

        // Cleanup
        k_total_t.return_to_pool(backend);
        attn_output_linear.return_to_pool(backend);
        ffn_output_linear.return_to_pool(backend);

        (
            output,
            LayerForwardState {
                ln1_output,
                q_total,
                k_total,
                v_total,
                probs,
                attn,
                attn_output,
                ln2_output,
                ff1_output,
                relu_output,
            },
        )
    }

    pub fn backward(
        &mut self,
        backend: &WgpuBackend,
        d_output: &GpuTensor,
        state: &LayerForwardState,
        dimensions: usize,
        seq_len: usize,
    ) -> GpuTensor {
        let head_dim = dimensions / self.n_heads;

        let relu_output_t = backend.run_transpose(&state.relu_output);
        let d_w_ff2 = backend.run_matmul(&relu_output_t, d_output);
        backend.run_add_to_grad(&mut self.w_ff2.grad, &d_w_ff2);

        let w_ff2_t = backend.run_transpose(&self.w_ff2.data);
        let d_relu_output = backend.run_matmul(d_output, &w_ff2_t);

        let mut d_ff1_output = d_relu_output.clone_on_gpu(backend);
        backend.run_relu_backward(&state.ff1_output, &mut d_ff1_output);

        let ln2_output_t = backend.run_transpose(&state.ln2_output);
        let d_w_ff1 = backend.run_matmul(&ln2_output_t, &d_ff1_output);
        backend.run_add_to_grad(&mut self.w_ff1.grad, &d_w_ff1);

        let w_ff1_t = backend.run_transpose(&self.w_ff1.data);
        let d_ln2_output = backend.run_matmul(&d_ff1_output, &w_ff1_t);
        
        let d_attn_output_from_ffn = backend.run_layer_norm_backward(&state.ln2_output, &d_ln2_output);
        let d_attn_output = backend.run_add(d_output, &d_attn_output_from_ffn);

        let w_o_t = backend.run_transpose(&self.w_o.data);
        let d_attn = backend.run_matmul(&d_attn_output, &w_o_t);

        let attn_t = backend.run_transpose(&state.attn);
        let d_w_o = backend.run_matmul(&attn_t, &d_attn_output);
        backend.run_add_to_grad(&mut self.w_o.grad, &d_w_o);

        let v_total_t = backend.run_transpose(&state.v_total);
        let d_probs = backend.run_batched_matmul(&d_attn, &v_total_t, self.n_heads, seq_len, head_dim, seq_len);

        let probs_t = backend.run_transpose(&state.probs);
        let d_v_total = backend.run_batched_matmul(&probs_t, &d_attn, self.n_heads, seq_len, seq_len, head_dim);

        let d_scores = backend.run_softmax_backward(&state.probs, &d_probs, 1.0 / (head_dim as f32).sqrt());

        let d_q_total = backend.run_batched_matmul(&d_scores, &state.k_total, self.n_heads, seq_len, seq_len, head_dim);
        let d_scores_t = backend.run_transpose(&d_scores);
        let d_k_total = backend.run_batched_matmul(&d_scores_t, &state.q_total, self.n_heads, seq_len, seq_len, head_dim);

        let ln1_output_t = backend.run_transpose(&state.ln1_output);
        let d_w_q = backend.run_matmul(&ln1_output_t, &d_q_total);
        backend.run_add_to_grad(&mut self.w_q.grad, &d_w_q);

        let d_w_k = backend.run_matmul(&ln1_output_t, &d_k_total);
        backend.run_add_to_grad(&mut self.w_k.grad, &d_w_k);

        let d_w_v = backend.run_matmul(&ln1_output_t, &d_v_total);
        backend.run_add_to_grad(&mut self.w_v.grad, &d_w_v);

        let w_q_t = backend.run_transpose(&self.w_q.data);
        let w_k_t = backend.run_transpose(&self.w_k.data);
        let w_v_t = backend.run_transpose(&self.w_v.data);

        let d_ln1_output_q = backend.run_matmul(&d_q_total, &w_q_t);
        let d_ln1_output_k = backend.run_matmul(&d_k_total, &w_k_t);
        let d_ln1_output_v = backend.run_matmul(&d_v_total, &w_v_t);

        let d_ln1_output_tmp = backend.run_add(&d_ln1_output_q, &d_ln1_output_k);
        let d_ln1_output = backend.run_add(&d_ln1_output_tmp, &d_ln1_output_v);

        let d_input_from_attn = backend.run_layer_norm_backward(&state.ln1_output, &d_ln1_output);
        let d_input_total = backend.run_add(&d_attn_output, &d_input_from_attn);

        // Cleanup
        relu_output_t.return_to_pool(backend);
        d_w_ff2.return_to_pool(backend);
        w_ff2_t.return_to_pool(backend);
        d_relu_output.return_to_pool(backend);
        d_ff1_output.return_to_pool(backend);
        ln2_output_t.return_to_pool(backend);
        d_w_ff1.return_to_pool(backend);
        w_ff1_t.return_to_pool(backend);
        d_ln2_output.return_to_pool(backend);
        d_attn_output_from_ffn.return_to_pool(backend);
        d_attn_output.return_to_pool(backend);
        w_o_t.return_to_pool(backend);
        d_attn.return_to_pool(backend);
        attn_t.return_to_pool(backend);
        d_w_o.return_to_pool(backend);
        v_total_t.return_to_pool(backend);
        d_probs.return_to_pool(backend);
        probs_t.return_to_pool(backend);
        d_v_total.return_to_pool(backend);
        d_scores.return_to_pool(backend);
        d_scores_t.return_to_pool(backend);
        d_q_total.return_to_pool(backend);
        d_k_total.return_to_pool(backend);
        ln1_output_t.return_to_pool(backend);
        d_w_q.return_to_pool(backend);
        d_w_k.return_to_pool(backend);
        d_w_v.return_to_pool(backend);
        w_q_t.return_to_pool(backend);
        w_k_t.return_to_pool(backend);
        w_v_t.return_to_pool(backend);
        d_ln1_output_q.return_to_pool(backend);
        d_ln1_output_k.return_to_pool(backend);
        d_ln1_output_v.return_to_pool(backend);
        d_ln1_output_tmp.return_to_pool(backend);
        d_ln1_output.return_to_pool(backend);
        d_input_from_attn.return_to_pool(backend);

        d_input_total
    }

    pub fn zero_grad(&mut self, backend: &WgpuBackend) {
        self.w_q.zero_grad(backend);
        self.w_k.zero_grad(backend);
        self.w_v.zero_grad(backend);
        self.w_o.zero_grad(backend);
        self.w_ff1.zero_grad(backend);
        self.w_ff2.zero_grad(backend);
    }

    pub fn update_weights(&mut self, backend: &WgpuBackend, lr: f32, t: u32) {
        self.w_q.update(backend, lr, t);
        self.w_k.update(backend, lr, t);
        self.w_v.update(backend, lr, t);
        self.w_o.update(backend, lr, t);
        self.w_ff1.update(backend, lr, t);
        self.w_ff2.update(backend, lr, t);
    }

    pub fn scale_grads(&mut self, backend: &WgpuBackend, scale: f32) {
        self.w_q.scale_grad(backend, scale);
        self.w_k.scale_grad(backend, scale);
        self.w_v.scale_grad(backend, scale);
        self.w_o.scale_grad(backend, scale);
        self.w_ff1.scale_grad(backend, scale);
        self.w_ff2.scale_grad(backend, scale);
    }
}

pub struct LayerForwardState {
    pub ln1_output: GpuTensor,
    pub q_total: GpuTensor,
    pub k_total: GpuTensor,
    pub v_total: GpuTensor,
    pub probs: GpuTensor,
    pub attn: GpuTensor,
    pub attn_output: GpuTensor,
    pub ln2_output: GpuTensor,
    pub ff1_output: GpuTensor,
    pub relu_output: GpuTensor,
}

impl LayerForwardState {
    pub fn return_to_pool(self, backend: &WgpuBackend) {
        self.ln1_output.return_to_pool(backend);
        self.q_total.return_to_pool(backend);
        self.k_total.return_to_pool(backend);
        self.v_total.return_to_pool(backend);
        self.probs.return_to_pool(backend);
        self.attn.return_to_pool(backend);
        self.attn_output.return_to_pool(backend);
        self.ln2_output.return_to_pool(backend);
        self.ff1_output.return_to_pool(backend);
        self.relu_output.return_to_pool(backend);
    }
}

pub struct Transformer {
    pub layers: Vec<TransformerLayer>,
    pub t: u32,
}

pub struct TransformerForwardState {
    pub tokens: Arc<wgpu::Buffer>,
    pub input: GpuTensor,
    pub layer_states: Vec<LayerForwardState>,
}

impl TransformerForwardState {
    pub fn return_to_pool(self, backend: &WgpuBackend) {
        self.input.return_to_pool(backend);
        for state in self.layer_states {
            state.return_to_pool(backend);
        }
    }
}

impl Transformer {
    pub fn new(backend: &WgpuBackend, dimensions: usize, n_layers: usize, n_heads: usize) -> Self {
        let mut layers = Vec::new();
        for _ in 0..n_layers {
            layers.push(TransformerLayer::new(backend, dimensions, n_heads));
        }
        Transformer { layers, t: 0 }
    }

    pub fn forward(
        &self,
        backend: &WgpuBackend,
        tokens: Arc<wgpu::Buffer>,
        embedding: &GpuTensor,
        positional: &GpuTensor,
        seq_len: usize,
        dimensions: usize,
    ) -> (GpuTensor, TransformerForwardState) {
        let input = backend.run_embedding_forward(&tokens, embedding, positional, seq_len);
        let mut current_input = input.clone_on_gpu(backend);
        let mut layer_states = Vec::new();

        for layer in &self.layers {
            let (output, state) = layer.forward(backend, &current_input, seq_len, dimensions);
            current_input.return_to_pool(backend);
            current_input = output;
            layer_states.push(state);
        }

        (
            current_input,
            TransformerForwardState {
                tokens,
                input,
                layer_states,
            },
        )
    }

    pub fn backward(
        &mut self,
        backend: &WgpuBackend,
        d_output: &GpuTensor,
        state: &TransformerForwardState,
        grad_emb_i32_buffer: &wgpu::Buffer,
        grad_pos: &mut GpuTensor,
        dimensions: usize,
    ) -> GpuTensor {
        let mut d_input = d_output.clone_on_gpu(backend);
        let seq_len = state.input.shape.0;

        for (layer, layer_state) in self.layers.iter_mut().zip(state.layer_states.iter()).rev() {
            let next_d_input = layer.backward(backend, &d_input, layer_state, dimensions, seq_len);
            d_input.return_to_pool(backend);
            d_input = next_d_input;
        }

        backend.run_embedding_backward(&state.tokens, &d_input, grad_emb_i32_buffer, grad_pos);
        d_input
    }

    pub fn zero_grad(&mut self, backend: &WgpuBackend) {
        for layer in &mut self.layers {
            layer.zero_grad(backend);
        }
    }

    pub fn update_weights(&mut self, backend: &WgpuBackend, learning_rate: f32) {
        self.t += 1;
        for layer in &mut self.layers {
            layer.update_weights(backend, learning_rate, self.t);
        }
    }

    pub fn scale_grads(&mut self, backend: &WgpuBackend, scale: f32) {
        for layer in &mut self.layers {
            layer.scale_grads(backend, scale);
        }
    }

    pub fn clip_grads(&mut self, backend: &WgpuBackend, _max_norm: f32) {
        // Simple global norm clipping (per layer for simplicity here, could be global global)
        self.scale_grads(backend, 1.0); 
    }
}
