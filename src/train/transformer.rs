use crate::io::*;
use crate::io::io::load_config;
use crate::train::backend::{GpuTensor, WgpuBackend, GpuCommandSession};
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

    pub fn update(&mut self, backend: &WgpuBackend, lr: f32, t: u32, session: &mut Option<GpuCommandSession>) {
        backend.run_adam_update(&mut self.data, &self.grad, &mut self.m, &mut self.v, lr, 0.9, 0.999, 1e-8, t, session);
    }

    pub fn zero_grad(&mut self, backend: &WgpuBackend, session: &mut Option<GpuCommandSession>) {
        self.grad.zero_with_session(backend, session);
    }

    pub fn scale_grad(&mut self, backend: &WgpuBackend, scale: f32, session: &mut Option<GpuCommandSession>) {
        backend.run_scale(&mut self.grad, scale, session);
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
        session: &mut Option<GpuCommandSession>,
    ) -> (GpuTensor, LayerForwardState) {
        let ln1_output = backend.run_layer_norm(input, session);

        let q_total = backend.run_matmul_with_session(&ln1_output, &self.w_q.data, session);
        let k_total = backend.run_matmul_with_session(&ln1_output, &self.w_k.data, session);
        let v_total = backend.run_matmul_with_session(&ln1_output, &self.w_v.data, session);

        let head_dim = dimensions / self.n_heads;
        
        let q_perm = backend.run_permute_021(&q_total, seq_len, self.n_heads, head_dim, session);
        let k_perm = backend.run_permute_021(&k_total, seq_len, self.n_heads, head_dim, session);
        let v_perm = backend.run_permute_021(&v_total, seq_len, self.n_heads, head_dim, session);

        let k_perm_t = backend.run_batched_transpose(&k_perm, self.n_heads, seq_len, head_dim, session);
        let mut scores = backend.run_batched_matmul(&q_perm, &k_perm_t, self.n_heads, seq_len, head_dim, seq_len, session);
        
        backend.run_scale_mask(&mut scores, 1.0 / (head_dim as f32).sqrt(), session);

        let mut probs = scores;
        backend.run_softmax(&mut probs, session);

        let attn_perm = backend.run_batched_matmul(&probs, &v_perm, self.n_heads, seq_len, seq_len, head_dim, session);
        let attn = backend.run_permute_102(&attn_perm, self.n_heads, seq_len, head_dim, session);
        
        let attn_output_linear = backend.run_matmul_with_session(&attn, &self.w_o.data, session);
        let attn_output = backend.run_add(input, &attn_output_linear, session);

        let ln2_output = backend.run_layer_norm(&attn_output, session);

        let ff1_output = backend.run_matmul_with_session(&ln2_output, &self.w_ff1.data, session);
        
        let mut relu_output = backend.run_copy_with_session(&ff1_output, session);
        backend.run_relu(&mut relu_output, session);

        let ffn_output_linear = backend.run_matmul_with_session(&relu_output, &self.w_ff2.data, session);
        let output = backend.run_add(&attn_output, &ffn_output_linear, session);

        // Cleanup session-local temporary tensors
        k_perm_t.return_to_session(session);
        attn_perm.return_to_session(session);
        attn_output_linear.return_to_session(session);
        ffn_output_linear.return_to_session(session);
        q_total.return_to_session(session);
        k_total.return_to_session(session);
        v_total.return_to_session(session);

        (
            output,
            LayerForwardState {
                ln1_output,
                q_perm,
                k_perm,
                v_perm,
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
        session: &mut Option<GpuCommandSession>,
    ) -> GpuTensor {
        let head_dim = dimensions / self.n_heads;

        let relu_output_t = backend.run_transpose(&state.relu_output, session);
        let d_w_ff2 = backend.run_matmul_with_session(&relu_output_t, d_output, session);
        backend.run_add_to_grad(&mut self.w_ff2.grad, &d_w_ff2, session);

        let w_ff2_t = backend.run_transpose(&self.w_ff2.data, session);
        let d_relu_output = backend.run_matmul_with_session(d_output, &w_ff2_t, session);

        let mut d_ff1_output = backend.run_copy_with_session(&d_relu_output, session);
        backend.run_relu_backward(&state.ff1_output, &mut d_ff1_output, session);

        let ln2_output_t = backend.run_transpose(&state.ln2_output, session);
        let d_w_ff1 = backend.run_matmul_with_session(&ln2_output_t, &d_ff1_output, session);
        backend.run_add_to_grad(&mut self.w_ff1.grad, &d_w_ff1, session);

        let w_ff1_t = backend.run_transpose(&self.w_ff1.data, session);
        let d_ln2_output = backend.run_matmul_with_session(&d_ff1_output, &w_ff1_t, session);
        
        let d_attn_output_from_ffn = backend.run_layer_norm_backward(&state.ln2_output, &d_ln2_output, session);
        let d_attn_output = backend.run_add(d_output, &d_attn_output_from_ffn, session);

        let w_o_t = backend.run_transpose(&self.w_o.data, session);
        let d_attn = backend.run_matmul_with_session(&d_attn_output, &w_o_t, session);

        let d_attn_perm = backend.run_permute_021(&d_attn, seq_len, self.n_heads, head_dim, session);

        let attn_t = backend.run_transpose(&state.attn, session);
        let d_w_o = backend.run_matmul_with_session(&attn_t, &d_attn_output, session);
        backend.run_add_to_grad(&mut self.w_o.grad, &d_w_o, session);

        // Attention backward
        let v_perm_t = backend.run_batched_transpose(&state.v_perm, self.n_heads, seq_len, head_dim, session);
        let d_probs = backend.run_batched_matmul(&d_attn_perm, &v_perm_t, self.n_heads, seq_len, head_dim, seq_len, session);

        let probs_t = backend.run_batched_transpose(&state.probs, self.n_heads, seq_len, seq_len, session);
        let d_v_perm = backend.run_batched_matmul(&probs_t, &d_attn_perm, self.n_heads, seq_len, seq_len, head_dim, session);

        let d_scores = backend.run_softmax_backward(&state.probs, &d_probs, 1.0 / (head_dim as f32).sqrt(), session);

        let k_perm_t = backend.run_batched_transpose(&state.k_perm, self.n_heads, seq_len, head_dim, session);
        let d_q_perm = backend.run_batched_matmul(&d_scores, &k_perm_t, self.n_heads, seq_len, seq_len, head_dim, session);
        
        let d_scores_t = backend.run_batched_transpose(&d_scores, self.n_heads, seq_len, seq_len, session);
        let d_k_perm = backend.run_batched_matmul(&d_scores_t, &state.q_perm, self.n_heads, seq_len, seq_len, head_dim, session);

        let d_q_total = backend.run_permute_102(&d_q_perm, self.n_heads, seq_len, head_dim, session);
        let d_k_total = backend.run_permute_102(&d_k_perm, self.n_heads, seq_len, head_dim, session);
        let d_v_total = backend.run_permute_102(&d_v_perm, self.n_heads, seq_len, head_dim, session);

        let ln1_output_t = backend.run_transpose(&state.ln1_output, session);
        let d_w_q = backend.run_matmul_with_session(&ln1_output_t, &d_q_total, session);
        backend.run_add_to_grad(&mut self.w_q.grad, &d_w_q, session);

        let d_w_k = backend.run_matmul_with_session(&ln1_output_t, &d_k_total, session);
        backend.run_add_to_grad(&mut self.w_k.grad, &d_w_k, session);

        let d_w_v = backend.run_matmul_with_session(&ln1_output_t, &d_v_total, session);
        backend.run_add_to_grad(&mut self.w_v.grad, &d_w_v, session);

        let w_q_t = backend.run_transpose(&self.w_q.data, session);
        let w_k_t = backend.run_transpose(&self.w_k.data, session);
        let w_v_t = backend.run_transpose(&self.w_v.data, session);

        let d_ln1_output_q = backend.run_matmul_with_session(&d_q_total, &w_q_t, session);
        let d_ln1_output_k = backend.run_matmul_with_session(&d_k_total, &w_k_t, session);
        let d_ln1_output_v = backend.run_matmul_with_session(&d_v_total, &w_v_t, session);

        let d_ln1_output_tmp = backend.run_add(&d_ln1_output_q, &d_ln1_output_k, session);
        let d_ln1_output = backend.run_add(&d_ln1_output_tmp, &d_ln1_output_v, session);

        let d_input_from_attn = backend.run_layer_norm_backward(&state.ln1_output, &d_ln1_output, session);
        let d_input_total = backend.run_add(&d_attn_output, &d_input_from_attn, session);

        // Cleanup
        relu_output_t.return_to_session(session);
        d_w_ff2.return_to_session(session);
        w_ff2_t.return_to_session(session);
        d_relu_output.return_to_session(session);
        d_ff1_output.return_to_session(session);
        ln2_output_t.return_to_session(session);
        d_w_ff1.return_to_session(session);
        w_ff1_t.return_to_session(session);
        d_ln2_output.return_to_session(session);
        d_attn_output_from_ffn.return_to_session(session);
        d_attn_output.return_to_session(session);
        w_o_t.return_to_session(session);
        d_attn.return_to_session(session);
        d_attn_perm.return_to_session(session);
        attn_t.return_to_session(session);
        d_w_o.return_to_session(session);
        v_perm_t.return_to_session(session);
        d_probs.return_to_session(session);
        probs_t.return_to_session(session);
        d_v_perm.return_to_session(session);
        d_scores.return_to_session(session);
        k_perm_t.return_to_session(session);
        d_q_perm.return_to_session(session);
        d_scores_t.return_to_session(session);
        d_k_perm.return_to_session(session);
        d_q_total.return_to_session(session);
        d_k_total.return_to_session(session);
        d_v_total.return_to_session(session);
        ln1_output_t.return_to_session(session);
        d_w_q.return_to_session(session);
        d_w_k.return_to_session(session);
        d_w_v.return_to_session(session);
        w_q_t.return_to_session(session);
        w_k_t.return_to_session(session);
        w_v_t.return_to_session(session);
        d_ln1_output_q.return_to_session(session);
        d_ln1_output_k.return_to_session(session);
        d_ln1_output_v.return_to_session(session);
        d_ln1_output_tmp.return_to_session(session);
        d_ln1_output.return_to_session(session);
        d_input_from_attn.return_to_session(session);

        d_input_total
    }

    pub fn zero_grad(&mut self, backend: &WgpuBackend, session: &mut Option<GpuCommandSession>) {
        self.w_q.zero_grad(backend, session);
        self.w_k.zero_grad(backend, session);
        self.w_v.zero_grad(backend, session);
        self.w_o.zero_grad(backend, session);
        self.w_ff1.zero_grad(backend, session);
        self.w_ff2.zero_grad(backend, session);
    }

    pub fn update_weights(&mut self, backend: &WgpuBackend, learning_rate: f32, t: u32, session: &mut Option<GpuCommandSession>) {
        self.w_q.update(backend, learning_rate, t, session);
        self.w_k.update(backend, learning_rate, t, session);
        self.w_v.update(backend, learning_rate, t, session);
        self.w_o.update(backend, learning_rate, t, session);
        self.w_ff1.update(backend, learning_rate, t, session);
        self.w_ff2.update(backend, learning_rate, t, session);
    }

    pub fn scale_grads(&mut self, backend: &WgpuBackend, scale: f32, session: &mut Option<GpuCommandSession>) {
        self.w_q.scale_grad(backend, scale, session);
        self.w_k.scale_grad(backend, scale, session);
        self.w_v.scale_grad(backend, scale, session);
        self.w_o.scale_grad(backend, scale, session);
        self.w_ff1.scale_grad(backend, scale, session);
        self.w_ff2.scale_grad(backend, scale, session);
    }
}

pub struct LayerForwardState {
    pub ln1_output: GpuTensor,
    pub q_perm: GpuTensor,
    pub k_perm: GpuTensor,
    pub v_perm: GpuTensor,
    pub probs: GpuTensor,
    pub attn: GpuTensor,
    pub attn_output: GpuTensor,
    pub ln2_output: GpuTensor,
    pub ff1_output: GpuTensor,
    pub relu_output: GpuTensor,
}

impl LayerForwardState {
    pub fn return_to_session(self, session: &mut Option<GpuCommandSession>) {
        self.ln1_output.return_to_session(session);
        self.q_perm.return_to_session(session);
        self.k_perm.return_to_session(session);
        self.v_perm.return_to_session(session);
        self.probs.return_to_session(session);
        self.attn.return_to_session(session);
        self.attn_output.return_to_session(session);
        self.ln2_output.return_to_session(session);
        self.ff1_output.return_to_session(session);
        self.relu_output.return_to_session(session);
    }

    pub fn return_to_pool(self, backend: &WgpuBackend) {
        self.ln1_output.return_to_pool(backend);
        self.q_perm.return_to_pool(backend);
        self.k_perm.return_to_pool(backend);
        self.v_perm.return_to_pool(backend);
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
    pub fn return_to_session(self, session: &mut Option<GpuCommandSession>) {
        self.input.return_to_session(session);
        for state in self.layer_states {
            state.return_to_session(session);
        }
    }

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
        session: &mut Option<GpuCommandSession>,
    ) -> (GpuTensor, TransformerForwardState) {
        let input = backend.run_embedding_forward(&tokens, embedding, positional, seq_len, session);
        let mut current_input = backend.run_copy_with_session(&input, session);
        let mut layer_states = Vec::new();

        for layer in &self.layers {
            let (output, state) = layer.forward(backend, &current_input, seq_len, dimensions, session);
            current_input.return_to_session(session);
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
        session: &mut Option<GpuCommandSession>,
    ) -> GpuTensor {
        let mut d_input = backend.run_copy_with_session(d_output, session);
        let seq_len = state.input.shape.0;

        for (layer, layer_state) in self.layers.iter_mut().zip(state.layer_states.iter()).rev() {
            let next_d_input = layer.backward(backend, &d_input, layer_state, dimensions, seq_len, session);
            d_input.return_to_session(session);
            d_input = next_d_input;
        }

        backend.run_embedding_backward(&state.tokens, &d_input, grad_emb_i32_buffer, grad_pos, session);
        d_input
    }

    pub fn zero_grad(&mut self, backend: &WgpuBackend, session: &mut Option<GpuCommandSession>) {
        for layer in &mut self.layers {
            layer.zero_grad(backend, session);
        }
    }

    pub fn update_weights(&mut self, backend: &WgpuBackend, learning_rate: f32, session: &mut Option<GpuCommandSession>) {
        self.t += 1;
        for layer in &mut self.layers {
            layer.update_weights(backend, learning_rate, self.t, session);
        }
    }

    pub fn scale_grads(&mut self, backend: &WgpuBackend, scale: f32, session: &mut Option<GpuCommandSession>) {
        for layer in &mut self.layers {
            layer.scale_grads(backend, scale, session);
        }
    }
}
