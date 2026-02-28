use crate::io::*;
use crate::tokenizer::decoder::decode;
use crate::tokenizer::encoder::encode;
use crate::train::transformer::{new_table, Transformer};
use crate::train::backend::{WgpuBackend, GpuTensor};
use rayon::prelude::*;
use pollster::block_on;
use wgpu::util::DeviceExt;
use std::sync::Arc;

pub fn vec_mat_mul(v: &Vec<f32>, m: &Vec<Vec<f32>>) -> Vec<f32> {
    let cols = m[0].len();
    let inner = v.len();
    let output: Vec<f32> = (0..cols).into_par_iter().map(|j| {
        let mut sum = 0.0;
        for k in 0..inner {
            sum += v[k] * m[k][j];
        }
        sum
    }).collect();
    output
}

pub fn softmax_1d(v: &mut Vec<f32>) {
    let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = v.iter().map(|x| (x - max).exp()).sum();
    for x in v.iter_mut() {
        *x = ((*x - max).exp()) / sum;
    }
}

pub fn generate_text_with_loaded_model(
    backend: &WgpuBackend,
    prompt: &str,
    max_length: usize,
    context_window: usize,
    _dimensions: usize,
) -> String {
    let (transformer_instance, embedding_table, positional_table, dim) =
        crate::train::transformer::init_transformer(backend);
    generate_text_with_model(
        backend,
        transformer_instance,
        embedding_table,
        positional_table,
        dim,
        prompt,
        max_length,
        context_window,
    )
}

use rand::distr::{weighted::WeightedIndex, Distribution};
use rand::rng;

pub fn sample_token(probs: &[f32], temperature: f32) -> u32 {
    let mut adjusted_probs = probs.to_vec();
    if temperature != 1.0 {
        for p in adjusted_probs.iter_mut() {
            *p = p.powf(1.0 / temperature);
        }
        let sum: f32 = adjusted_probs.iter().sum();
        for p in adjusted_probs.iter_mut() {
            *p /= sum;
        }
    }
    let dist = WeightedIndex::new(&adjusted_probs).expect("Failed to create weighted index");
    dist.sample(&mut rng()) as u32
}

pub fn generate_text_with_model(
    backend: &WgpuBackend,
    transformer: Transformer,
    embedding_table: GpuTensor,
    positional_table: GpuTensor,
    dimensions: usize,
    prompt: &str,
    max_length: usize,
    _context_window_request: usize,
) -> String {
    let vocab = read_vocab("data/pairs.bin").expect("Failed to read vocab");
    let w_lm_raw = load_matrix("data/w_lm.bin").unwrap_or_else(|_| new_table(dimensions, vocab.len() + 256));
    let w_lm = GpuTensor::from_cpu(backend, &w_lm_raw);

    let mut generated_tokens: Vec<u32> = encode(prompt.to_string(), vocab.clone());
    let context_window_limit = positional_table.shape.0;
    let temperature = 0.8;

    println!("Starting generation with prompt: \"{}\"", prompt);

    for _ in 0..max_length {
        let current_input_tokens = if generated_tokens.len() > context_window_limit {
            generated_tokens[generated_tokens.len() - context_window_limit..].to_vec()
        } else {
            generated_tokens.clone()
        };

        if current_input_tokens.is_empty() { break; }

        let tokens_gpu = Arc::new(backend.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&current_input_tokens),
            usage: wgpu::BufferUsages::STORAGE,
        }));

        let (transformer_output, _state) = transformer.forward(backend, tokens_gpu, &embedding_table, &positional_table, current_input_tokens.len(), dimensions);

        let output_cpu = block_on(transformer_output.to_cpu(backend));
        let last_token_output = output_cpu.last().unwrap();

        let w_lm_cpu = block_on(w_lm.to_cpu(backend));
        let mut logits = vec_mat_mul(last_token_output, &w_lm_cpu);

        softmax_1d(&mut logits);
        let next_token_id = sample_token(&logits, temperature);
        generated_tokens.push(next_token_id);
    }

    decode(&generated_tokens, &vocab)
}
