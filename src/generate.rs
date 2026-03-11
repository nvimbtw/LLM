use crate::io::*;
use crate::tokenizer::decoder::decode;
use crate::tokenizer::encoder::encode;
use crate::train::backend::{GpuTensor, WgpuBackend};
use crate::train::transformer::{new_table, Transformer};
use pollster::block_on;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub fn generate_text_with_loaded_model(
    backend: &WgpuBackend,
    prompt: &str,
    max_length: usize,
    context_window: usize,
    _dimensions: usize,
) -> String {
    let (transformer_instance, embedding_table, positional_table, dim) =
        crate::train::transformer::init_transformer(backend, _dimensions, context_window);
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
    }
    
    let sum: f32 = adjusted_probs.iter().sum();
    if sum <= 0.0 || sum.is_nan() {
        // Fallback to argmax if something went wrong with probabilities
        return adjusted_probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
    }

    if temperature != 1.0 {
        for p in adjusted_probs.iter_mut() {
            *p /= sum;
        }
    }
    
    let dist = WeightedIndex::new(&adjusted_probs).unwrap_or_else(|_| {
        WeightedIndex::new(&vec![1.0; adjusted_probs.len()]).unwrap()
    });
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
    let vocab_size = vocab.len() + 256;
    let w_lm_raw =
        load_matrix("data/w_lm.bin").unwrap_or_else(|_| new_table(dimensions, vocab_size));
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

        if current_input_tokens.is_empty() {
            break;
        }

        let tokens_gpu = Arc::new(backend.device.create_buffer_init(
            &wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(&current_input_tokens),
                usage: wgpu::BufferUsages::STORAGE,
            },
        ));

        let (transformer_output, state) = transformer.forward(
            backend,
            tokens_gpu,
            &embedding_table,
            &positional_table,
            current_input_tokens.len(),
            dimensions,
        );

        // Perform final matmul and softmax on GPU
        let mut logits_tensor = backend.run_matmul(&transformer_output, &w_lm);
        backend.run_softmax(&mut logits_tensor);

        // Download only the last row of logits for sampling
        let logits = block_on(logits_tensor.last_row_to_cpu(backend));

        let next_token_id = sample_token(&logits, temperature);
        generated_tokens.push(next_token_id);

        // Cleanup: Return intermediate tensors to pool
        transformer_output.return_to_pool(backend);
        logits_tensor.return_to_pool(backend);
        state.return_to_pool(backend);
    }

    decode(&generated_tokens, &vocab)
}
