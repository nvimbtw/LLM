pub mod backend;
pub mod loss;
pub mod output;
pub mod transformer;

use crate::io::*;
use crate::train::loss::cross_entropy_loss;
use crate::train::transformer::*;
use crate::train::backend::{WgpuBackend, GpuTensor};
use pollster::block_on;
use wgpu::util::DeviceExt;
use std::sync::Arc;

pub fn train() {
    let backend = block_on(WgpuBackend::new()).expect("Failed to initialize GPU backend");
    
    let tokens = read_tokens("data/tokens.bin").expect("Failed to read tokens");
    let vocab = read_vocab("data/pairs.bin").expect("Failed to read pairs");

    let dimensions = 128;
    let context_window = 128;
    let learning_rate = 0.001;
    let epochs = 10;
    let batch_size = 1;

    let vocab_size = vocab.len() + 256;
    
    let (mut transformer, mut embedding_table, mut positional_table, _) = init_transformer(&backend);
    
    let w_lm_raw = load_matrix("data/w_lm.bin").unwrap_or_else(|_| new_table(dimensions, vocab_size));
    let mut w_lm = GpuTensor::from_cpu(&backend, &w_lm_raw);

    let mut grad_embedding = GpuTensor::from_cpu(&backend, &vec![vec![0.0f32; dimensions]; vocab_size]);
    let mut grad_positional = GpuTensor::from_cpu(&backend, &vec![vec![0.0f32; dimensions]; context_window]);
    let mut grad_w_lm = GpuTensor::from_cpu(&backend, &vec![vec![0.0f32; vocab_size]; dimensions]);

    println!(
        "Starting 100% GPU training on {} tokens, vocab size {}",
        tokens.len(),
        vocab_size
    );

    let stride = 32;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut count = 0;

        for i in (0..(tokens.len() - context_window - 1)).step_by(stride) {
            let input_tokens = &tokens[i..i + context_window];
            let target_tokens = &tokens[i + 1..i + context_window + 1];

            // 1. Send tokens to GPU
            let tokens_gpu = Arc::new(backend.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: None,
                contents: bytemuck::cast_slice(input_tokens),
                usage: wgpu::BufferUsages::STORAGE,
            }));

            // 2. Forward Pass (100% GPU)
            let (transformer_output, state) = transformer.forward(&backend, tokens_gpu, &embedding_table, &positional_table, context_window, dimensions);

            let logits_batch_gpu = backend.run_matmul(&transformer_output, &w_lm);
            let logits_batch = block_on(logits_batch_gpu.to_cpu(&backend));

            // 3. Compute Loss (CPU)
            let mut step_loss = 0.0;
            let mut grad_logits_batch_raw = vec![vec![0.0f32; vocab_size]; context_window];

            for t in 0..context_window {
                let target_token = target_tokens[t] as usize;
                let (loss, grad_logits) = cross_entropy_loss(&logits_batch[t], target_token);
                step_loss += loss / context_window as f32;
                for j in 0..vocab_size {
                    grad_logits_batch_raw[t][j] = grad_logits[j] / context_window as f32;
                }
            }
            let grad_logits_batch = GpuTensor::from_cpu(&backend, &grad_logits_batch_raw);

            total_loss += step_loss;
            count += 1;

            // 4. Backward Pass (100% GPU)
            let transformer_output_t = backend.run_transpose(&transformer_output);
            let d_w_lm = backend.run_matmul(&transformer_output_t, &grad_logits_batch);
            backend.run_add_to_grad(&mut grad_w_lm, &d_w_lm);

            let w_lm_t = backend.run_transpose(&w_lm);
            let d_transformer_output = backend.run_matmul(&grad_logits_batch, &w_lm_t);

            transformer.backward(&backend, &d_transformer_output, &state, &mut grad_embedding, &mut grad_positional, dimensions);

            // 5. Update weights (SGD on GPU)
            if count % batch_size == 0 {
                transformer.update_weights(&backend, learning_rate);
                backend.run_update(&mut embedding_table, &grad_embedding, learning_rate);
                backend.run_update(&mut positional_table, &grad_positional, learning_rate);
                backend.run_update(&mut w_lm, &grad_w_lm, learning_rate);

                transformer.zero_grad(&backend);
                grad_embedding.zero(&backend);
                grad_positional.zero(&backend);
                grad_w_lm.zero(&backend);
            }

            if count % 100 == 0 {
                println!("Epoch {} | Step {} | Loss: {:.4}", epoch, count, step_loss);
            }
        }

        println!("Epoch {} completed. Average Loss: {:.4}", epoch, total_loss / count as f32);
    }

    // Save model (Download from GPU first)
    println!("Saving model...");
    save_matrix(&block_on(embedding_table.to_cpu(&backend)), "data/embedding_table.bin").unwrap();
    save_matrix(&block_on(positional_table.to_cpu(&backend)), "data/positional_table.bin").unwrap();
    save_matrix(&block_on(w_lm.to_cpu(&backend)), "data/w_lm.bin").unwrap();
    save_matrix(&block_on(transformer.w_q.to_cpu(&backend)), "data/w_q.bin").unwrap();
    save_matrix(&block_on(transformer.w_k.to_cpu(&backend)), "data/w_k.bin").unwrap();
    save_matrix(&block_on(transformer.w_v.to_cpu(&backend)), "data/w_v.bin").unwrap();
    save_matrix(&block_on(transformer.w_o.to_cpu(&backend)), "data/w_o.bin").unwrap();
    save_matrix(&block_on(transformer.w_ff1.to_cpu(&backend)), "data/w_ff1.bin").unwrap();
    save_matrix(&block_on(transformer.w_ff2.to_cpu(&backend)), "data/w_ff2.bin").unwrap();

    println!("Training finished.");
}
