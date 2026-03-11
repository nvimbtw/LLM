pub mod backend;
pub mod output;
pub mod transformer;

use crate::io::*;
use crate::train::backend::{GpuTensor, WgpuBackend};
use crate::train::transformer::*;
use pollster::block_on;
use std::sync::Arc;
use wgpu::util::DeviceExt;

pub fn train() {
    let backend = block_on(WgpuBackend::new()).expect("Failed to initialize GPU backend");

    let tokens = read_tokens("data/tokens.bin").expect("Failed to read tokens");
    let vocab = read_vocab("data/pairs.bin").expect("Failed to read pairs");

    let dimensions = 1024;
    let context_window = 128;
    let learning_rate = 0.001;
    let epochs = 10;
    let batch_size = 512;

    let vocab_size = vocab.len() + 256;

    let (mut transformer, mut embedding_table, mut positional_table, _) =
        init_transformer(&backend, dimensions, context_window);

    let w_lm_raw =
        load_matrix("data/w_lm.bin").unwrap_or_else(|_| new_table(dimensions, vocab_size));
    let mut w_lm = GpuTensor::from_cpu(&backend, &w_lm_raw);

    // grad_embedding now refers to the i32 buffer for fixed-point gradients
    let grad_embedding_i32_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grad_embedding_i32_buffer"),
        size: (vocab_size * dimensions * 4) as u64, // 4 bytes per i32
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut grad_positional =
        GpuTensor::from_cpu(&backend, &vec![vec![0.0f32; dimensions]; context_window]);
    let mut grad_w_lm = GpuTensor::from_cpu(&backend, &vec![vec![0.0f32; vocab_size]; dimensions]);

    // Initialize i32 grad_embedding to zeros
    GpuTensor::zero_i32(&grad_embedding_i32_buffer, &backend);

    println!(
        "Starting 100% GPU training on {} tokens, vocab size {}",
        tokens.len(),
        vocab_size
    );

    let stride = 16;

    for epoch in 0..epochs {
        let mut total_loss = 0.0;
        let mut count = 0;

        for i in (0..(tokens.len() - context_window - 1)).step_by(stride) {
            let input_tokens = &tokens[i..i + context_window];
            let target_tokens = &tokens[i + 1..i + context_window + 1];

            // 1. Send tokens to GPU
            let tokens_gpu = Arc::new(backend.device.create_buffer_init(
                &wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(input_tokens),
                    usage: wgpu::BufferUsages::STORAGE,
                },
            ));

            // 2. Forward Pass (100% GPU)
            let (transformer_output, state) = transformer.forward(
                &backend,
                tokens_gpu,
                &embedding_table,
                &positional_table,
                context_window,
                dimensions,
            );

            let logits_batch_gpu = backend.run_matmul(&transformer_output, &w_lm);

            // 3. Compute Loss and Gradients (GPU)
            let target_tokens_u32: Vec<u32> = target_tokens.iter().map(|&x| x as u32).collect();
            let (loss_tensor, grad_logits_batch) =
                backend.run_cross_entropy(&logits_batch_gpu, &target_tokens_u32);
            let loss_cpu = block_on(loss_tensor.to_cpu(&backend));
            let step_loss: f32 =
                loss_cpu.iter().map(|row| row[0]).sum::<f32>() / context_window as f32;

            total_loss += step_loss;
            count += 1;

            // 4. Backward Pass (100% GPU)
            let transformer_output_t = backend.run_transpose(&transformer_output);
            let d_w_lm = backend.run_matmul(&transformer_output_t, &grad_logits_batch);
            backend.run_add_to_grad(&mut grad_w_lm, &d_w_lm);

            let w_lm_t = backend.run_transpose(&w_lm);
            let d_transformer_output = backend.run_matmul(&grad_logits_batch, &w_lm_t);

            transformer.backward(
                &backend,
                &d_transformer_output,
                &state,
                &grad_embedding_i32_buffer, // Pass the i32 buffer
                &mut grad_positional,
                dimensions,
            );

            // Return intermediate tensors to pool
            backend.pool.return_buffer(transformer_output.buffer);
            backend.pool.return_buffer(logits_batch_gpu.buffer);
            backend.pool.return_buffer(loss_tensor.buffer);
            backend.pool.return_buffer(grad_logits_batch.buffer);
            backend.pool.return_buffer(transformer_output_t.buffer);
            backend.pool.return_buffer(w_lm_t.buffer);
            backend.pool.return_buffer(d_transformer_output.buffer);
            backend.pool.return_buffer(d_w_lm.buffer);

            // 5. Update weights (SGD on GPU)
            if count % batch_size == 0 {
                let scale = 1.0 / (batch_size * context_window) as f32;

                transformer.scale_grads(&backend, scale);
                backend.run_scale(&mut grad_positional, scale);
                backend.run_scale(&mut grad_w_lm, scale);

                transformer.update_weights(&backend, learning_rate);
                // grad_embedding uses fixed-point i32, so call run_update_i32
                // We divide the LR by the same factor since we didn't scale the i32 gradients
                backend.run_update_i32(
                    &mut embedding_table,
                    &grad_embedding_i32_buffer,
                    learning_rate * scale,
                );

                // positional_table and w_lm use f32, so call run_update_f32
                backend.run_update_f32(&mut positional_table, &grad_positional, learning_rate);
                backend.run_update_f32(&mut w_lm, &grad_w_lm, learning_rate);

                transformer.zero_grad(&backend);
                GpuTensor::zero_i32(&grad_embedding_i32_buffer, &backend); // Zero i32 buffer
                grad_positional.zero(&backend);
                grad_w_lm.zero(&backend);
            }

            if count % 100 == 0 {
                println!("Epoch {} | Step {} | Loss: {:.4}", epoch, count, step_loss);
            }
        }

        println!(
            "Epoch {} completed. Average Loss: {:.4}",
            epoch,
            total_loss / count as f32
        );
    }

    // Save model (Download from GPU first)
    println!("Saving model...");
    save_matrix(
        &block_on(embedding_table.to_cpu(&backend)),
        "data/embedding_table.bin",
    )
    .unwrap();
    save_matrix(
        &block_on(positional_table.to_cpu(&backend)),
        "data/positional_table.bin",
    )
    .unwrap();
    save_matrix(&block_on(w_lm.to_cpu(&backend)), "data/w_lm.bin").unwrap();
    save_matrix(&block_on(transformer.w_q.to_cpu(&backend)), "data/w_q.bin").unwrap();
    save_matrix(&block_on(transformer.w_k.to_cpu(&backend)), "data/w_k.bin").unwrap();
    save_matrix(&block_on(transformer.w_v.to_cpu(&backend)), "data/w_v.bin").unwrap();
    save_matrix(&block_on(transformer.w_o.to_cpu(&backend)), "data/w_o.bin").unwrap();
    save_matrix(
        &block_on(transformer.w_ff1.to_cpu(&backend)),
        "data/w_ff1.bin",
    )
    .unwrap();
    save_matrix(
        &block_on(transformer.w_ff2.to_cpu(&backend)),
        "data/w_ff2.bin",
    )
    .unwrap();

    println!("Training finished.");
}
