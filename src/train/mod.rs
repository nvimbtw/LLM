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
    let learning_rate = 0.0001; // Adam usually needs lower LR
    let epochs = 10;
    let batch_size = 512;

    let vocab_size = vocab.len() + 256;

    println!(
        "Starting 100% GPU training initialization on {} tokens, vocab size {}, dimensions {}",
        tokens.len(),
        vocab_size,
        dimensions
    );

    println!("Initializing transformer weights...");
    let (mut transformer, mut embedding_table, mut positional_table, actual_dimensions) =
        init_transformer(&backend, context_window);

    println!("Initializing language model head...");
    let w_lm_raw =
        load_matrix("data/w_lm.bin").unwrap_or_else(|_| new_table(actual_dimensions, vocab_size));
    let mut w_lm = GpuTensor::from_cpu(&backend, &w_lm_raw);

    println!("Creating gradient buffers...");
    // grad_embedding now refers to the i32 buffer for fixed-point gradients
    let grad_embedding_i32_buffer = backend.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("grad_embedding_i32_buffer"),
        size: (vocab_size * actual_dimensions * 4) as u64, // 4 bytes per i32
        usage: wgpu::BufferUsages::STORAGE
            | wgpu::BufferUsages::COPY_SRC
            | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut grad_positional =
        GpuTensor::from_cpu(&backend, &vec![vec![0.0f32; actual_dimensions]; context_window]);
    let mut grad_w_lm = GpuTensor::from_cpu(&backend, &vec![vec![0.0f32; vocab_size]; actual_dimensions]);

    // Initialize i32 grad_embedding to zeros
    GpuTensor::zero_i32(&grad_embedding_i32_buffer, &backend);

    println!("Initialization complete. Starting training loop...");

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
                actual_dimensions,
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

            let d_input = transformer.backward(
                &backend,
                &d_transformer_output,
                &state,
                &grad_embedding_i32_buffer, // Pass the i32 buffer
                &mut grad_positional,
                actual_dimensions,
            );

            // Return intermediate tensors to pool
            transformer_output.return_to_pool(&backend);
            logits_batch_gpu.return_to_pool(&backend);
            loss_tensor.return_to_pool(&backend);
            grad_logits_batch.return_to_pool(&backend);
            transformer_output_t.return_to_pool(&backend);
            w_lm_t.return_to_pool(&backend);
            d_transformer_output.return_to_pool(&backend);
            d_w_lm.return_to_pool(&backend);
            d_input.return_to_pool(&backend);
            state.return_to_pool(&backend);

            // 5. Update weights (Adam/SGD on GPU)
            if count % batch_size == 0 {
                let scale = 1.0 / (batch_size * context_window) as f32;

                transformer.scale_grads(&backend, scale);
                backend.run_scale(&mut grad_positional, scale);
                backend.run_scale(&mut grad_w_lm, scale);

                // Adam update
                transformer.update_weights(&backend, learning_rate);
                
                // embedding, positional, w_lm still use SGD for simplicity or should be Adam too?
                // The plan focused on Transformer layers for Adam.
                
                // grad_embedding uses fixed-point i32, so call run_update_i32
                backend.run_update_i32(
                    &mut embedding_table,
                    &grad_embedding_i32_buffer,
                    learning_rate * scale,
                );

                // positional_table and w_lm use f32
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
    
    // Save layers
    for (i, layer) in transformer.layers.iter().enumerate() {
        save_matrix(&block_on(layer.w_q.data.to_cpu(&backend)), &format!("data/layer_{}_w_q.bin", i)).unwrap();
        save_matrix(&block_on(layer.w_k.data.to_cpu(&backend)), &format!("data/layer_{}_w_k.bin", i)).unwrap();
        save_matrix(&block_on(layer.w_v.data.to_cpu(&backend)), &format!("data/layer_{}_w_v.bin", i)).unwrap();
        save_matrix(&block_on(layer.w_o.data.to_cpu(&backend)), &format!("data/layer_{}_w_o.bin", i)).unwrap();
        save_matrix(&block_on(layer.w_ff1.data.to_cpu(&backend)), &format!("data/layer_{}_w_ff1.bin", i)).unwrap();
        save_matrix(&block_on(layer.w_ff2.data.to_cpu(&backend)), &format!("data/layer_{}_w_ff2.bin", i)).unwrap();
    }

    println!("Training finished.");
}
