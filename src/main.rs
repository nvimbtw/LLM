use std::env;

mod generate;
mod io;
mod tokenizer;
mod train;

use crate::generate::*;
use crate::tokenizer::*;
use crate::train::backend::WgpuBackend;
use crate::train::transformer;
use pollster::block_on;

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    println!("Running with args: {:?}", args);

    if args.is_empty() {
        full_process();
    } else {
        match args[0].as_str() {
            "tokenise" => vocab::build_vocab(),
            "encode" => encoder::encoder(),
            "decode" => {
                println!("Reading vocab...");
                let vocab = io::read_vocab("data/pairs.bin").expect("Failed to read vocab");
                println!("Reading tokens...");
                let tokens = io::read_tokens("data/tokens.bin").expect("Failed to read tokens");
                println!(
                    "Loaded {} vocab pairs and {} tokens.",
                    vocab.len(),
                    tokens.len()
                );
                let limit = tokens.len().min(100);
                let text = decoder::decode(&tokens[..limit], &vocab);
                println!("Decoded text:\n{}", text);
            }
            "train" => {
                train::train();
                println!("Training completed.");
            }
            "generate" => {
                if args.len() < 2 {
                    println!("Usage: cargo run -- generate \"<prompt>\"");
                    return;
                }
                let prompt = args[1].clone();
                let backend = block_on(WgpuBackend::new()).expect("Failed to init GPU");

                let dimensions = 128;
                let context_window = 1024;
                let max_length = 256;

                let generated_text = generate_text_with_loaded_model(
                    &backend,
                    &prompt,
                    max_length,
                    context_window,
                    dimensions,
                );
                println!("Generated text:\n{}", generated_text);
            }
            _ => {
                println!("Unknown command: {}", args[0]);
            }
        }
    }
}

fn full_process() {
    vocab::build_vocab();
    encoder::encoder();
    let backend = block_on(WgpuBackend::new()).expect("Failed to init GPU");
    let dimensions = 256;
    let context_window = 1024;
    let (transformer_instance, embedding_table, positional_table, dimensions) =
        transformer::init_transformer(&backend, dimensions, context_window);
    let prompt = "The quick brown fox";
    let context_window = 1024;
    let max_length = 256;
    let generated_text = generate_text_with_model(
        &backend,
        transformer_instance,
        embedding_table,
        positional_table,
        dimensions,
        prompt,
        max_length,
        context_window,
    );
    println!("Generated text (full_process):\n{}", generated_text);
}
