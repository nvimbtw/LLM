use llm::{infer::generate_response, train::train};
use llm::train::sft_train;
use std::sync::{Arc, Mutex};

#[test]
fn test_training_runs() {
    // The crate is layout-agnostic: the caller supplies every path. The backend
    // builds these via its storage module; here we point at backend/data/ by
    // hand (this file lives at backend/src/llm/tests/) and write the run into a
    // throwaway directory next to it.
    let model_dir = std::path::Path::new("target/test_model/test");
    let corpus_path = std::path::Path::new("../../data/datasets/corpus.bin");
    let merges_path = std::path::Path::new("../../data/artifacts/tokenizer/merges.bin");

    // Use small dimensions, 1 epoch, small context
    let result = train(
        32, 16, 4, 1, 1, 0.01, 2, model_dir, corpus_path, merges_path, None,
        Box::new(|_| {}), None,
    );
    assert!(result.is_ok(), "Training failed: {:?}", result.err());

    // Test inference with the trained model
    let response = generate_response(model_dir, "Hello", 10, 0.0, 1);
    assert!(response.is_ok(), "Inference failed: {:?}", response.err());
    let text = response.unwrap();
    eprintln!("Generated: {:?}", text);
    assert!(!text.is_empty(), "Generated text should not be empty");

    // Cleanup test artifacts
    let _ = std::fs::remove_dir_all(model_dir);
}

// ISSUE-0035: sft_train loads a pretrained base and fine-tunes it on a masked
// SFT dataset, driving the masked step. This is the epic's smoke-test seed: run
// a few SFT epochs over a tiny masked fixture and assert the (assistant-region)
// loss decreases — i.e. the base actually learns from the masked-in positions.
#[test]
fn test_sft_train_masked_loss_decreases() {
    let merges_path = std::path::Path::new("../../data/artifacts/tokenizer/merges.bin");
    let corpus_path = std::path::Path::new("../../data/datasets/corpus.bin");

    // 1. Produce a tiny pretrained base model (small dims, ctx 16, 1 epoch).
    let base_dir = std::path::Path::new("target/test_sft/base");
    let _ = std::fs::remove_dir_all(base_dir);
    let base = train(
        32, 16, 4, 1, 1, 0.01, 2, base_dir, corpus_path, merges_path, None,
        Box::new(|_| {}), None,
    );
    assert!(base.is_ok(), "base pretraining failed: {:?}", base.err());

    // 2. Hand-build a masked SFT fixture: one memorizable 16-token sequence.
    //    The mask ignores the first 4 positions (stand-in for system/user
    //    scaffold) and trains on the rest (stand-in for the assistant reply).
    let dataset_dir = std::path::Path::new("target/test_sft/dataset");
    std::fs::create_dir_all(dataset_dir).unwrap();
    let ids: Vec<u32> = (0..16).map(|i| 256 + (i % 4)).collect();
    let corpus_bytes: Vec<u8> = ids.iter().flat_map(|t| t.to_le_bytes()).collect();
    let mask: Vec<u8> = (0..16u8).map(|i| if i < 4 { 0 } else { 1 }).collect();
    let sft_corpus = dataset_dir.join("sft_corpus.bin");
    let sft_mask = dataset_dir.join("sft_mask.bin");
    std::fs::write(&sft_corpus, &corpus_bytes).unwrap();
    std::fs::write(&sft_mask, &mask).unwrap();

    // 3. Fine-tune, capturing the per-epoch average (masked) loss.
    let out_dir = std::path::Path::new("target/test_sft/out");
    let _ = std::fs::remove_dir_all(out_dir);
    let losses: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&losses);
    let result = sft_train(
        base_dir, &sft_corpus, &sft_mask, out_dir, 8, 0.01, None,
        Box::new(move |p| {
            if p.epoch_done && p.avg_loss.is_finite() {
                sink.lock().unwrap().push(p.avg_loss);
            }
        }),
        None,
    );
    assert!(result.is_ok(), "sft_train failed: {:?}", result.err());

    // The fine-tuned model must load standalone (merges copied next to weights).
    assert!(out_dir.join("model.bin").is_file(), "no fine-tuned model.bin");
    assert!(out_dir.join("merges.bin").is_file(), "base merges not copied");

    let losses = losses.lock().unwrap();
    assert!(losses.len() >= 2, "expected multiple epoch losses, got {:?}", *losses);
    let first = losses[0];
    let last = *losses.last().unwrap();
    assert!(
        last < first,
        "masked SFT loss did not decrease: first={} last={} ({:?})",
        first, last, *losses
    );

    // Cleanup.
    let _ = std::fs::remove_dir_all(std::path::Path::new("target/test_sft"));
}
