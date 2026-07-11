use llm::training::transformer::Transformer;
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

// Runs `sft_train` on the shared fixture with the given mask and returns the
// per-epoch average losses plus the saved model bytes. Helper for the epic smoke
// test below.
fn run_sft_fixture(
    base_dir: &std::path::Path,
    corpus_path: &std::path::Path,
    mask_path: &std::path::Path,
    out_dir: &std::path::Path,
    epochs: i64,
) -> (Vec<f32>, Vec<u8>) {
    let losses: Arc<Mutex<Vec<f32>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = Arc::clone(&losses);
    let r = sft_train(
        base_dir, corpus_path, mask_path, out_dir, epochs, 0.05, None,
        Box::new(move |p| {
            if p.epoch_done && p.avg_loss.is_finite() {
                sink.lock().unwrap().push(p.avg_loss);
            }
        }),
        None,
    );
    assert!(r.is_ok(), "sft_train failed: {:?}", r.err());
    let bytes = std::fs::read(out_dir.join("model.bin")).unwrap();
    let losses = Arc::try_unwrap(losses).unwrap().into_inner().unwrap();
    (losses, bytes)
}

// ISSUE-0030 (epic) definition-of-done smoke test: fine-tune a base model through
// `sft_train` and demonstrate, end-to-end, the two properties the whole epic
// exists to guarantee —
//   (a) masked-out (system/user/scaffold) tokens contribute NOTHING to the SFT
//       objective: an all-zero mask yields exactly 0.0 loss at every position, so
//       there is no learning signal outside the assistant span; and
//   (b) the assistant region actually LEARNS: with a real mask the loss drops, and
//       the resulting model differs from the all-zero-mask model.
// (The rigorous "zero gradient ⇒ GPU weights unchanged" is proven at the step
// level by `transformer::tests::test_train_step_masked_all_zero_trains_nothing`;
// here we assert the observable end-to-end behaviour. Byte-equality against the
// on-disk base is deliberately NOT asserted — training round-trips weights through
// f16 on the GPU, so even a zero-gradient run re-quantizes the f32 base.)
// Uses `Transformer::init` for the base (a random model is enough to exercise the
// load→masked-loop→save path) so the test is fast — no full-corpus pretrain.
#[test]
fn test_sft_epic_masking_isolates_gradient() {
    let root = std::path::Path::new("target/test_sft_epic");
    let _ = std::fs::remove_dir_all(root);
    let base_dir = root.join("base");
    std::fs::create_dir_all(&base_dir).unwrap();

    // Tiny random base model, saved to disk.
    let base = Transformer::init(32, 300, 16, 4, 2, 1);
    base.save(&base_dir.to_string_lossy()).unwrap();

    // A 16-token single-chunk fixture the tiny model can memorize.
    let ids: Vec<u32> = (0..16u32).map(|i| 32 + (i % 5)).collect();
    let corpus_path = root.join("sft_corpus.bin");
    std::fs::write(
        &corpus_path,
        ids.iter().flat_map(|t| t.to_le_bytes()).collect::<Vec<u8>>(),
    )
    .unwrap();

    // "real" mask: first 4 positions (scaffold) = 0, rest (assistant) = 1.
    let real_mask_path = root.join("real_mask.bin");
    std::fs::write(
        &real_mask_path,
        (0..16u8).map(|i| u8::from(i >= 4)).collect::<Vec<u8>>(),
    )
    .unwrap();
    // "zero" mask: everything masked out.
    let zero_mask_path = root.join("zero_mask.bin");
    std::fs::write(&zero_mask_path, vec![0u8; 16]).unwrap();

    // (a) All-masked-out fine-tune: every position is excluded, so every reported
    //     loss is exactly 0.0 — there is no signal to learn from.
    let (zero_losses, zero_bytes) =
        run_sft_fixture(&base_dir, &corpus_path, &zero_mask_path, &root.join("out_zero"), 4);
    assert!(!zero_losses.is_empty(), "zero-mask run reported no epochs");
    assert!(
        zero_losses.iter().all(|&l| l == 0.0),
        "all-masked-out SFT must report 0.0 loss everywhere, got {:?}",
        zero_losses
    );

    // (b) Real-mask fine-tune: the assistant region contributes loss, so it drops,
    //     and the fine-tuned model differs from the all-zero-mask model.
    let (real_losses, real_bytes) =
        run_sft_fixture(&base_dir, &corpus_path, &real_mask_path, &root.join("out_real"), 12);
    assert!(real_losses.len() >= 2, "expected multiple epoch losses: {:?}", real_losses);
    assert!(real_losses[0] > 0.0, "assistant region should have real loss");
    assert!(
        *real_losses.last().unwrap() < real_losses[0],
        "assistant-region loss did not drop: {:?}",
        real_losses
    );
    assert_ne!(
        zero_bytes, real_bytes,
        "real-mask SFT must move the model where the all-zero-mask run did not"
    );

    let _ = std::fs::remove_dir_all(root);
}
