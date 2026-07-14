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

// ─── ISSUE-0057: mixed-precision (f32 master weights) lifts the LR cap ────────
// The epic (ISSUE-0052) added an f32 master copy of every trained weight so Adam
// accumulates the step in full precision and only casts a disposable f16 *view*
// for the matmuls. Before the fix the master lived in f16, so an update smaller
// than the f16 ULP near the weight rounded away entirely — updates vanished at
// small LR and lurched in coarse quanta (diverging) at large LR. These two tests
// prove the fix end-to-end.

use llm::training::gpu_workspace::GpuWorkspace;
use llm::training::transformer::{gpu_adam_update, sync_weights_from_gpu};

// Deterministic, unit-level proof that a sub-f16-ULP update now ACCUMULATES in
// the f32 master instead of rounding away. `ln_gamma` initializes to exactly 1.0,
// where the f16 gap (ULP) is 2^-10 ≈ 9.77e-4. We drive a constant gradient into
// that weight's grad buffer at an LR whose per-step Adam move (≈ lr, because Adam
// normalizes a constant gradient to ~sign(g)) is 1e-4 — below one f16 ULP. Over
// N steps the master must move by ≈ N·lr; a pre-fix f16 master would not budge,
// since each 1e-4 step rounds back to 1.0.
#[test]
fn test_master_accumulates_sub_ulp_updates() {
    let (vocab, d, ctx, heads, ff_mult, layers) = (20, 16, 8, 2, 2, 2);
    let t = Transformer::init(d, vocab, ctx, heads, ff_mult, layers);
    let ws = GpuWorkspace::new(&t, ctx as usize);

    // All grad buffers are zeroed by GpuWorkspace::new. Set a constant, positive
    // gradient on layer-0 ln_gamma only, and never zero it, so every Adam step
    // sees the same gradient.
    let grad = vec![1.0f32; d as usize];
    ws.g_dln_gamma[0].upload(&grad);

    let lr = 1e-4f32; // per-step move ≈ 1e-4, below the f16 ULP (~9.77e-4) at w=1.0
    let n_steps = 50i32;
    for step in 1..=n_steps {
        gpu_adam_update(&ws, lr, step, 1, layers as usize);
        // deliberately NOT zeroing accumulators — keep the gradient constant
    }

    let mut t_synced = Transformer {
        embedding_table: t.embedding_table.clone(),
        positional_table: t.positional_table.clone(),
        wq: t.wq.clone(), wk: t.wk.clone(), wv: t.wv.clone(),
        wo: t.wo.clone(), w1: t.w1.clone(), w2: t.w2.clone(),
        b1: t.b1.clone(), b2: t.b2.clone(),
        ln_gamma: t.ln_gamma.clone(), ln_beta: t.ln_beta.clone(),
        w_output: t.w_output.clone(),
        heads: t.heads, head_dim: t.head_dim, layers: t.layers,
    };
    sync_weights_from_gpu(&ws, &mut t_synced);

    // ln_gamma started at 1.0; with a positive gradient Adam drives it DOWN by
    // ≈ n_steps·lr = 5e-3 — several f16 ULPs, impossible if the master were f16.
    let moved = 1.0f32 - t_synced.ln_gamma[0][0];
    let ulp = 2f32.powi(-10); // f16 ULP near 1.0
    eprintln!(
        "  ln_gamma[0][0]: 1.0 -> {:.6}  (moved {:.6}, {:.1} ULPs, expected ≈ {:.6})",
        t_synced.ln_gamma[0][0], moved, moved / ulp, n_steps as f32 * lr
    );
    assert!(t_synced.ln_gamma[0][0].is_finite(), "weight went non-finite");
    assert!(
        moved > 2.0 * ulp,
        "sub-ULP updates did not accumulate: weight moved only {:.6} (<2 f16 ULPs) \
         — the f32 master is not the source of truth",
        moved
    );
    // The Adam-normalized move is ≈ lr/step, so total ≈ n_steps·lr. Allow a wide
    // band (bias-correction ramp + eps make early steps slightly smaller).
    let expected = n_steps as f32 * lr;
    assert!(
        moved > 0.5 * expected && moved < 1.5 * expected,
        "accumulated move {:.6} not within 2x of expected {:.6}",
        moved, expected
    );
}

// A previously-cap-bound LR now trains stably: overfit a single fixed sequence at
// the epic's target pretrain LR (1e-4) and assert the loss stays finite and
// trends down rather than blowing up. Mirrors `transformer::tests::
// test_overfits_single_sequence` but at the raised LR the epic exists to unlock.
//
// Observed ceiling (this toy: vocab 20, d 16, 2 layers, single fixed sequence,
// post-fix): 1e-4 trains cleanly (loss 3.22 -> 1.10, max never above first);
// 3e-4 is also stable and stronger (loss 3.04 -> 0.09). The loss never exceeds
// its starting value at either LR — no divergence. We assert on 1e-4 (the
// conservative pretrain default ISSUE-0058 bakes in); 3e-4 is comfortably in
// range on this toy. The old f16-master path could not sustain these because the
// per-step update rounded away / lurched in f16 quanta.
#[test]
fn test_raised_lr_trains_stably() {
    let (vocab, d, ctx, heads, ff_mult, layers) = (20, 16, 8, 2, 2, 2);
    let t = Transformer::init(d, vocab, ctx, heads, ff_mult, layers);
    let mut ws = GpuWorkspace::new(&t, ctx as usize);
    let chunk: Vec<u32> = vec![3, 7, 1, 14, 2, 9, 5, 11];

    let lr = 1e-4f32;
    let first = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
    ws.zero_accumulators();
    let mut last = first;
    let mut max_seen = first;
    for step in 1..=300 {
        last = pollster::block_on(t.train_step(&mut ws, &chunk, layers as usize));
        gpu_adam_update(&ws, lr, step, 1, layers as usize);
        ws.zero_accumulators();
        assert!(last.is_finite(), "loss went non-finite at step {} (lr={})", step, lr);
        if last > max_seen { max_seen = last; }
    }
    eprintln!(
        "  lr={:.0e}: first_loss={:.4} last_loss={:.4} max_loss={:.4}",
        lr, first, last, max_seen
    );
    assert!(last.is_finite(), "final loss non-finite at lr={}", lr);
    assert!(
        last < first,
        "loss did not decrease at lr={}: {:.4} -> {:.4} (max {:.4})",
        lr, first, last, max_seen
    );
}
