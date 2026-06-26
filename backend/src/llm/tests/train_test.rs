use llm::{infer::generate_response, train::train};

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
