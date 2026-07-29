use tokenize::bpe::{decode, encode, train};
use tokenize::pretokenize::pre_tokenize_text;

fn run_roundtrip(text: &str) -> String {
    let pre_tokenized = pre_tokenize_text(text);
    let (_vocab, _corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();
    let encoded = encode(text, &merges);
    decode(&encoded, &merges)
}

#[test]
fn test_basic_roundtrip() {
    let text = "hello world";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}

#[test]
fn test_special_tokens_roundtrip() {
    let text = "<|user|>hello<|/user|>";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}

#[test]
fn test_punctuation_roundtrip() {
    let text = "hello, world! how are you?";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}

#[test]
fn test_newlines_and_tabs_roundtrip() {
    let text = "line one\nline two\tindented";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}

#[test]
fn test_multiple_special_tokens() {
    let text = "<|system|>You are a helpful assistant.<|/system|><|user|>Hello!<|/user|><|assistant|>Hi there!<|/assistant|>";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}

#[test]
fn test_unicode_roundtrip() {
    let text = "café résumé 北京";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}

#[test]
fn test_repeated_patterns() {
    let text = "aaabbbaaabbb";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}

#[test]
fn test_empty_string() {
    let text = "";
    let decoded = run_roundtrip(text);
    assert_eq!(decoded, text);
}
