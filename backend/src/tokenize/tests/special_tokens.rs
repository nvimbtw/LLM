use tokenize::pretokenize::{lookup_special, pre_tokenize_str, pre_tokenize_text};

#[test]
fn test_special_tokens_mapped_to_pua() {
    let text = "<|user|>hello<|/user|>";
    let words = pre_tokenize_str(text);
    let combined: String = words.join("");
    assert!(
        combined.contains('\u{E004}'),
        "Expected <|user|> to become U+E004, got: {:?}",
        combined
    );
    assert!(
        combined.contains('\u{E005}'),
        "Expected <|/user|> to become U+E005, got: {:?}",
        combined
    );
}

#[test]
fn test_lookup_special_roundtrip() {
    assert_eq!(lookup_special('\u{E000}'), Some("<|endoftext|>"));
    assert_eq!(lookup_special('\u{E001}'), Some("<|par|>"));
    assert_eq!(lookup_special('\u{E004}'), Some("<|user|>"));
    assert_eq!(lookup_special('\u{E005}'), Some("<|/user|>"));
    assert_eq!(lookup_special('\u{E006}'), Some("<|assistant|>"));
    assert_eq!(lookup_special('a'), None);
}

#[test]
fn test_pua_survives_pre_tokenize() {
    let text = "<|code|>fn main() {}\n<|/code|>";
    let tokens = pre_tokenize_text(text);
    let has_code = tokens.iter().any(|word| word.contains(&'\u{E002}'));
    let has_end_code = tokens.iter().any(|word| word.contains(&'\u{E003}'));
    assert!(has_code, "Pre-tokenized words should contain U+E002");
    assert!(has_end_code, "Pre-tokenized words should contain U+E003");
}

#[test]
fn test_lookup_returns_none_for_normal_chars() {
    for c in ['a', 'Z', '0', '\n', ' ', '.', ','] {
        assert!(lookup_special(c).is_none(), "Normal char {:?} should not be special", c);
    }
}
