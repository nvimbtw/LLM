pub const SPECIAL_TOKENS: &[(&str, char)] = &[
    ("<|endoftext|>", '\u{E000}'),
    ("<|par|>", '\u{E001}'),
    ("<|code|>", '\u{E002}'),
    ("<|/code|>", '\u{E003}'),
    ("<|user|>", '\u{E004}'),
    ("<|/user|>", '\u{E005}'),
    ("<|assistant|>", '\u{E006}'),
    ("<|/assistant|>", '\u{E007}'),
    ("<|tool_call|>", '\u{E008}'),
    ("<|/tool_call|>", '\u{E009}'),
    ("<|tool_result|>", '\u{E00A}'),
    ("<|/tool_result|>", '\u{E00B}'),
    ("<|system|>", '\u{E00C}'),
    ("<|/system|>", '\u{E00D}'),
    ("<|search|>", '\u{E00E}'),
    ("<|bash|>", '\u{E00F}'),
    ("<|title|>", '\u{E010}'),
    ("<|/title|>", '\u{E011}'),
    ("<|heading|>", '\u{E012}'),
    ("<|/heading|>", '\u{E013}'),
    ("<|subheading|>", '\u{E014}'),
    ("<|/subheading|>", '\u{E015}'),
    ("<|codeblock|>", '\u{E016}'),
    ("<|/codeblock|>", '\u{E017}'),
    ("<|link|>", '\u{E018}'),
    ("<|/link|>", '\u{E019}'),
    ("<|url|>", '\u{E01A}'),
    ("<|list|>", '\u{E01B}'),
    ("<|/list|>", '\u{E01C}'),
    ("<|item|>", '\u{E01D}'),
    ("<|/item|>", '\u{E01E}'),
    ("<|quote|>", '\u{E01F}'),
    ("<|/quote|>", '\u{E020}'),
    ("<|math|>", '\u{E021}'),
    ("<|/math|>", '\u{E022}'),
    ("<|lang|>", '\u{E023}'),
    ("<|/lang|>", '\u{E024}'),
    ("<|/par|>", '\u{E025}'),
    ("<|divider|>", '\u{E026}'),
    ("<|italic|>", '\u{E027}'),
    ("<|/italic|>", '\u{E028}'),
    ("<|bold|>", '\u{E029}'),
    ("<|/bold|>", '\u{E02A}'),
    // Paired thinking block: `<|thinking|>...reasoning...<|/thinking|>`. Both
    // delimiters are protected; the reasoning between them is ordinary generated
    // text. The frontend hides the content and shows a themed status label
    // instead, so the model only has to learn the two clean markers.
    ("<|thinking|>", '\u{E02B}'),
    ("<|/thinking|>", '\u{E02C}'),
];

pub fn lookup_special(ch: char) -> Option<&'static str> {
    SPECIAL_TOKENS
        .iter()
        .find(|&&(_, c)| c == ch)
        .map(|&(s, _)| s)
}

fn replace_special_tokens(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let mut matched = false;
        for &(token_str, pua) in SPECIAL_TOKENS {
            if text[i..].starts_with(token_str) {
                result.push(pua);
                i += token_str.len();
                matched = true;
                break;
            }
        }
        if !matched {
            let ch = text[i..].chars().next().unwrap();
            result.push(ch);
            i += ch.len_utf8();
        }
    }
    result
}

pub fn pre_tokenize_text(text: &str) -> Vec<Vec<char>> {
    let text = replace_special_tokens(text);
    let text = replace_special_characters(text.to_string());
    let text = append_space(text);
    let words = split_words(text);
    let words = split_punctuation(words);
    split_to_chars(words)
}

pub fn pre_tokenize_str(text: &str) -> Vec<String> {
    let text = replace_special_tokens(text);
    let text = replace_special_characters(text.to_string());
    let text = append_space(text);
    let words = split_words(text);
    split_punctuation(words)
}

fn replace_special_characters(text: String) -> String {
    text.chars()
        .map(|c| match c {
            '\n' => 'Ċ',
            '\t' => 'ĉ',
            '\r' => ' ',
            _ => c,
        })
        .collect()
}

fn append_space(text: String) -> String {
    let mut output = String::new();

    for c in text.chars() {
        output.push(c);

        if c == ' ' {
            output.push('Ġ');
        }
    }

    output
}

fn split_words(text: String) -> Vec<String> {
    text.split(" ").map(|w| w.to_string()).collect()
}

fn split_punctuation(words: Vec<String>) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();

    for word in words {
        let mut current = String::new();

        for c in word.chars() {
            if matches!(c, ',' | '.' | '!' | '?' | '"' | ':' | ';' | 'Ċ' | 'ĉ') {
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
                result.push(c.to_string());
            } else if c == '\'' {
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
                current.push(c);
            } else {
                current.push(c);
            }
        }

        if !current.is_empty() {
            result.push(current);
        }
    }

    result
}

fn split_to_chars(words: Vec<String>) -> Vec<Vec<char>> {
    let mut words_in_chars: Vec<Vec<char>> = Vec::new();

    for word in words {
        let mut chars: Vec<char> = Vec::new();
        for c in word.chars() {
            chars.push(c);
        }
        words_in_chars.push(chars)
    }

    words_in_chars
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // replace_special_characters
    // -----------------------------------------------------------------------
    #[test]
    fn test_replace_special_characters_newline() {
        assert_eq!(replace_special_characters("\n".to_string()), "Ċ");
    }

    #[test]
    fn test_replace_special_characters_tab() {
        assert_eq!(replace_special_characters("\t".to_string()), "ĉ");
    }

    #[test]
    fn test_replace_special_characters_carriage_return() {
        assert_eq!(replace_special_characters("\r".to_string()), " ");
    }

    #[test]
    fn test_replace_special_characters_normal_text_unchanged() {
        let input = "hello world";
        assert_eq!(replace_special_characters(input.to_string()), input);
    }

    #[test]
    fn test_replace_special_characters_mixed() {
        assert_eq!(
            replace_special_characters("a\nb\tc\rd".to_string()),
            "aĊbĉc d"
        );
    }

    // -----------------------------------------------------------------------
    // append_space
    // -----------------------------------------------------------------------
    #[test]
    fn test_append_space_single_space() {
        // For each char: push char; if it's a space, push Ġ after it.
        // 'a' → push 'a', ' ' → push ' ', push 'Ġ', 'b' → push 'b'
        assert_eq!(append_space("a b".to_string()), "a Ġb");
    }

    #[test]
    fn test_append_space_no_spaces() {
        assert_eq!(append_space("abc".to_string()), "abc");
    }

    #[test]
    fn test_append_space_consecutive_spaces() {
        // Each space gets a Ġ appended right after it
        assert_eq!(append_space("a  b".to_string()), "a Ġ Ġb");
    }

    #[test]
    fn test_append_space_empty_string() {
        assert_eq!(append_space(String::new()), "");
    }

    #[test]
    fn test_append_space_leading_trailing_spaces() {
        assert_eq!(append_space(" a ".to_string()), " Ġa Ġ");
    }

    // -----------------------------------------------------------------------
    // split_words
    // -----------------------------------------------------------------------
    #[test]
    fn test_split_words_basic() {
        let result = split_words("hello world".to_string());
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn test_split_words_empty() {
        let result = split_words(String::new());
        assert_eq!(result, vec![""]);
    }

    #[test]
    fn test_split_words_no_spaces() {
        let result = split_words("hello".to_string());
        assert_eq!(result, vec!["hello"]);
    }

    #[test]
    fn test_split_words_multiple_spaces() {
        // After append_space, "a  b" becomes "a Ġ Ġb"
        // split on " " gives ["a", "Ġ", "Ġb"]
        let result = split_words(append_space("a  b".to_string()));
        assert_eq!(result, vec!["a", "Ġ", "Ġb"]);
    }

    // -----------------------------------------------------------------------
    // split_punctuation
    // -----------------------------------------------------------------------
    #[test]
    fn test_split_punctuation_basic() {
        let result = split_punctuation(vec!["hello".to_string(), "world".to_string()]);
        assert_eq!(result, vec!["hello", "world"]);
    }

    #[test]
    fn test_split_punctuation_comma() {
        let result = split_punctuation(vec!["hello,".to_string()]);
        assert_eq!(result, vec!["hello", ","]);
    }

    #[test]
    fn test_split_punctuation_period() {
        let result = split_punctuation(vec!["end.".to_string()]);
        assert_eq!(result, vec!["end", "."]);
    }

    #[test]
    fn test_split_punctuation_exclamation() {
        let result = split_punctuation(vec!["wow!".to_string()]);
        assert_eq!(result, vec!["wow", "!"]);
    }

    #[test]
    fn test_split_punctuation_question() {
        let result = split_punctuation(vec!["what?".to_string()]);
        assert_eq!(result, vec!["what", "?"]);
    }

    #[test]
    fn test_split_punctuation_colon_semicolon() {
        let result = split_punctuation(vec!["a:".to_string(), "b;".to_string()]);
        assert_eq!(result, vec!["a", ":", "b", ";"]);
    }

    #[test]
    fn test_split_punctuation_apostrophe_groups_right() {
        let result = split_punctuation(vec!["don't".to_string()]);
        assert_eq!(result, vec!["don", "'t"]);
    }

    #[test]
    fn test_split_punctuation_apostrophe_alone() {
        let result = split_punctuation(vec!["'".to_string()]);
        assert_eq!(result, vec!["'"]);
    }

    #[test]
    fn test_split_punctuation_consecutive_punctuation() {
        let result = split_punctuation(vec!["a.,b".to_string()]);
        assert_eq!(result, vec!["a", ".", ",", "b"]);
    }

    #[test]
    fn test_split_punctuation_newline_token() {
        let result = split_punctuation(vec!["helloĊworld".to_string()]);
        assert_eq!(result, vec!["hello", "Ċ", "world"]);
    }

    #[test]
    fn test_split_punctuation_tab_token() {
        let result = split_punctuation(vec!["aĉb".to_string()]);
        assert_eq!(result, vec!["a", "ĉ", "b"]);
    }

    #[test]
    fn test_split_punctuation_quotes() {
        let result = split_punctuation(vec!["\"hello\"".to_string()]);
        assert_eq!(result, vec!["\"", "hello", "\""]);
    }

    // -----------------------------------------------------------------------
    // split_to_chars
    // -----------------------------------------------------------------------
    #[test]
    fn test_split_to_chars_basic() {
        let result = split_to_chars(vec!["ab".to_string(), "c".to_string()]);
        assert_eq!(result, vec![vec!['a', 'b'], vec!['c']]);
    }

    #[test]
    fn test_split_to_chars_empty_word() {
        let result = split_to_chars(vec![String::new()]);
        assert_eq!(result, vec![vec![]]);
    }

    // -----------------------------------------------------------------------
    // pre_tokenize_str (public API)
    // -----------------------------------------------------------------------
    #[test]
    fn test_pre_tokenize_str_simple() {
        let result = pre_tokenize_str("hello world");
        assert_eq!(result, vec!["hello", "Ġworld"]);
    }

    #[test]
    fn test_pre_tokenize_str_with_punctuation() {
        let result = pre_tokenize_str("hello, world!");
        assert_eq!(result, vec!["hello", ",", "Ġworld", "!"]);
    }

    #[test]
    fn test_pre_tokenize_str_with_newline() {
        let result = pre_tokenize_str("a\nb");
        // \n → Ċ, then split_punctuation on Ċ
        assert_eq!(result, vec!["a", "Ċ", "b"]);
    }

    #[test]
    fn test_pre_tokenize_str_with_tab() {
        let result = pre_tokenize_str("a\tb");
        // \t → ĉ, then split_punctuation on ĉ
        assert_eq!(result, vec!["a", "ĉ", "b"]);
    }

    #[test]
    fn test_pre_tokenize_str_with_special_token() {
        let result = pre_tokenize_str("<|user|>hello");
        let combined: String = result.join("");
        assert!(
            combined.contains('\u{E004}'),
            "Expected U+E004 for <|user|>, got {:?}",
            combined
        );
        assert!(combined.contains("hello"), "Expected 'hello' to survive");
    }

    #[test]
    fn test_pre_tokenize_str_empty() {
        let result = pre_tokenize_str("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_pre_tokenize_str_only_punctuation() {
        let result = pre_tokenize_str("...");
        assert_eq!(result, vec![".", ".", "."]);
    }

    #[test]
    fn test_pre_tokenize_str_unicode() {
        let result = pre_tokenize_str("héllo");
        assert!(!result.is_empty());
        let combined: String = result.iter().flat_map(|s| s.chars()).collect();
        assert!(combined.contains('é'));
    }

    // -----------------------------------------------------------------------
    // pre_tokenize_text (public API, returns Vec<Vec<char>>)
    // -----------------------------------------------------------------------
    #[test]
    fn test_pre_tokenize_text_simple() {
        let result = pre_tokenize_text("ab");
        assert_eq!(result, vec![vec!['a', 'b']]);
    }

    #[test]
    fn test_pre_tokenize_text_empty() {
        let result = pre_tokenize_text("");
        assert!(result.is_empty());
    }

    #[test]
    fn test_pre_tokenize_text_with_special_tokens() {
        let result = pre_tokenize_text("<|code|>fn");
        let has_pua = result.iter().any(|word| word.contains(&'\u{E002}'));
        assert!(has_pua, "Expected U+E002 for <|code|>");
    }

    #[test]
    fn test_pre_tokenize_text_spaces() {
        let result = pre_tokenize_text("a b");
        // "a b" → append_space → "aĠ bĠ" → split → ["aĠ", "bĠ"] → chars
        assert!(!result.is_empty());
        let all_chars: Vec<char> = result.iter().flat_map(|w| w.iter()).copied().collect();
        assert!(all_chars.contains(&'Ġ'));
    }

    // -----------------------------------------------------------------------
    // lookup_special (public API)
    // -----------------------------------------------------------------------
    #[test]
    fn test_lookup_all_special_tokens() {
        for &(token_str, pua) in SPECIAL_TOKENS {
            assert_eq!(
                lookup_special(pua),
                Some(token_str),
                "lookup_special(0x{:04X}) should return {:?}",
                pua as u32,
                token_str,
            );
        }
    }

    #[test]
    fn test_lookup_special_none_for_whitespace() {
        assert!(lookup_special(' ').is_none());
        assert!(lookup_special('\n').is_none());
        assert!(lookup_special('\t').is_none());
    }

    // -----------------------------------------------------------------------
    // Integration: special tokens roundtrip
    // -----------------------------------------------------------------------
    #[test]
    fn test_special_tokens_roundtrip_through_pre_tokenize_str() {
        let text = "<|user|>hello<|/user|>";
        let words = pre_tokenize_str(text);
        let combined: String = words.join("");
        // The special tokens should have been replaced with PUA chars
        assert!(combined.contains('\u{E004}'), "missing <|user|> mapping");
        assert!(combined.contains('\u{E005}'), "missing <|/user|> mapping");
        // Normal text should survive
        assert!(combined.contains('h'));
    }

    #[test]
    fn test_multiple_special_tokens_in_text() {
        let text = "<|system|>You are<|/system|><|user|>Hi<|/user|><|assistant|>Hello<|/assistant|>";
        let words = pre_tokenize_str(text);
        let combined: String = words.join("");
        assert!(combined.contains('\u{E00C}'), "missing <|system|>");
        assert!(combined.contains('\u{E00D}'), "missing <|/system|>");
        assert!(combined.contains('\u{E004}'), "missing <|user|>");
        assert!(combined.contains('\u{E005}'), "missing <|/user|>");
        assert!(combined.contains('\u{E006}'), "missing <|assistant|>");
        assert!(combined.contains('\u{E007}'), "missing <|/assistant|>");
    }

    #[test]
    fn test_special_tokens_not_present_in_text() {
        let text = "Just regular text with no special tokens.";
        let words = pre_tokenize_str(text);
        let combined: String = words.join("");
        // No PUA characters should be present
        for &(_, pua) in SPECIAL_TOKENS {
            assert!(!combined.contains(pua), "unexpected PUA char in plain text");
        }
    }

    // -----------------------------------------------------------------------
    // Paired thinking block: both delimiters protected, reasoning is text
    // -----------------------------------------------------------------------
    #[test]
    fn test_paired_thinking_tokens() {
        let words = pre_tokenize_str("<|thinking|>weighing the options<|/thinking|>");
        let combined: String = words.join("");
        // Both delimiters map to their atomic PUA markers.
        assert!(combined.contains('\u{E02B}'), "missing <|thinking|>");
        assert!(combined.contains('\u{E02C}'), "missing <|/thinking|>");
        // The reasoning survives as ordinary text between them.
        assert!(combined.contains("weighing"), "reasoning text lost");
        assert!(combined.contains("options"), "reasoning text lost");
        // The bracket structure must not leak through as literal characters.
        assert!(!combined.contains("<|thinking"), "wrapper leaked as text");
    }
}
