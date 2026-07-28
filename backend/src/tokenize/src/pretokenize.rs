// The protected vocabulary: each marker → a Private-Use-Area char it's swapped
// for before tokenization, and (via `special_id`) a reserved single token id.
// Only markers something actually emits are listed — the chat scaffold, the
// document markup the frontend renders, and the thinking block. The PUA
// codepoints are stable, otherwise-arbitrary labels; the gaps (e.g. E008–E00B)
// are ids that once held tool-call / search scaffolding and were pruned. Re-add
// an entry here (and re-pretrain) when a feature genuinely needs a new marker.
pub const SPECIAL_TOKENS: &[(&str, char)] = &[
    ("<|endoftext|>", '\u{E000}'),
    ("<|par|>", '\u{E001}'),
    ("<|code|>", '\u{E002}'),
    ("<|/code|>", '\u{E003}'),
    ("<|user|>", '\u{E004}'),
    ("<|/user|>", '\u{E005}'),
    ("<|assistant|>", '\u{E006}'),
    ("<|/assistant|>", '\u{E007}'),
    ("<|system|>", '\u{E00C}'),
    ("<|/system|>", '\u{E00D}'),
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
    ("<|list|>", '\u{E01B}'),
    ("<|/list|>", '\u{E01C}'),
    ("<|item|>", '\u{E01D}'),
    ("<|/item|>", '\u{E01E}'),
    ("<|quote|>", '\u{E01F}'),
    ("<|/quote|>", '\u{E020}'),
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
    // Callout and reply-reference, rendered by the frontend (lib/message-parser.ts)
    // ahead of any model emitting them. `<|reply|>`'s content is a message id, not
    // prose — see `synth::NOT_REPLY_MARKUP` for why generated data stays off it.
    ("<|warn|>", '\u{E02D}'),
    ("<|/warn|>", '\u{E02E}'),
    ("<|reply|>", '\u{E02F}'),
    ("<|/reply|>", '\u{E030}'),
];

pub fn lookup_special(ch: char) -> Option<&'static str> {
    SPECIAL_TOKENS
        .iter()
        .find(|&&(_, c)| c == ch)
        .map(|&(s, _)| s)
}

/// How many special tokens are actually defined. The vocabulary is laid out as
/// `[0,256)` raw bytes, `[256, 256+SPECIAL_BLOCK)` the reserved special-token
/// block, then BPE merges above that (see `bpe::MERGE_BASE`). Reserving a
/// dedicated, corpus-independent id per special token means `<|assistant|>`,
/// `<|par|>`, … are each a *single* token with their own trainable embedding —
/// instead of being shredded into the byte triple `[0xEE,0x80,0xNN]` that every
/// marker used to share, which the model couldn't tell apart or learn cleanly.
pub const NUM_SPECIAL: usize = SPECIAL_TOKENS.len();

/// Size of the reserved id block, which is deliberately larger than
/// `NUM_SPECIAL` and **fixed**.
///
/// Merge ids are absolute and are baked into every saved `merges.bin`, so if the
/// block were sized to `NUM_SPECIAL` then adding one marker would shift
/// `MERGE_BASE` and renumber every learned merge — silently reinterpreting every
/// existing tokenizer and model against the new layout (the first merges of an
/// old `merges.bin` would decode as the newly added markers). Padding the block
/// means a new marker takes the next free slot and nothing below it moves, so
/// growing the vocabulary no longer invalidates trained models.
///
/// Slots between `NUM_SPECIAL` and here are unallocated: `special_char_from_id`
/// returns `None` for them and nothing emits them, so they cost one embedding row
/// each and no correctness.
pub const SPECIAL_BLOCK: usize = 64;

// A marker past the block would collide with merge id space, which is exactly the
// renumbering this constant exists to prevent — so it must fail the build, not
// corrupt a corpus. Raising SPECIAL_BLOCK is itself a vocabulary change: it moves
// MERGE_BASE and so needs a fresh tokenizer and a re-pretrain.
const _: () = assert!(NUM_SPECIAL <= SPECIAL_BLOCK);

/// The reserved id for a special token's PUA char, or `None` for ordinary chars.
/// Special token `i` (in `SPECIAL_TOKENS` order) owns id `256 + i`.
pub fn special_id(ch: char) -> Option<u32> {
    SPECIAL_TOKENS
        .iter()
        .position(|&(_, c)| c == ch)
        .map(|i| 256 + i as u32)
}

/// Inverse of `special_id`: a reserved id back to its PUA char, or `None` if the
/// id isn't in the special-token block. Used by `decode` to turn an atomic
/// special id back into its marker text.
pub fn special_char_from_id(id: u32) -> Option<char> {
    let idx = id.checked_sub(256)? as usize;
    if idx < NUM_SPECIAL {
        Some(SPECIAL_TOKENS[idx].1)
    } else {
        None
    }
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
            if lookup_special(c).is_some() {
                // A special-token PUA char is always its own pretoken unit, so it
                // maps to a single reserved id and never glues to adjacent text
                // (e.g. `<|assistant|>Hello` → `[<|assistant|>]` + `[Hello]`).
                if !current.is_empty() {
                    result.push(current.clone());
                    current.clear();
                }
                result.push(c.to_string());
            } else if matches!(c, ',' | '.' | '!' | '?' | '"' | ':' | ';' | 'Ċ' | 'ĉ') {
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

    // -----------------------------------------------------------------------
    // The reserved block
    // -----------------------------------------------------------------------

    // The property the padding exists to buy: every marker's id sits inside the
    // block, so merge ids (which start at its end) never have to move to make
    // room for a new one.
    #[test]
    fn test_every_special_id_fits_inside_the_reserved_block() {
        for &(name, pua) in SPECIAL_TOKENS {
            let id = special_id(pua).expect("marker has no reserved id");
            assert!(
                (256..256 + SPECIAL_BLOCK as u32).contains(&id),
                "{name} at id {id} is outside the reserved block",
            );
        }
        assert!(NUM_SPECIAL <= SPECIAL_BLOCK, "more markers than slots");
    }

    // The slack between the markers and the end of the block is unallocated, not
    // a second marker table: an id in it must not decode as some other marker.
    #[test]
    fn test_unallocated_block_slots_decode_to_nothing() {
        for i in NUM_SPECIAL..SPECIAL_BLOCK {
            let id = 256 + i as u32;
            assert_eq!(special_char_from_id(id), None, "id {id} decoded as a marker");
        }
    }

    #[test]
    fn test_warn_and_reply_markers_are_protected() {
        let words = pre_tokenize_str("<|warn|>mind this<|/warn|><|reply|>abc<|/reply|>");
        let combined: String = words.join("");
        for marker in ["<|warn|>", "<|/warn|>", "<|reply|>", "<|/reply|>"] {
            let pua = SPECIAL_TOKENS
                .iter()
                .find(|&&(name, _)| name == marker)
                .map(|&(_, pua)| pua)
                .expect("marker missing from the vocabulary");
            assert!(combined.contains(pua), "{marker} was not protected");
        }
        assert!(combined.contains("mind"), "callout text lost");
        assert!(!combined.contains("<|warn"), "wrapper leaked as text");
    }
}
