use std::collections::{BinaryHeap, HashMap};
use std::fs::File;
use std::io;
use std::io::Write;

use crate::pretokenize::{lookup_special, special_char_from_id, special_id, NUM_SPECIAL};

/// First id available to BPE merges: raw bytes fill `[0,256)` and the reserved
/// special-token block fills `[256, 256+NUM_SPECIAL)`, so learned merges start
/// here. `train` counts up from this, and `vocab_size` folds the same offset in.
pub const MERGE_BASE: u32 = 256 + NUM_SPECIAL as u32;

/// Total vocabulary size for a tokenizer with `num_merges` learned merges:
/// bytes + reserved special tokens + merges. The single source of truth for
/// vocab sizing — every caller (model init, job metadata) routes through here so
/// the offset can't drift between the tokenizer and the model.
pub fn vocab_size(num_merges: usize) -> i64 {
    MERGE_BASE as i64 + num_merges as i64
}

pub fn train(
    pre_tokenized: Vec<Vec<char>>,
    max_vocab_size: Option<u32>,
    on_log: impl Fn(&str),
) -> Result<(i64, Vec<Vec<u32>>, Vec<((u32, u32), u32)>), io::Error> {
    let mut corpus = convert_to_bytes(pre_tokenized);
    let (mut pair_counts, mut pair_index) = count_pairs(&corpus);
    let mut heap = create_heap(pair_counts.clone());

    let max = max_vocab_size.unwrap_or(32768).max(MERGE_BASE);
    // Merge ids start above the reserved byte + special-token block, so a learned
    // merge can never collide with a raw byte or an atomic special-token id.
    let mut token: u32 = MERGE_BASE;
    let mut merges: Vec<((u32, u32), u32)> = Vec::new();

    let log_every = ((max - 256) / 50).max(1);

    while heap.len() > 0 && max > token {
        let (count, pair) = heap.pop().unwrap();

        let actual_count = *pair_counts.get(&pair).unwrap_or(&0);

        if actual_count != count {
            continue;
        }

        if actual_count == 0 {
            continue;
        }

        merges.push((pair, token));
        print!("\rMerge {}: {:?} -> {}          ", token - 256, pair, token);
        std::io::stdout().flush().unwrap();
        if (token - 256) % log_every == 0 {
            on_log(&format!("BPE merge {}/{}", token - 256, max - 256));
        }

        apply_merge(
            &mut corpus,
            pair,
            token,
            &mut pair_counts,
            &mut pair_index,
            &mut heap,
        );

        pair_counts.insert(pair, 0);
        pair_index.remove(&pair);

        token += 1;
    }

    println!();
    println!("Training complete! {} merges performed.", merges.len());

    let vocab = vocab_size(merges.len());

    Ok((vocab, corpus, merges))
}

fn convert_to_bytes(pre_tokenized: Vec<Vec<char>>) -> Vec<Vec<u32>> {
    pre_tokenized
        .iter()
        .map(|word| {
            word.iter()
                .flat_map(|c| match special_id(*c) {
                    // A special token enters BPE as its atomic reserved id, not its
                    // raw bytes — since the pretokenizer isolates it into its own
                    // word it has no neighbours to pair with, so it's never merged
                    // and keeps that id straight through training.
                    Some(id) => vec![id],
                    None => c.to_string().into_bytes().into_iter().map(|b| b as u32).collect(),
                })
                .collect()
        })
        .collect()
}

fn count_pairs(
    corpus: &Vec<Vec<u32>>,
) -> (
    HashMap<(u32, u32), usize>,
    HashMap<(u32, u32), Vec<(usize, usize)>>,
) {
    let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
    let mut pair_index: HashMap<(u32, u32), Vec<(usize, usize)>> = HashMap::new();

    for (word_idx, word) in corpus.iter().enumerate() {
        for (pos, pair) in word.windows(2).enumerate() {
            let (a, b) = (pair[0], pair[1]);
            *pair_counts.entry((a, b)).or_insert(0) += 1;

            pair_index
                .entry((a, b))
                .or_insert_with(Vec::new)
                .push((word_idx, pos));
        }
    }

    (pair_counts, pair_index)
}

fn create_heap(pair_counts: HashMap<(u32, u32), usize>) -> BinaryHeap<(usize, (u32, u32))> {
    let mut heap: BinaryHeap<(usize, (u32, u32))> = BinaryHeap::new();

    for (pair, count) in pair_counts {
        heap.push((count, pair));
    }

    heap
}

fn apply_merge(
    corpus: &mut Vec<Vec<u32>>,
    pair: (u32, u32),
    token: u32,
    pair_counts: &mut HashMap<(u32, u32), usize>,
    pair_index: &mut HashMap<(u32, u32), Vec<(usize, usize)>>,
    heap: &mut BinaryHeap<(usize, (u32, u32))>,
) {
    let locations = match pair_index.get(&pair) {
        Some(locs) => locs.clone(),
        _ => return,
    };

    let mut by_word: HashMap<usize, Vec<usize>> = HashMap::new();
    for (word_idx, pos) in &locations {
        by_word.entry(*word_idx).or_default().push(*pos);
    }

    for (word_idx, mut positions) in by_word {
        positions.sort_unstable();
        positions.reverse();

        for pos in positions {
            let word = &corpus[word_idx];
            if pos + 1 >= word.len() || word[pos] != pair.0 || word[pos + 1] != pair.1 {
                continue;
            }

            if pos > 0 {
                let left = (corpus[word_idx][pos - 1], pair.0);
                *pair_counts.entry(left).or_insert(0) =
                    pair_counts.get(&left).unwrap_or(&0).saturating_sub(1);
            }
            if pos + 2 < corpus[word_idx].len() {
                let right = (pair.1, corpus[word_idx][pos + 2]);
                *pair_counts.entry(right).or_insert(0) =
                    pair_counts.get(&right).unwrap_or(&0).saturating_sub(1);
            }

            corpus[word_idx].remove(pos + 1);
            corpus[word_idx][pos] = token;

            if pos > 0 {
                let new_left = (corpus[word_idx][pos - 1], token);
                *pair_counts.entry(new_left).or_insert(0) += 1;
                heap.push((*pair_counts.get(&new_left).unwrap(), new_left));
                pair_index
                    .entry(new_left)
                    .or_default()
                    .push((word_idx, pos - 1));
            }
            if pos + 1 < corpus[word_idx].len() {
                let new_right = (token, corpus[word_idx][pos + 1]);
                *pair_counts.entry(new_right).or_insert(0) += 1;
                heap.push((*pair_counts.get(&new_right).unwrap(), new_right));
                pair_index
                    .entry(new_right)
                    .or_default()
                    .push((word_idx, pos));
            }
        }
    }
}

pub fn save_merges(merges: &[((u32, u32), u32)], path: &str) -> Result<(), io::Error> {
    let mut file = File::create(path)?;
    for ((a, b), token) in merges {
        file.write_all(&a.to_le_bytes())?;
        file.write_all(&b.to_le_bytes())?;
        file.write_all(&token.to_le_bytes())?;
    }
    Ok(())
}

// Serializes a flat token-id stream as little-endian u32s — the `corpus.bin`
// format `llm::train` reads back. The BPE `train` returns the corpus split per
// word (`Vec<Vec<u32>>`); the caller flattens it before saving here.
pub fn save_corpus(corpus: &[u32], path: &str) -> Result<(), io::Error> {
    let mut file = File::create(path)?;
    for token in corpus {
        file.write_all(&token.to_le_bytes())?;
    }
    Ok(())
}

pub fn load_merges(path: &str) -> Result<Vec<((u32, u32), u32)>, io::Error> {
    let data = std::fs::read(path)?;
    let mut merges = Vec::new();
    // chunks_exact(12) yields exactly-12-byte chunks, so each fixed 4-byte
    // sub-slice below is guaranteed to be 4 bytes and try_into can't fail. A
    // trailing partial chunk (corrupt file) is dropped by chunks_exact.
    for chunk in data.chunks_exact(12) {
        let a = u32::from_le_bytes(chunk[0..4].try_into().unwrap());
        let b = u32::from_le_bytes(chunk[4..8].try_into().unwrap());
        let token = u32::from_le_bytes(chunk[8..12].try_into().unwrap());
        merges.push(((a, b), token));
    }
    Ok(merges)
}

pub fn encode(text: &str, merges: &[((u32, u32), u32)]) -> Vec<u32> {
    let mut pair_to_token: HashMap<(u32, u32), u32> = HashMap::new();
    for &((a, b), token) in merges {
        pair_to_token.insert((a, b), token);
    }

    let words = crate::pretokenize::pre_tokenize_str(text);
    let mut result = Vec::new();
    for word in words {
        // The pretokenizer isolates each special token into its own word, so a
        // one-char word that is a special PUA char maps straight to its atomic
        // reserved id — matching exactly how `convert_to_bytes` fed it to BPE.
        let mut chars = word.chars();
        if let (Some(c), None) = (chars.next(), chars.next()) {
            if let Some(id) = special_id(c) {
                result.push(id);
                continue;
            }
        }

        let mut ids: Vec<u32> = word
            .chars()
            .flat_map(|c| {
                let s = c.to_string();
                s.into_bytes().into_iter().map(|b| b as u32)
            })
            .collect();

        loop {
            let mut best_pair: Option<(usize, u32)> = None;
            for (pos, w) in ids.windows(2).enumerate() {
                let key = (w[0], w[1]);
                if let Some(&token) = pair_to_token.get(&key) {
                    match best_pair {
                        Some((_, best_token)) if token > best_token => {}
                        _ => best_pair = Some((pos, token)),
                    }
                }
            }
            match best_pair {
                Some((pos, token)) => {
                    ids[pos] = token;
                    ids.remove(pos + 1);
                }
                _ => break,
            }
        }
        result.extend(ids);
    }
    result
}

pub fn decode(ids: &[u32], merges: &[((u32, u32), u32)]) -> String {
    let mut token_to_pair: HashMap<u32, (u32, u32)> = HashMap::new();
    for &((a, b), token) in merges {
        token_to_pair.insert(token, (a, b));
    }

    let mut bytes = Vec::new();
    for &id in ids {
        let mut stack = vec![id];
        while let Some(t) = stack.pop() {
            if t < 256 {
                bytes.push(t as u8);
            } else if let Some(ch) = special_char_from_id(t) {
                // Atomic special-token id → its PUA char's utf-8 bytes; the
                // `replace_special_chars` post-pass turns that back into the
                // marker string (e.g. `<|assistant|>`).
                let mut buf = [0u8; 4];
                bytes.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
            } else if let Some(&(a, b)) = token_to_pair.get(&t) {
                stack.push(b);
                stack.push(a);
            }
        }
    }
    let decoded = String::from_utf8_lossy(&bytes)
        .replace('Ġ', " ")
        .replace('Ċ', "\n")
        .replace('ĉ', "\t")
        .replace('\u{FFFD}', "")
        .to_string();

    replace_special_chars(&decoded)
}

fn replace_special_chars(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    for c in text.chars() {
        match lookup_special(c) {
            Some(marker) => result.push_str(marker),
            _ => result.push(c),
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // convert_to_bytes
    // -----------------------------------------------------------------------
    #[test]
    fn test_convert_to_bytes_ascii() {
        let input = vec![vec!['a', 'b']];
        let output = convert_to_bytes(input);
        assert_eq!(output, vec![vec![97u32, 98]]);
    }

    #[test]
    fn test_convert_to_bytes_unicode_multi_byte() {
        // 'é' (U+00E9) encodes as [195, 169] in UTF-8
        let input = vec![vec!['é']];
        let output = convert_to_bytes(input);
        assert_eq!(output, vec![vec![195u32, 169]]);
    }

    #[test]
    fn test_convert_to_bytes_empty_word() {
        let input = vec![vec![]];
        let output = convert_to_bytes(input);
        assert_eq!(output, vec![vec![]]);
    }

    #[test]
    fn test_convert_to_bytes_multiple_words() {
        let input = vec![vec!['a', 'a'], vec!['b']];
        let output = convert_to_bytes(input);
        assert_eq!(output, vec![vec![97u32, 97], vec![98]]);
    }

    // -----------------------------------------------------------------------
    // count_pairs
    // -----------------------------------------------------------------------
    #[test]
    fn test_count_pairs_single_word() {
        let corpus = vec![vec![97u32, 98, 99]];
        let (counts, index) = count_pairs(&corpus);
        assert_eq!(counts.len(), 2);
        assert_eq!(counts[&(97, 98)], 1);
        assert_eq!(counts[&(98, 99)], 1);
        assert_eq!(index[&(97, 98)], vec![(0, 0)]);
        assert_eq!(index[&(98, 99)], vec![(0, 1)]);
    }

    #[test]
    fn test_count_pairs_multiple_words() {
        let corpus = vec![vec![97u32, 98], vec![97u32, 98]];
        let (counts, index) = count_pairs(&corpus);
        assert_eq!(counts[&(97, 98)], 2);
        let mut positions = index[&(97, 98)].clone();
        positions.sort_unstable();
        assert_eq!(positions, vec![(0, 0), (1, 0)]);
    }

    #[test]
    fn test_count_pairs_single_byte_word() {
        let corpus = vec![vec![97u32]];
        let (counts, _index) = count_pairs(&corpus);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_count_pairs_empty_corpus() {
        let corpus: Vec<Vec<u32>> = vec![];
        let (counts, _index) = count_pairs(&corpus);
        assert!(counts.is_empty());
    }

    #[test]
    fn test_count_pairs_overlapping() {
        let corpus = vec![vec![97u32, 97, 97]];
        let (counts, index) = count_pairs(&corpus);
        assert_eq!(counts[&(97, 97)], 2);
        let mut positions = index[&(97, 97)].clone();
        positions.sort_unstable();
        assert_eq!(positions, vec![(0, 0), (0, 1)]);
    }

    // -----------------------------------------------------------------------
    // create_heap
    // -----------------------------------------------------------------------
    #[test]
    fn test_create_heap_orders_by_count_descending() {
        let mut counts = HashMap::new();
        counts.insert((97u32, 98u32), 5usize);
        counts.insert((98u32, 99u32), 2usize);

        let mut heap = create_heap(counts);
        let (top_count, top_pair) = heap.pop().unwrap();
        assert_eq!(top_count, 5);
        assert_eq!(top_pair, (97, 98));
    }

    #[test]
    fn test_create_heap_empty() {
        let counts = HashMap::new();
        let mut heap = create_heap(counts);
        assert!(heap.pop().is_none());
    }

    // -----------------------------------------------------------------------
    // apply_merge
    // -----------------------------------------------------------------------
    fn setup_merge_state(
        corpus: Vec<Vec<u32>>,
    ) -> (
        Vec<Vec<u32>>,
        HashMap<(u32, u32), usize>,
        HashMap<(u32, u32), Vec<(usize, usize)>>,
        BinaryHeap<(usize, (u32, u32))>,
    ) {
        let (pair_counts, pair_index) = count_pairs(&corpus);
        let heap = create_heap(pair_counts.clone());
        (corpus, pair_counts, pair_index, heap)
    }

    #[test]
    fn test_apply_merge_basic() {
        let (mut corpus, mut pair_counts, mut pair_index, mut heap) =
            setup_merge_state(vec![vec![97u32, 97]]);

        apply_merge(
            &mut corpus,
            (97, 97),
            256,
            &mut pair_counts,
            &mut pair_index,
            &mut heap,
        );

        assert_eq!(corpus, vec![vec![256]]);
    }

    #[test]
    fn test_apply_merge_middle_of_word() {
        let (mut corpus, mut pair_counts, mut pair_index, mut heap) =
            setup_merge_state(vec![vec![97u32, 98, 99]]);

        apply_merge(
            &mut corpus,
            (97, 98),
            256,
            &mut pair_counts,
            &mut pair_index,
            &mut heap,
        );

        // After merging (97,98) → 256 at pos 0, word becomes [256, 99]
        assert_eq!(corpus, vec![vec![256, 99]]);

        // New pair (256, 99) should be tracked
        assert_eq!(*pair_counts.get(&(256, 99)).unwrap_or(&0), 1);
        assert!(pair_index.contains_key(&(256, 99)));

        // Original right pair (98, 99) should have been decremented
        assert_eq!(*pair_counts.get(&(98, 99)).unwrap_or(&0), 0);

        // Heap should have the new pair
        let contains_new = heap.iter().any(|&(_, p)| p == (256, 99));
        assert!(contains_new, "heap should contain new pair (256, 99)");
    }

    #[test]
    fn test_apply_merge_end_of_word() {
        let (mut corpus, mut pair_counts, mut pair_index, mut heap) =
            setup_merge_state(vec![vec![97u32, 98, 99]]);

        apply_merge(
            &mut corpus,
            (98, 99),
            256,
            &mut pair_counts,
            &mut pair_index,
            &mut heap,
        );

        // After merging (98,99) → 256 at pos 1, word becomes [97, 256]
        assert_eq!(corpus, vec![vec![97, 256]]);

        // New pair (97, 256) should be tracked
        assert_eq!(*pair_counts.get(&(97, 256)).unwrap_or(&0), 1);
        assert!(pair_index.contains_key(&(97, 256)));

        // Original left pair (97, 98) should have been decremented
        assert_eq!(*pair_counts.get(&(97, 98)).unwrap_or(&0), 0);
    }

    #[test]
    fn test_apply_merge_skips_already_merged_position() {
        // Positions 0 and 1 both have (97,97) in "aaa"
        let (mut corpus, mut pair_counts, mut pair_index, mut heap) =
            setup_merge_state(vec![vec![97u32, 97, 97]]);

        apply_merge(
            &mut corpus,
            (97, 97),
            256,
            &mut pair_counts,
            &mut pair_index,
            &mut heap,
        );

        // After first merge (pos=1, reversed), word becomes [97, 256].
        // The second position (pos=0) no longer has (97,97), so it's skipped.
        assert_eq!(corpus, vec![vec![97, 256]]);
    }

    // -----------------------------------------------------------------------
    // train
    // -----------------------------------------------------------------------
    #[test]
    fn test_train_single_repeated_pair() {
        let pre_tokenized = vec![vec!['a', 'a']];
        let (vocab, corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();

        assert!(vocab > 256, "vocab should grow beyond byte-only tokens");
        assert!(!merges.is_empty(), "should have at least one merge");

        // The first merge should combine 'a' (97) with itself, and take the first
        // id above the reserved byte + special-token block.
        let first = &merges[0];
        assert_eq!(first.0, (97, 97));
        assert_eq!(first.1, MERGE_BASE);

        // corpus should contain the merged token
        assert_eq!(corpus, vec![vec![MERGE_BASE]]);
    }

    #[test]
    fn test_train_multiple_merges() {
        // "aabb" produces pairs (97,97), (97,98), (98,98)
        let pre_tokenized = vec![vec!['a', 'a', 'b', 'b']];
        let (_vocab, _corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();

        // We expect at least 2 merges (both repeated-byte pairs)
        assert!(
            merges.len() >= 2,
            "expected at least 2 merges, got {}",
            merges.len()
        );
    }

    #[test]
    fn test_train_no_pairs_produces_no_merges() {
        // Single byte has no pairs to merge
        let pre_tokenized = vec![vec!['a']];
        let (vocab, corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();

        // No merges → vocab is just the bytes + reserved special-token block.
        assert_eq!(vocab, vocab_size(0));
        assert!(merges.is_empty());
        assert_eq!(corpus, vec![vec![97]]);
    }

    #[test]
    fn test_train_multiple_words() {
        let pre_tokenized = vec![vec!['a', 'a'], vec!['b', 'b']];
        let (_vocab, _corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();

        // Both (97,97) and (98,98) should be merged
        let pairs: Vec<(u32, u32)> = merges.iter().map(|(p, _)| *p).collect();
        assert!(pairs.contains(&(97, 97)), "should merge (97,97)");
        assert!(pairs.contains(&(98, 98)), "should merge (98,98)");
    }

    // -----------------------------------------------------------------------
    // encode / decode
    // -----------------------------------------------------------------------
    #[test]
    fn test_encode_simple_after_train() {
        let pre_tokenized = vec![vec!['a', 'a']];
        let (_vocab, _corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();

        let encoded = encode("aa", &merges);
        // Should produce a single merged token (or at least fewer bytes)
        assert!(!encoded.is_empty());
    }

    #[test]
    fn test_encode_no_merges_applicable() {
        let merges = vec![((97u32, 98u32), 256u32)];
        let encoded = encode("xyz", &merges);
        // No merge rule applies → identity: bytes of each char
        assert_eq!(encoded, vec![120, 121, 122]);
    }

    #[test]
    fn test_decode_simple() {
        // Merge ids live above the reserved block now; use MERGE_BASE so the id
        // isn't mistaken for an atomic special token.
        let merges = vec![((97u32, 97u32), MERGE_BASE)];
        let decoded = decode(&[MERGE_BASE], &merges);
        assert_eq!(decoded, "aa");
    }

    #[test]
    fn test_decode_identity_bytes() {
        let merges = vec![];
        let decoded = decode(&[97, 98, 99], &merges);
        assert_eq!(decoded, "abc");
    }

    // The core of fix 1: a special token is one atomic id (in the reserved block),
    // not the three shared bytes it used to shred into — and it roundtrips through
    // encode/decode even with no merges, gluing cleanly to adjacent text.
    #[test]
    fn test_special_token_is_atomic_id() {
        use crate::pretokenize::special_id;
        let merges: Vec<((u32, u32), u32)> = vec![];

        // `<|assistant|>` → exactly one id, and it's its reserved id (< MERGE_BASE).
        let ids = encode("<|assistant|>", &merges);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], special_id('\u{E006}').unwrap());
        assert!(ids[0] < MERGE_BASE && ids[0] >= 256);

        // Two different markers get two different ids (they no longer alias on a
        // shared `[0xEE,0x80]` prefix).
        assert_ne!(encode("<|par|>", &merges)[0], encode("<|/par|>", &merges)[0]);

        // A marker glued to text splits into [marker id, …text tokens], and the
        // whole thing decodes back to the original string.
        let glued = encode("<|assistant|>hi", &merges);
        assert_eq!(glued[0], special_id('\u{E006}').unwrap());
        assert_eq!(decode(&glued, &merges), "<|assistant|>hi");
    }

    #[test]
    fn test_decode_unicode_preserved() {
        let merges = vec![];
        let decoded = decode(&[195, 169], &merges);
        assert_eq!(decoded, "é");
    }

    #[test]
    fn test_encode_decode_roundtrip_simple() {
        let text = "aa";
        let pre_tokenized = vec![vec!['a', 'a']];
        let (_vocab, _corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();

        let encoded = encode(text, &merges);
        let decoded = decode(&encoded, &merges);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_decode_roundtrip_with_unicode() {
        let text = "héllo";
        let pre_tokenized = crate::pretokenize::pre_tokenize_text(text);
        let (_vocab, _corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();

        let encoded = encode(text, &merges);
        let decoded = decode(&encoded, &merges);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_encode_empty_string() {
        let merges = vec![];
        let encoded = encode("", &merges);
        assert!(encoded.is_empty());
    }

    #[test]
    fn test_decode_empty_slice() {
        let merges = vec![((97u32, 97u32), 256u32)];
        let decoded = decode(&[], &merges);
        assert_eq!(decoded, "");
    }

    // -----------------------------------------------------------------------
    // save_merges / load_merges
    // -----------------------------------------------------------------------
    #[test]
    fn test_save_load_merges_roundtrip() {
        let merges = vec![((97u32, 98u32), 256u32), ((99u32, 100u32), 257u32)];
        let path = std::env::temp_dir().join("test_bpe_merges.bin");

        // Remove leftover from a previous run
        let _ = std::fs::remove_file(&path);

        save_merges(&merges, path.to_str().unwrap()).unwrap();
        let loaded = load_merges(path.to_str().unwrap()).unwrap();

        assert_eq!(merges, loaded);

        // Clean up
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_save_load_empty_merges() {
        let merges: Vec<((u32, u32), u32)> = vec![];
        let path = std::env::temp_dir().join("test_bpe_merges_empty.bin");
        let _ = std::fs::remove_file(&path);

        save_merges(&merges, path.to_str().unwrap()).unwrap();
        let loaded = load_merges(path.to_str().unwrap()).unwrap();

        assert!(loaded.is_empty());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_merges_invalid_path() {
        let result = load_merges("/nonexistent/path/merges.bin");
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // save_corpus
    // -----------------------------------------------------------------------
    #[test]
    fn test_save_corpus_little_endian_layout() {
        // 97 -> [0x61,0,0,0], 256 -> [0,1,0,0]: same flat LE-u32 layout
        // `llm::train` decodes via chunks_exact(4) + u32::from_le_bytes.
        let corpus = vec![97u32, 256u32];
        let path = std::env::temp_dir().join("test_bpe_corpus.bin");
        let _ = std::fs::remove_file(&path);

        save_corpus(&corpus, path.to_str().unwrap()).unwrap();
        let bytes = std::fs::read(&path).unwrap();
        let roundtrip: Vec<u32> = bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        assert_eq!(bytes, vec![0x61, 0, 0, 0, 0, 1, 0, 0]);
        assert_eq!(roundtrip, corpus);

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // decode with special tokens support
    // -----------------------------------------------------------------------
    #[test]
    fn test_decode_with_special_token_roundtrip() {
        // U+E000 (<|endoftext|>) in UTF-8 is [0xEE, 0x80, 0x80]
        let merges = vec![];
        // The three bytes of U+E000 as raw bytes (no merge rules)
        let decoded = decode(&[0xEE, 0x80, 0x80], &merges);
        assert_eq!(decoded, "<|endoftext|>");
    }
}
#[test]
fn test_bpe_phantom_merges() {
    let pre_tokenized = vec![vec!['a', 'a', 'a', 'a']];
    let (vocab, corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();
    println!("Merges: {:?}", merges);
    println!("Corpus: {:?}", corpus);
    assert!(merges.len() <= 2);
}
#[test]
fn test_overlap() {
    let pre_tokenized = vec![vec!['a', 'a', 'a']];
    let (_vocab, corpus, merges) = train(pre_tokenized, None, |_| {}).unwrap();
    println!("Merges: {:?}", merges);
    let encoded = encode("aaa", &merges);
    println!("Encoded: {:?}", encoded);
    println!("Corpus: {:?}", corpus);
    // corpus should match encoded exactly!
    assert_eq!(encoded, corpus[0]);
}
