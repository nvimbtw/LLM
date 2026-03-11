use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::Write;
use std::cmp::Ordering;
use crate::io::*;

#[derive(Eq, PartialEq)]
struct MergeCandidate {
    rank: usize,
    pair: (u32, u32),
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Lower rank is better (higher priority)
        other.rank.cmp(&self.rank)
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn encoder() {
    let vocab_path = "data/pairs.bin";
    let vocab = read_vocab(vocab_path).expect("Failed to read tokenizer binary file");
    let text = read_text_file("data/input.txt");
    println!("Encoding {} bytes of text...", text.len());
    let tokens = encode(text, vocab);
    write_tokens(&tokens, "data/tokens.bin").expect("Failed to write tokens to file");
    println!("Encoding complete. Tokens saved to data/tokens.bin");
}

pub fn encode(text: String, vocab: Vec<(u32, u32)>) -> Vec<u32> {
    let mut tokens: Vec<u32> = text.bytes().map(|b| b as u32).collect();
    if tokens.len() < 2 { return tokens; }

    let mut vocab_map: HashMap<(u32, u32), usize> = HashMap::new();
    for (i, &pair) in vocab.iter().enumerate() {
        vocab_map.insert(pair, i);
    }

    let mut next: Vec<Option<usize>> = (1..=tokens.len()).map(|i| if i < tokens.len() { Some(i) } else { None }).collect();
    let mut prev: Vec<Option<usize>> = (0..tokens.len()).map(|i| if i > 0 { Some(i - 1) } else { None }).collect();
    
    // Track where each pair occurs
    let mut pair_positions: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();
    let mut heap = BinaryHeap::new();
    let mut in_heap: HashSet<(u32, u32)> = HashSet::new();

    println!("Initialising pair tracking...");
    for i in 0..tokens.len() - 1 {
        let pair = (tokens[i], tokens[i+1]);
        if let Some(&rank) = vocab_map.get(&pair) {
            pair_positions.entry(pair).or_default().insert(i);
            if in_heap.insert(pair) {
                heap.push(MergeCandidate { rank, pair });
            }
        }
    }

    let total_merges = vocab.len();
    let mut merges_done = 0;

    println!("Starting merges...");
    while let Some(MergeCandidate { rank, pair }) = heap.pop() {
        in_heap.remove(&pair);
        
        let positions: Vec<usize> = match pair_positions.get(&pair) {
            Some(p) if !p.is_empty() => p.iter().copied().collect(),
            _ => continue,
        };
        
        merges_done += 1;
        if merges_done % 100 == 0 {
            print!("\rApplying vocab rule {}/{}", merges_done, total_merges);
            std::io::stdout().flush().unwrap();
        }

        let mut sorted_positions = positions;
        sorted_positions.sort_unstable();

        let new_token_id = (256 + rank) as u32;

        for &pos in &sorted_positions {
            // Validate match
            if tokens[pos] != pair.0 { continue; }
            let next_idx = match next[pos] {
                Some(ni) if tokens[ni] == pair.1 => ni,
                _ => continue,
            };

            // Remove old neighbor pairs from tracking
            if let Some(p) = prev[pos] {
                let left_pair = (tokens[p], tokens[pos]);
                if let Some(pos_set) = pair_positions.get_mut(&left_pair) {
                    pos_set.remove(&p);
                }
            }
            if let Some(nn) = next[next_idx] {
                let right_pair = (tokens[next_idx], tokens[nn]);
                if let Some(pos_set) = pair_positions.get_mut(&right_pair) {
                    pos_set.remove(&next_idx);
                }
            }

            // Perform Merge
            tokens[pos] = new_token_id;
            let nn_opt = next[next_idx];
            next[pos] = nn_opt;
            if let Some(nn) = nn_opt {
                prev[nn] = Some(pos);
            }

            // Add new neighbor pairs to tracking
            if let Some(p) = prev[pos] {
                let left_pair = (tokens[p], tokens[pos]);
                if let Some(&r) = vocab_map.get(&left_pair) {
                    pair_positions.entry(left_pair).or_default().insert(p);
                    if in_heap.insert(left_pair) {
                        heap.push(MergeCandidate { rank: r, pair: left_pair });
                    }
                }
            }
            if let Some(nn) = next[pos] {
                let right_pair = (tokens[pos], tokens[nn]);
                if let Some(&r) = vocab_map.get(&right_pair) {
                    pair_positions.entry(right_pair).or_default().insert(pos);
                    if in_heap.insert(right_pair) {
                        heap.push(MergeCandidate { rank: r, pair: right_pair });
                    }
                }
            }
        }
        // Pair is fully processed for now
        pair_positions.remove(&pair);
    }
    println!("\nEncoding finished.");

    let mut final_tokens = Vec::new();
    let mut curr = Some(0);
    while let Some(i) = curr {
        final_tokens.push(tokens[i]);
        curr = next[i];
    }
    final_tokens
}
