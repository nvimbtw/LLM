use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::io::Write;

use crate::io::io::{read_text_file, write_vocab};

#[derive(Eq, PartialEq, Clone, Copy, Debug)]
struct PairCount {
    pair: (u32, u32),
    count: usize,
    generation: usize,
}

impl Ord for PairCount {
    fn cmp(&self, other: &Self) -> Ordering {
        self.count
            .cmp(&other.count)
            .then_with(|| self.pair.cmp(&other.pair))
    }
}

impl PartialOrd for PairCount {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub fn build_vocab() {
    println!("Building vocab");
    let input = read_text_file("inputs/input.txt");
    let input_bytes = input.bytes().map(|b| b as u32).collect();
    let vocab = bpe(input_bytes, 10000);
    write_vocab(&vocab, "data/pairs.bin").unwrap();
}

struct BpeState {
    tokens: Vec<u32>,
    next: Vec<Option<usize>>,
    prev: Vec<Option<usize>>,
    pair_counts: HashMap<(u32, u32), usize>,
    pair_positions: HashMap<(u32, u32), HashSet<usize>>,
    heap: BinaryHeap<PairCount>,
    generation_map: HashMap<(u32, u32), usize>,
    current_generation: usize,
}

impl BpeState {
    fn new(input: Vec<u32>) -> Self {
        let mut state = BpeState {
            tokens: input.clone(),
            next: (1..=input.len())
                .map(|i| if i < input.len() { Some(i) } else { None })
                .collect(),
            prev: (0..input.len())
                .map(|i| if i > 0 { Some(i - 1) } else { None })
                .collect(),
            pair_counts: HashMap::new(),
            pair_positions: HashMap::new(),
            heap: BinaryHeap::new(),
            generation_map: HashMap::new(),
            current_generation: 0,
        };

        for i in 0..state.tokens.len() - 1 {
            let pair = (state.tokens[i], state.tokens[i + 1]);
            *state.pair_counts.entry(pair).or_insert(0) += 1;
            state
                .pair_positions
                .entry(pair)
                .or_insert_with(HashSet::new)
                .insert(i);
        }

        for (&pair, &count) in &state.pair_counts {
            state.heap.push(PairCount {
                pair,
                count,
                generation: 0,
            });
            state.generation_map.insert(pair, 0);
        }

        state
    }

    fn find_best_pair(&mut self) -> Option<(u32, u32)> {
        loop {
            if let Some(pc) = self.heap.pop() {
                if let Some(&current_gen) = self.generation_map.get(&pc.pair) {
                    if pc.generation == current_gen && pc.count > 0 {
                        return Some(pc.pair);
                    }
                }
            } else {
                return None;
            }
        }
    }

    fn merge_pair(&mut self, pair: (u32, u32), replacement: u32) {
        let positions: Vec<usize> = self
            .pair_positions
            .get(&pair)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();

        self.current_generation += 1;
        let mut affected_pairs: HashSet<(u32, u32)> = HashSet::new();

        for &pos in &positions {
            if self.next[pos].is_none() || self.tokens[pos] != pair.0 {
                continue;
            }
            if let Some(next_pos) = self.next[pos] {
                if self.tokens[next_pos] != pair.1 {
                    continue;
                }

                if let Some(p) = self.prev[pos] {
                    let left_pair = (self.tokens[p], self.tokens[pos]);
                    affected_pairs.insert(left_pair);
                }
                if let Some(nn) = self.next[next_pos] {
                    let right_pair = (self.tokens[next_pos], self.tokens[nn]);
                    affected_pairs.insert(right_pair);
                }

                self.tokens[pos] = replacement;

                self.next[pos] = self.next[next_pos];
                if let Some(nn) = self.next[next_pos] {
                    self.prev[nn] = Some(pos);
                }

                if let Some(p) = self.prev[pos] {
                    let new_pair = (self.tokens[p], replacement);
                    affected_pairs.insert(new_pair);
                }
                if let Some(nn) = self.next[pos] {
                    let new_pair = (replacement, self.tokens[nn]);
                    affected_pairs.insert(new_pair);
                }
            }
        }

        self.pair_counts.remove(&pair);
        self.pair_positions.remove(&pair);
        self.update_affected_pairs(affected_pairs);
    }

    fn update_affected_pairs(&mut self, affected_pairs: HashSet<(u32, u32)>) {
        for &affected_pair in &affected_pairs {
            let mut count = 0;
            let mut positions_set = HashSet::new();
            let mut i = 0;
            while i < self.tokens.len() {
                if let Some(next_i) = self.next[i] {
                    if self.tokens[i] == affected_pair.0 && self.tokens[next_i] == affected_pair.1 {
                        count += 1;
                        positions_set.insert(i);
                    }
                    i = next_i;
                } else {
                    break;
                }
            }

            if count > 0 {
                self.pair_counts.insert(affected_pair, count);
                self.pair_positions.insert(affected_pair, positions_set);
                self.current_generation += 1;
                self.generation_map
                    .insert(affected_pair, self.current_generation);
                self.heap.push(PairCount {
                    pair: affected_pair,
                    count,
                    generation: self.current_generation,
                });
            } else {
                self.pair_counts.remove(&affected_pair);
                self.pair_positions.remove(&affected_pair);
            }
        }
    }
}

fn bpe(input: Vec<u32>, max_size: usize) -> Vec<String> {
    if input.len() < 2 {
        return Vec::new();
    }

    let mut state = BpeState::new(input);
    let mut vocab: Vec<String> = Vec::new();

    for merge_num in 0..max_size {
        print!("\rBuilt {} pairs", merge_num);
        std::io::stdout().flush().unwrap();
        if let Some(pair_to_merge) = state.find_best_pair() {
            let replacement_token = (256 + merge_num) as u32;

            state.merge_pair(pair_to_merge, replacement_token);

            vocab.push(format!("{},{}", pair_to_merge.0, pair_to_merge.1));
        } else {
            break;
        }
    }
    println!("\nFinished BPE encoding.");
    vocab
}
