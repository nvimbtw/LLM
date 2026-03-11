use rayon::prelude::*;
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
    let input = read_text_file("data/input.txt");
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
        let len = input.len();
        let mut state = BpeState {
            tokens: input,
            next: (1..=len).map(|i| if i < len { Some(i) } else { None }).collect(),
            prev: (0..len).map(|i| if i > 0 { Some(i - 1) } else { None }).collect(),
            pair_counts: HashMap::new(),
            pair_positions: HashMap::new(),
            heap: BinaryHeap::new(),
            generation_map: HashMap::new(),
            current_generation: 0,
        };

        if len < 2 { return state; }

        // Parallel initial counting using Rayon
        let chunk_size = (len / rayon::current_num_threads()).max(1024);
        let partial_counts: Vec<HashMap<(u32, u32), Vec<usize>>> = (0..len - 1)
            .into_par_iter()
            .step_by(chunk_size)
            .map(|start| {
                let end = (start + chunk_size).min(len - 1);
                let mut local_map: HashMap<(u32, u32), Vec<usize>> = HashMap::new();
                for i in start..end {
                    let pair = (state.tokens[i], state.tokens[i + 1]);
                    local_map.entry(pair).or_default().push(i);
                }
                local_map
            })
            .collect();

        for map in partial_counts {
            for (pair, positions) in map {
                state.pair_counts.entry(pair).and_modify(|c| *c += positions.len()).or_insert(positions.len());
                state.pair_positions.entry(pair).or_default().extend(positions);
            }
        }

        for (&pair, &count) in &state.pair_counts {
            state.heap.push(PairCount { pair, count, generation: 0 });
            state.generation_map.insert(pair, 0);
        }

        state
    }

    fn find_best_pair(&mut self) -> Option<(u32, u32)> {
        while let Some(pc) = self.heap.pop() {
            if let Some(&current_gen) = self.generation_map.get(&pc.pair) {
                if pc.generation == current_gen {
                    let actual_count = *self.pair_counts.get(&pc.pair).unwrap_or(&0);
                    if actual_count > 0 && actual_count == pc.count {
                        return Some(pc.pair);
                    }
                }
            }
        }
        None
    }

    fn update_pair_count(&mut self, pair: (u32, u32), _delta: i32, pos: usize, add: bool) {
        if add {
            *self.pair_counts.entry(pair).or_insert(0) += 1;
            self.pair_positions.entry(pair).or_default().insert(pos);
        } else {
            if let Some(count) = self.pair_counts.get_mut(&pair) {
                if *count > 0 { *count -= 1; }
            }
            if let Some(positions) = self.pair_positions.get_mut(&pair) {
                positions.remove(&pos);
            }
        }
        
        // Push to heap lazily
        self.current_generation += 1;
        let new_count = *self.pair_counts.get(&pair).unwrap_or(&0);
        self.generation_map.insert(pair, self.current_generation);
        self.heap.push(PairCount {
            pair,
            count: new_count,
            generation: self.current_generation,
        });
    }

    fn merge_pair(&mut self, pair: (u32, u32), replacement: u32) {
        let positions: Vec<usize> = self.pair_positions.get(&pair)
            .map(|s| s.iter().copied().collect())
            .unwrap_or_default();
        
        // We sort positions to handle the "overlapping" case correctly (e.g. AAA -> BA)
        let mut sorted_positions = positions;
        sorted_positions.sort_unstable();

        for &pos in &sorted_positions {
            // Check if this position is still valid and contains the left part of the pair
            if self.tokens[pos] != pair.0 { continue; }
            let next_pos = match self.next[pos] {
                Some(np) if self.tokens[np] == pair.1 => np,
                _ => continue,
            };

            // 1. Remove old pairs affected by the merge
            // Left neighbor
            if let Some(p) = self.prev[pos] {
                let left_pair = (self.tokens[p], self.tokens[pos]);
                self.update_pair_count(left_pair, -1, p, false);
            }
            // The pair itself
            // self.update_pair_count(pair, -1, pos, false); // Already handled by clearing the whole pair later
            
            // Right neighbor
            if let Some(nn) = self.next[next_pos] {
                let right_pair = (self.tokens[next_pos], self.tokens[nn]);
                self.update_pair_count(right_pair, -1, next_pos, false);
            }

            // 2. Perform the merge
            self.tokens[pos] = replacement;
            let nn_opt = self.next[next_pos];
            self.next[pos] = nn_opt;
            if let Some(nn) = nn_opt {
                self.prev[nn] = Some(pos);
            }
            // next_pos is now effectively deleted

            // 3. Add new pairs created by the merge
            if let Some(p) = self.prev[pos] {
                let left_pair = (self.tokens[p], self.tokens[pos]);
                self.update_pair_count(left_pair, 1, p, true);
            }
            if let Some(nn) = self.next[pos] {
                let right_pair = (self.tokens[pos], self.tokens[nn]);
                self.update_pair_count(right_pair, 1, pos, true);
            }
        }

        // Final cleanup for the merged pair
        self.pair_counts.remove(&pair);
        self.pair_positions.remove(&pair);
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
