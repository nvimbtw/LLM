use std::collections::hash_map::Entry;
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::fs::File;
use std::io;
use std::io::{BufWriter, Read, Seek, Write};

use crate::pretokenize::{lookup_special, special_char_from_id, special_id, SPECIAL_BLOCK};

/// First id available to BPE merges: raw bytes fill `[0,256)` and the reserved
/// special-token block fills `[256, 256+SPECIAL_BLOCK)`, so learned merges start
/// here. `train` counts up from this, and `vocab_size` folds the same offset in.
///
/// This is keyed to the block size rather than the marker count on purpose — see
/// `pretokenize::SPECIAL_BLOCK` for why a moving `MERGE_BASE` invalidates every
/// tokenizer and model already on disk.
pub const MERGE_BASE: u32 = 256 + SPECIAL_BLOCK as u32;

/// Total vocabulary size for a tokenizer with `num_merges` learned merges:
/// bytes + the reserved special-token block + merges. The single source of truth
/// for vocab sizing — every caller (model init, job metadata) routes through here
/// so the offset can't drift between the tokenizer and the model.
pub fn vocab_size(num_merges: usize) -> i64 {
    MERGE_BASE as i64 + num_merges as i64
}

/// Learns a merge table, returning `(vocab_size, corpus, merges)`.
///
/// **Training and encoding apply merges by the same rule, and must keep doing
/// so.** Each learned merge is applied to every *word* independently, in rank
/// order, to fixed point — which is exactly what `Encoder::merge_word` does when
/// it repeatedly takes the lowest-id applicable merge. That equivalence is what
/// makes every learned merge reachable at encode time, and it is not free:
///
/// This used to apply each merge across a flat corpus while tracking pair
/// positions in a side index, and the index went stale the moment a merge
/// shortened a word — every later position in that word shifted left by one and
/// was never corrected, so occurrences were silently skipped. The merges learned
/// after that point were fitted to a segmentation `Encoder` cannot reproduce, and
/// **38% of the table (3,004 of 7,872 merges) could never be emitted at all**.
/// The casualties were the most common words in English: `" the"` was learned as
/// `" th" + "e"` at rank 6, but the encoder reaches `[" t", "he"]` and stops, so
/// `" the"` was in the vocabulary and unreachable. That cost ~35% of the
/// tokenizer's compression — 51% of emitted tokens were single characters, and
/// bytes-per-token sat at 2.13 where ~3.2 is normal at this vocabulary size.
///
/// `every_learned_merge_is_reachable_by_the_encoder` is the guard. It is a total
/// property, not a sample: if it fails, the two algorithms have drifted again.
pub fn train(
    pre_tokenized: Vec<Vec<char>>,
    max_vocab_size: Option<u32>,
    on_log: impl Fn(&str),
) -> Result<(i64, Vec<Vec<u32>>, Vec<((u32, u32), u32)>), io::Error> {
    let (mut words, freq, of_occurrence) = dedupe_words(convert_to_bytes(pre_tokenized));
    let (mut pair_counts, mut pair_words) = index_pairs(&words, &freq);
    let mut heap = create_heap(pair_counts.clone());

    // Clamped to what `corpus.bin` can represent — an id past this would be
    // rejected at write time, after the whole corpus had been encoded.
    let max = max_vocab_size.unwrap_or(32768).max(MERGE_BASE).min(MAX_VOCAB);
    // Merge ids start above the reserved byte + special-token block, so a learned
    // merge can never collide with a raw byte or an atomic special-token id.
    let mut token: u32 = MERGE_BASE;
    let mut merges: Vec<((u32, u32), u32)> = Vec::new();

    let log_every = ((max - 256) / 50).max(1);

    while heap.len() > 0 && max > token {
        let (count, pair) = heap.pop().unwrap();

        // Lazy deletion: the heap keeps outdated entries for pairs whose count has
        // moved since they were pushed, so a popped entry counts only if it still
        // matches the live tally.
        if *pair_counts.get(&pair).unwrap_or(&0) != count || count == 0 {
            continue;
        }

        merges.push((pair, token));
        print!("\rMerge {}: {:?} -> {}          ", token - 256, pair, token);
        std::io::stdout().flush().unwrap();
        if (token - 256) % log_every == 0 {
            on_log(&format!("BPE merge {}/{}", token - 256, max - 256));
        }

        // Only the words that actually hold the pair can change, so the rest of the
        // vocabulary is never walked. Collected first because the loop rewrites
        // `pair_words` as it goes.
        let affected: Vec<usize> = match pair_words.get(&pair) {
            Some(set) => set.iter().copied().collect(),
            None => Vec::new(),
        };

        for w in affected {
            let merged = apply_pair(&words[w], pair, token);
            if merged == words[w] {
                continue;
            }
            // Retire the word's old pairs and file its new ones. Recomputing the
            // whole word rather than patching positions around the merge point is
            // what retires the stale-index bug: a word's tally is always derived
            // from the word as it now stands, never from a remembered offset.
            retire_pairs(
                &words[w],
                freq[w],
                &mut pair_counts,
                &mut pair_words,
                &mut heap,
                w,
            );
            words[w] = merged;
            admit_pairs(
                &words[w],
                freq[w],
                &mut pair_counts,
                &mut pair_words,
                &mut heap,
                w,
            );
        }

        // `apply_pair` runs to fixed point, so the merged pair is gone everywhere.
        pair_counts.remove(&pair);
        pair_words.remove(&pair);

        token += 1;
    }

    println!();
    println!("Training complete! {} merges performed.", merges.len());

    let vocab = vocab_size(merges.len());
    let corpus: Vec<Vec<u32>> = of_occurrence.iter().map(|&w| words[w].clone()).collect();

    Ok((vocab, corpus, merges))
}

/// Applies one merge to a word, to fixed point, exactly as `Encoder::merge_word`
/// would if this were the only merge available to it.
///
/// It takes the **rightmost** occurrence each pass, which is the position
/// `merge_word`'s tie-break settles on when several positions hold the same
/// (lowest) token id — `Some((_, best)) if token > best => {}` keeps the incumbent
/// only when the new id is strictly greater, so an equal id replaces it and the
/// last match wins. On `[a,a,a]` that yields `[a,X]`, where a left-to-right sweep
/// would yield `[X,a]`. The two disagree on any odd run of a repeated symbol, so
/// this is load-bearing rather than stylistic.
fn apply_pair(word: &[u32], pair: (u32, u32), token: u32) -> Vec<u32> {
    let mut out = word.to_vec();
    loop {
        let mut at = None;
        for pos in 0..out.len().saturating_sub(1) {
            if out[pos] == pair.0 && out[pos + 1] == pair.1 {
                at = Some(pos);
            }
        }
        match at {
            Some(pos) => {
                out[pos] = token;
                out.remove(pos + 1);
            }
            None => return out,
        }
    }
}

/// Collapses the pretoken stream to its distinct words, returning
/// `(words, freq, of_occurrence)` — the distinct words, how often each occurs, and
/// which word every original occurrence maps to (so the corpus can be rebuilt in
/// input order).
///
/// BPE only ever needs a word once, weighted by its count: the pair tallies are
/// sums over occurrences either way. Real text repeats itself heavily, so this is
/// also what keeps the trainer's footprint proportional to the *vocabulary* of the
/// sample rather than to its length.
fn dedupe_words(occurrences: Vec<Vec<u32>>) -> (Vec<Vec<u32>>, Vec<usize>, Vec<usize>) {
    let mut seen: HashMap<Vec<u32>, usize> = HashMap::new();
    let mut words: Vec<Vec<u32>> = Vec::new();
    let mut freq: Vec<usize> = Vec::new();
    let mut of_occurrence: Vec<usize> = Vec::with_capacity(occurrences.len());

    for word in occurrences {
        match seen.entry(word) {
            Entry::Occupied(e) => {
                let w = *e.get();
                freq[w] += 1;
                of_occurrence.push(w);
            }
            Entry::Vacant(e) => {
                let w = words.len();
                words.push(e.key().clone());
                freq.push(1);
                of_occurrence.push(w);
                e.insert(w);
            }
        }
    }

    (words, freq, of_occurrence)
}

/// The live pair tally: how often each adjacent pair occurs across the whole
/// sample, and which distinct words contain it.
fn index_pairs(
    words: &[Vec<u32>],
    freq: &[usize],
) -> (
    HashMap<(u32, u32), usize>,
    HashMap<(u32, u32), HashSet<usize>>,
) {
    let mut pair_counts: HashMap<(u32, u32), usize> = HashMap::new();
    let mut pair_words: HashMap<(u32, u32), HashSet<usize>> = HashMap::new();

    for (w, word) in words.iter().enumerate() {
        for pair in word.windows(2) {
            let key = (pair[0], pair[1]);
            *pair_counts.entry(key).or_insert(0) += freq[w];
            pair_words.entry(key).or_default().insert(w);
        }
    }

    (pair_counts, pair_words)
}

// Subtracts word `w`'s contribution to every pair it currently holds.
//
// A decremented pair is pushed back at its *new* count, which is not optional: the
// heap only ever holds the counts it was pushed with, and a pair that shrank would
// otherwise be represented solely by stale-high entries that fail the tally check
// on pop. It would then never be reconsidered, and a merge that is still the most
// frequent one available would be skipped entirely.
fn retire_pairs(
    word: &[u32],
    weight: usize,
    pair_counts: &mut HashMap<(u32, u32), usize>,
    pair_words: &mut HashMap<(u32, u32), HashSet<usize>>,
    heap: &mut BinaryHeap<(usize, (u32, u32))>,
    w: usize,
) {
    for pair in word.windows(2) {
        let key = (pair[0], pair[1]);
        if let Some(count) = pair_counts.get_mut(&key) {
            *count = count.saturating_sub(weight);
            if *count == 0 {
                pair_counts.remove(&key);
            } else {
                heap.push((*count, key));
            }
        }
        if let Some(set) = pair_words.get_mut(&key) {
            set.remove(&w);
            if set.is_empty() {
                pair_words.remove(&key);
            }
        }
    }
}

// Adds word `w`'s contribution back, pushing each touched pair's new count so the
// heap always holds a current entry for anything that moved.
fn admit_pairs(
    word: &[u32],
    weight: usize,
    pair_counts: &mut HashMap<(u32, u32), usize>,
    pair_words: &mut HashMap<(u32, u32), HashSet<usize>>,
    heap: &mut BinaryHeap<(usize, (u32, u32))>,
    w: usize,
) {
    for pair in word.windows(2) {
        let key = (pair[0], pair[1]);
        let count = pair_counts.entry(key).or_insert(0);
        *count += weight;
        heap.push((*count, key));
        pair_words.entry(key).or_default().insert(w);
    }
}

// Consumes the input by value and iterates it by value on purpose: taking it by
// reference kept the whole `Vec<Vec<char>>` alive alongside the `Vec<Vec<u32>>`
// being built, doubling the peak at the point in training where it is already
// highest. Draining lets each word's chars be freed as it is converted.
fn convert_to_bytes(pre_tokenized: Vec<Vec<char>>) -> Vec<Vec<u32>> {
    pre_tokenized
        .into_iter()
        .map(|word| {
            word.into_iter()
                .flat_map(|c| match special_id(c) {
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

fn create_heap(pair_counts: HashMap<(u32, u32), usize>) -> BinaryHeap<(usize, (u32, u32))> {
    let mut heap: BinaryHeap<(usize, (u32, u32))> = BinaryHeap::new();

    for (pair, count) in pair_counts {
        heap.push((count, pair));
    }

    heap
}

pub fn save_merges(merges: &[((u32, u32), u32)], path: &str) -> Result<(), io::Error> {
    let mut file = BufWriter::new(File::create(path)?);
    for ((a, b), token) in merges {
        file.write_all(&a.to_le_bytes())?;
        file.write_all(&b.to_le_bytes())?;
        file.write_all(&token.to_le_bytes())?;
    }
    // BufWriter's Drop discards a failed flush, which would silently truncate the
    // merge list and make every model trained against it gibberish.
    file.flush()
}

// ── the corpus format ───────────────────────────────────────────────────────

/// Marks a file as this project's token stream, so a corpus written under an
/// older layout fails loudly instead of being read as garbage ids.
const CORPUS_MAGIC: [u8; 4] = *b"LLMC";
const CORPUS_VERSION: u16 = 1;

/// Bytes per stored token id.
///
/// Two, not four: the vocabulary is `MERGE_BASE + merges`, and `max_vocab` is
/// capped at `MAX_VOCAB` below, so every id fits in a `u16`. Halving the element
/// width halves the corpus on disk *and* the resident `Vec` during training,
/// which on a small machine is the difference between a run that fits and one
/// that is killed before the first step.
const CORPUS_WIDTH: u16 = 2;

/// Header bytes preceding the ids: magic, version, element width.
const CORPUS_HEADER_BYTES: usize = 8;

/// Largest vocabulary the corpus format can represent. Callers that accept a
/// vocabulary size from a user must reject anything above this **before**
/// training a tokenizer, or the ids will not round-trip through `corpus.bin`.
pub const MAX_VOCAB: u32 = u16::MAX as u32 + 1;

const CORPUS_BUF_BYTES: usize = 1 << 20;

/// Streams token ids into `corpus.bin`.
///
/// Exists because `save_corpus` needs the whole corpus in memory and issued one
/// unbuffered 4-byte write per token — at hundreds of millions of tokens that is
/// both a memory wall and hundreds of millions of syscalls.
pub struct CorpusWriter {
    file: BufWriter<File>,
    tokens: usize,
}

impl CorpusWriter {
    /// Truncates `path` and writes the header. One writer covers one build;
    /// there is deliberately no append mode, which would imply resume semantics
    /// nothing has specified.
    pub fn create(path: &str) -> Result<CorpusWriter, io::Error> {
        let mut file = BufWriter::with_capacity(CORPUS_BUF_BYTES, File::create(path)?);
        file.write_all(&CORPUS_MAGIC)?;
        file.write_all(&CORPUS_VERSION.to_le_bytes())?;
        file.write_all(&CORPUS_WIDTH.to_le_bytes())?;
        Ok(CorpusWriter { file, tokens: 0 })
    }

    /// Ids above `u16::MAX` are rejected rather than truncated: a silently
    /// wrapped id is a valid-looking token pointing at the wrong embedding row,
    /// which would surface only as a model that never learns.
    pub fn write(&mut self, tokens: &[u32]) -> Result<(), io::Error> {
        for &token in tokens {
            let narrowed = u16::try_from(token).map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("token id {token} exceeds the corpus format's {MAX_VOCAB}-id limit"),
                )
            })?;
            self.file.write_all(&narrowed.to_le_bytes())?;
        }
        self.tokens += tokens.len();
        Ok(())
    }

    /// Tokens written so far — the caller's next document `start` offset.
    pub fn len(&self) -> usize {
        self.tokens
    }

    pub fn is_empty(&self) -> bool {
        self.tokens == 0
    }

    /// Flushes and fsyncs, returning the total written. Explicit rather than left
    /// to `Drop`, which cannot report a failed flush — and a corpus silently
    /// missing its last buffer is worse than one that failed to build.
    pub fn finish(mut self) -> Result<usize, io::Error> {
        self.file.flush()?;
        self.file.get_ref().sync_all()?;
        Ok(self.tokens)
    }
}

// Serializes a flat token-id stream in the `corpus.bin` format. Kept for callers
// that already hold the whole corpus (the SFT/KTO datasets, which are small by
// construction); the pretrain path streams through `CorpusWriter` instead.
pub fn save_corpus(corpus: &[u32], path: &str) -> Result<(), io::Error> {
    let mut writer = CorpusWriter::create(path)?;
    writer.write(corpus)?;
    writer.finish()?;
    Ok(())
}

/// Reads `len` ids starting at token offset `start`, seeking rather than scanning.
///
/// For callers that want a handful of spans out of a corpus far too large to hold
/// — sampling replay documents out of a pretrain corpus, say. A span running past
/// the end of the file is truncated to what is there, so a stale `segments.json`
/// yields short spans rather than an error.
pub fn load_corpus_span(path: &str, start: usize, len: usize) -> Result<Vec<u16>, io::Error> {
    let mut file = File::open(path)?;
    let mut header = [0u8; CORPUS_HEADER_BYTES];
    file.read_exact(&mut header)?;
    if header[0..4] != CORPUS_MAGIC || u16::from_le_bytes([header[6], header[7]]) != CORPUS_WIDTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: not a readable corpus (rebuild it)"),
        ));
    }

    let offset = CORPUS_HEADER_BYTES as u64 + (start as u64) * CORPUS_WIDTH as u64;
    let available = file.metadata()?.len().saturating_sub(offset);
    let take = len.min((available / CORPUS_WIDTH as u64) as usize);
    if take == 0 {
        return Ok(Vec::new());
    }

    file.seek(io::SeekFrom::Start(offset))?;
    let mut bytes = vec![0u8; take * CORPUS_WIDTH as usize];
    file.read_exact(&mut bytes)?;
    Ok(bytes
        .chunks_exact(CORPUS_WIDTH as usize)
        .map(|p| u16::from_le_bytes([p[0], p[1]]))
        .collect())
}

/// Reads a `corpus.bin` back.
///
/// Returns `u16` rather than widening to `u32`, and preallocates from the file
/// length rather than collecting, so the peak is one copy of the corpus instead
/// of the `Vec<u8>` + `Vec<u32>` pair the old read-then-collect produced.
/// Callers widen per chunk at the point of use.
pub fn load_corpus(path: &str) -> Result<Vec<u16>, io::Error> {
    let mut file = File::open(path)?;
    let mut header = [0u8; CORPUS_HEADER_BYTES];
    file.read_exact(&mut header).map_err(|_| {
        io::Error::new(io::ErrorKind::InvalidData, format!("{path}: too short to be a corpus"))
    })?;
    if header[0..4] != CORPUS_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: not a corpus file (or written before the u16 format — rebuild it)"),
        ));
    }
    let version = u16::from_le_bytes([header[4], header[5]]);
    let width = u16::from_le_bytes([header[6], header[7]]);
    if version != CORPUS_VERSION || width != CORPUS_WIDTH {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("{path}: corpus version {version}/width {width} is not readable here"),
        ));
    }

    let bytes = file.metadata()?.len().saturating_sub(CORPUS_HEADER_BYTES as u64);
    let mut corpus: Vec<u16> = Vec::with_capacity((bytes / CORPUS_WIDTH as u64) as usize);
    // Read in fixed blocks rather than slurping the file: at corpus scale the
    // intermediate byte buffer would be as large as the result.
    let mut block = vec![0u8; CORPUS_BUF_BYTES];
    let mut carry: Option<u8> = None;
    loop {
        let read = file.read(&mut block)?;
        if read == 0 {
            break;
        }
        let mut chunk = &block[..read];
        // A block boundary can land mid-id, so the odd byte is held over.
        if let Some(low) = carry.take() {
            if let Some((&high, rest)) = chunk.split_first() {
                corpus.push(u16::from_le_bytes([low, high]));
                chunk = rest;
            } else {
                carry = Some(low);
                continue;
            }
        }
        let pairs = chunk.chunks_exact(2);
        if let [last] = pairs.remainder() {
            carry = Some(*last);
        }
        corpus.extend(pairs.map(|p| u16::from_le_bytes([p[0], p[1]])));
    }
    Ok(corpus)
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

/// A merge table indexed once for repeated encoding.
///
/// The free `encode` rebuilds this map from the merge slice on every call, which
/// costs nothing for a chat prompt and dominates when encoding a corpus chunk by
/// chunk — hence a type that holds the index across calls. Encoding is otherwise
/// identical: `encode` is `Encoder::new(merges).encode(text)`.
pub struct Encoder {
    pair_to_token: HashMap<(u32, u32), u32>,
}

impl Encoder {
    pub fn new(merges: &[((u32, u32), u32)]) -> Encoder {
        let mut pair_to_token = HashMap::with_capacity(merges.len());
        for &((a, b), token) in merges {
            pair_to_token.insert((a, b), token);
        }
        Encoder { pair_to_token }
    }

    /// Appends `text`'s ids to `out` **without clearing it**, so a streaming
    /// caller reuses one buffer across chunks instead of allocating per call.
    pub fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        let words = crate::pretokenize::pre_tokenize_str(text);
        let mut ids: Vec<u32> = Vec::new();
        for word in words {
            // The pretokenizer isolates each special token into its own word, so a
            // one-char word that is a special PUA char maps straight to its atomic
            // reserved id — matching exactly how `convert_to_bytes` fed it to BPE.
            let mut chars = word.chars();
            if let (Some(c), None) = (chars.next(), chars.next()) {
                if let Some(id) = special_id(c) {
                    out.push(id);
                    continue;
                }
            }

            ids.clear();
            ids.extend(word.chars().flat_map(|c| {
                let s = c.to_string();
                s.into_bytes().into_iter().map(|b| b as u32)
            }));
            self.merge_word(&mut ids);
            out.extend_from_slice(&ids);
        }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    // Repeatedly applies the lowest-id applicable merge until none is left. The
    // lowest id is the earliest-learned merge, which is the order `train` applied
    // them in, so the result matches what the trainer would have produced.
    fn merge_word(&self, ids: &mut Vec<u32>) {
        loop {
            let mut best_pair: Option<(usize, u32)> = None;
            for (pos, w) in ids.windows(2).enumerate() {
                let key = (w[0], w[1]);
                if let Some(&token) = self.pair_to_token.get(&key) {
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
    }
}

pub fn encode(text: &str, merges: &[((u32, u32), u32)]) -> Vec<u32> {
    Encoder::new(merges).encode(text)
}

/// A merge table indexed once for repeated decoding.
///
/// The counterpart of [`Encoder`], and it exists for a sharper reason. The free
/// `decode` rebuilds `token_to_pair` from the whole merge slice on every call —
/// ~7,900 hash inserts at vocab 8,192 — and the streaming chat loop called it once
/// per generated token to re-decode the reply so far. Measured at **292 us per
/// call, essentially all of it the rebuild** (decoding one token costs the same as
/// decoding a thousand), against a ~1.9 ms token: 15% of every token spent
/// reconstructing a table that never changes.
///
/// Decoding is otherwise identical: `decode` is `Decoder::new(merges).decode(ids)`.
pub struct Decoder {
    token_to_pair: HashMap<u32, (u32, u32)>,
}

impl Decoder {
    pub fn new(merges: &[((u32, u32), u32)]) -> Decoder {
        let mut token_to_pair = HashMap::with_capacity(merges.len());
        for &((a, b), token) in merges {
            token_to_pair.insert(token, (a, b));
        }
        Decoder { token_to_pair }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let token_to_pair = &self.token_to_pair;
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
}

pub fn decode(ids: &[u32], merges: &[((u32, u32), u32)]) -> String {
    Decoder::new(merges).decode(ids)
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
    // dedupe_words / index_pairs
    // -----------------------------------------------------------------------
    #[test]
    fn test_dedupe_words_counts_and_maps_back() {
        let (words, freq, of_occurrence) = dedupe_words(vec![
            vec![97u32, 98],
            vec![99u32],
            vec![97u32, 98],
        ]);

        assert_eq!(words.len(), 2, "two distinct words");
        assert_eq!(freq[of_occurrence[0]], 2, "the repeated word is weighted 2");
        assert_eq!(freq[of_occurrence[1]], 1);
        // of_occurrence must rebuild the original stream in order.
        let rebuilt: Vec<Vec<u32>> = of_occurrence.iter().map(|&w| words[w].clone()).collect();
        assert_eq!(rebuilt, vec![vec![97, 98], vec![99], vec![97, 98]]);
    }

    #[test]
    fn test_index_pairs_weights_by_word_frequency() {
        // One distinct word occurring 5 times contributes 5 to each of its pairs.
        let (counts, words_with) = index_pairs(&[vec![97u32, 98, 99]], &[5]);

        assert_eq!(counts[&(97, 98)], 5);
        assert_eq!(counts[&(98, 99)], 5);
        assert!(words_with[&(97, 98)].contains(&0));
    }

    #[test]
    fn test_index_pairs_single_symbol_word_has_no_pairs() {
        let (counts, _) = index_pairs(&[vec![97u32]], &[3]);
        assert!(counts.is_empty());
    }

    // -----------------------------------------------------------------------
    // apply_pair
    // -----------------------------------------------------------------------
    #[test]
    fn test_apply_pair_runs_to_fixed_point() {
        // Both occurrences go, unlike the old positional walk which skipped the
        // second once the first had shifted the word.
        assert_eq!(apply_pair(&[97, 97, 97, 97], (97, 97), 256), vec![256, 256]);
    }

    #[test]
    fn test_apply_pair_takes_the_rightmost_of_an_odd_run() {
        // The tie-break `Encoder::merge_word` settles on. A left-to-right sweep
        // would give [256, 97] here, and encoding would stop matching training.
        assert_eq!(apply_pair(&[97, 97, 97], (97, 97), 256), vec![97, 256]);
    }

    #[test]
    fn test_apply_pair_leaves_a_word_without_the_pair_alone() {
        assert_eq!(apply_pair(&[97, 98, 99], (98, 97), 256), vec![97, 98, 99]);
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
        // Header, then 97 -> [0x61,0], 256 -> [0,1]: flat LE-u16 ids.
        let corpus = vec![97u32, 256u32];
        let path = std::env::temp_dir().join("test_bpe_corpus.bin");
        let _ = std::fs::remove_file(&path);

        save_corpus(&corpus, path.to_str().unwrap()).unwrap();
        let bytes = std::fs::read(&path).unwrap();

        assert_eq!(&bytes[0..4], b"LLMC");
        assert_eq!(u16::from_le_bytes([bytes[4], bytes[5]]), CORPUS_VERSION);
        assert_eq!(u16::from_le_bytes([bytes[6], bytes[7]]), CORPUS_WIDTH);
        assert_eq!(&bytes[CORPUS_HEADER_BYTES..], &[0x61, 0, 0, 1]);
        assert_eq!(load_corpus(path.to_str().unwrap()).unwrap(), vec![97u16, 256u16]);

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // the u16 corpus format
    // -----------------------------------------------------------------------

    // The write path must survive a block boundary landing mid-id: `load_corpus`
    // reads in fixed blocks, and an id straddling two of them is the one case the
    // carry logic exists for. A corpus longer than the read buffer forces it.
    #[test]
    fn test_corpus_roundtrip_across_read_blocks() {
        let corpus: Vec<u32> = (0..CORPUS_BUF_BYTES as u32).map(|i| i % 30000).collect();
        let path = std::env::temp_dir().join(format!("test_corpus_blocks_{}.bin", std::process::id()));
        let _ = std::fs::remove_file(&path);

        let mut writer = CorpusWriter::create(path.to_str().unwrap()).unwrap();
        // Written in odd-sized batches so the buffer boundaries don't align with
        // the batches either.
        for batch in corpus.chunks(777) {
            writer.write(batch).unwrap();
        }
        assert_eq!(writer.len(), corpus.len());
        assert_eq!(writer.finish().unwrap(), corpus.len());

        let read = load_corpus(path.to_str().unwrap()).unwrap();
        let expected: Vec<u16> = corpus.iter().map(|&t| t as u16).collect();
        assert_eq!(read, expected);

        let _ = std::fs::remove_file(&path);
    }

    // A corpus written before the header existed must fail loudly. Read as u16 it
    // would otherwise decode to plausible-looking ids and train a model on noise.
    #[test]
    fn test_load_corpus_rejects_a_headerless_file() {
        let path = std::env::temp_dir().join(format!("test_corpus_old_{}.bin", std::process::id()));
        std::fs::write(&path, [0x61u8, 0, 0, 0, 0, 1, 0, 0]).unwrap();

        let err = load_corpus(path.to_str().unwrap()).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("rebuild"), "unhelpful message: {err}");

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_corpus_writer_rejects_an_unrepresentable_id() {
        let path = std::env::temp_dir().join(format!("test_corpus_wide_{}.bin", std::process::id()));
        let mut writer = CorpusWriter::create(path.to_str().unwrap()).unwrap();

        let err = writer.write(&[MAX_VOCAB]).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let _ = std::fs::remove_file(&path);
    }

    // -----------------------------------------------------------------------
    // Encoder
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoder_matches_the_free_encode() {
        let merges = vec![((97u32, 97u32), MERGE_BASE), ((MERGE_BASE, 97u32), MERGE_BASE + 1)];
        let encoder = Encoder::new(&merges);
        for text in ["aaaa bb", "<|assistant|>aaa", "", "a\nb", "don't"] {
            assert_eq!(encoder.encode(text), encode(text, &merges), "differed on {text:?}");
        }
    }

    #[test]
    fn test_encode_into_appends_rather_than_clearing() {
        let merges: Vec<((u32, u32), u32)> = Vec::new();
        let encoder = Encoder::new(&merges);
        let mut out = vec![999u32];
        encoder.encode_into("a", &mut out);
        encoder.encode_into("b", &mut out);
        assert_eq!(out[0], 999);
        assert_eq!(out, [vec![999u32], encoder.encode("a"), encoder.encode("b")].concat());
    }

    // The property the streaming corpus build rests on: a newline is a hard
    // pretoken boundary (`split_punctuation` splits on 'Ċ'), so cutting the input
    // there and encoding the halves separately must equal encoding it whole. If
    // this ever fails, chunked ingestion is silently producing a different corpus.
    #[test]
    fn test_chunking_at_newlines_is_seam_free() {
        let merges = vec![
            ((97u32, 97u32), MERGE_BASE),
            ((104u32, 101u32), MERGE_BASE + 1),
            ((MERGE_BASE + 1, 108u32), MERGE_BASE + 2),
        ];
        let encoder = Encoder::new(&merges);
        let text = "hello there\naaa world.\n<|par|>a quote — “héllo”\nend\ndon't stop\n";
        let whole = encoder.encode(text);

        let mut cuts = 0;
        for (i, c) in text.char_indices() {
            if c != '\n' {
                continue;
            }
            let split = i + 1;
            let joined = [encoder.encode(&text[..split]), encoder.encode(&text[split..])].concat();
            assert_eq!(joined, whole, "seam at byte {split} changed the tokens");
            cuts += 1;
        }
        assert!(cuts >= 4, "fixture must exercise several newlines, got {cuts}");
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
/// **The invariant the whole trainer exists to hold.** Every merge it learns must
/// be something `Encoder` can actually produce from that token's own text.
///
/// This is a total check over the table, not a spot sample, because the failure it
/// guards is silent and proportional: the old trainer left 38% of a real 7,872-merge
/// table unreachable — including `" the"`, `" in"`, `" from"` and `" they"` — and
/// nothing downstream noticed. The vocabulary looked full, the model trained, and
/// the only symptom was compression quietly sitting a third below where it belonged.
#[test]
fn every_learned_merge_is_reachable_by_the_encoder() {
    // Enough ordinary English that the table fills with the word-initial merges
    // (" th" → " the" → " they") whose chains the old trainer built unreachably.
    let text = "the theatre was there and they said that the others were \
                from the same theatre, in the end, because the theory of it \
                is that these things are their own reward and they know it. \
                there is no other way to think of the thing they are in. \
                consider how many people would rather remember something \
                important than understand another question completely, \
                since knowledge without understanding becomes information \
                nobody actually wanted in the first place after everything.";
    let repeated = text.repeat(20);
    let pre = crate::pretokenize::pre_tokenize_text(&repeated);
    let (_vocab, _corpus, merges) = train(pre, Some(700), |_| {}).unwrap();

    assert!(merges.len() > 150, "expected a full table, got {}", merges.len());

    let enc = Encoder::new(&merges);
    let dec = Decoder::new(&merges);

    // The property is "encodes to a *single* token", not "encodes to this exact
    // id". Those differ only for aliases — two ids that decode to the same text,
    // of which the encoder can naturally reach just one. On a real 8192 table that
    // is 4 merges, all of them a lone space, costing 4 embedding rows and no
    // compression. Fragmentation is the failure that matters: `" the"` coming back
    // as `[" t", "he"]` is a token the model can never be shown.
    let fragmented: Vec<(String, Vec<String>)> = merges
        .iter()
        .map(|&(_, id)| dec.decode(&[id]))
        .filter(|text| !text.is_empty())
        .filter_map(|text| {
            let got = enc.encode(&text);
            (got.len() != 1)
                .then(|| (text, got.iter().map(|&i| dec.decode(&[i])).collect()))
        })
        .collect();

    assert!(
        fragmented.is_empty(),
        "{} of {} merges fragment when re-encoded: {:?}",
        fragmented.len(),
        merges.len(),
        &fragmented[..fragmented.len().min(20)]
    );
}

/// The corpus `train` returns must be exactly what `Encoder` produces for the same
/// text. It was ~33% shorter before, holding ids inference could never emit.
#[test]
fn the_returned_corpus_matches_what_the_encoder_produces() {
    let text = "the theatre was there and they said that the others were from \
                the same theatre in the end because the theory of it is that";
    let repeated = text.repeat(20);
    let pre = crate::pretokenize::pre_tokenize_text(&repeated);
    let (_vocab, corpus, merges) = train(pre, Some(400), |_| {}).unwrap();

    let enc = Encoder::new(&merges);
    let flat: Vec<u32> = corpus.into_iter().flatten().collect();
    assert_eq!(flat, enc.encode(&repeated), "trainer and encoder disagree");
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

