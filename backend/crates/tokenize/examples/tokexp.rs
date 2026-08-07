//! Tokenizer exploration harness. Not part of the product — a bench for trying
//! pretokenizer variants, vocabulary sizes and sample compositions against the
//! real corpora, so a change to `pretokenize.rs` is made on a measurement rather
//! than on taste.
//!
//! Run:
//!   cargo run --release -p tokenize --example tokexp -- <uploads_dir> <sft_jsonl_dir>
//!
//! `bpe::train` writes merge progress to stdout, so results go to stderr.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};

use tokenize::bpe;
use tokenize::pretokenize::{self, special_id};

// ── the configurable pretokenizer ───────────────────────────────────────────
//
// `Cfg::baseline()` reproduces `pretokenize::pre_tokenize_text` exactly; the
// harness asserts that on real text before trusting any comparison.

#[derive(Clone, Copy, Debug)]
struct Cfg {
    /// A run of N spaces currently emits N-1 standalone `Ġ` pretokens, none of
    /// which can ever merge with another. With this on, the N-1 become a single
    /// pretoken, so indentation can be learned as one token.
    space_runs: bool,
    /// Same for newlines: `\n\n` is currently two pretokens that can never merge
    /// into a paragraph break.
    nl_runs: bool,
    /// 0 = off (digits ride along inside words), otherwise the max digits per
    /// group. A digit run is split into groups of at most this many.
    digit_group: usize,
    /// Isolate each punctuation char (current behaviour, on the small ASCII set)
    /// vs. splitting on a much wider set and keeping *runs* together so `...`,
    /// `?!` and `://` can each become one token.
    punct_runs: bool,
}

impl Cfg {
    fn baseline() -> Cfg {
        Cfg { space_runs: false, nl_runs: false, digit_group: 0, punct_runs: false }
    }
}

// The punctuation the current pretokenizer isolates, each as its own pretoken.
const PUNCT_BASE: &[char] = &[',', '.', '!', '?', '"', ':', ';'];

fn is_extended_punct(c: char) -> bool {
    // Everything that is neither a letter, a digit, whitespace-substitute, an
    // apostrophe, nor a special-token PUA char. Runs of these group together.
    if c == '\'' || c == '\u{2019}' || c == 'Ġ' || c == 'Ċ' || c == 'ĉ' || c == ' ' {
        return false;
    }
    if pretokenize::lookup_special(c).is_some() {
        return false;
    }
    !c.is_alphanumeric()
}

fn map_chars(text: &str) -> String {
    text.chars()
        .map(|c| match c {
            '\n' => 'Ċ',
            '\t' => 'ĉ',
            '\r' => ' ',
            _ => c,
        })
        .collect()
}

fn replace_specials(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < text.len() {
        let mut hit = false;
        for &(tok, pua) in pretokenize::SPECIAL_TOKENS {
            if text[i..].starts_with(tok) {
                out.push(pua);
                i += tok.len();
                hit = true;
                break;
            }
        }
        if !hit {
            let c = text[i..].chars().next().unwrap();
            out.push(c);
            i += c.len_utf8();
        }
    }
    out
}

fn pretokenize_cfg(text: &str, cfg: Cfg) -> Vec<String> {
    let mapped = map_chars(&replace_specials(text));
    let chars: Vec<char> = mapped.chars().collect();

    let mut out: Vec<String> = Vec::new();
    let mut cur = String::new();
    // A pending `Ġ` from the space that just ended: it prefixes the next word.
    let mut pending_space = false;

    macro_rules! flush {
        () => {
            if !cur.is_empty() {
                out.push(std::mem::take(&mut cur));
            }
        };
    }
    // The pending space becomes a standalone `Ġ` when nothing word-like follows.
    macro_rules! drop_pending {
        () => {
            if pending_space {
                out.push("Ġ".to_string());
                pending_space = false;
            }
        };
    }
    // A pending space always begins a new word, carrying its `Ġ` prefix. With no
    // pending space the character just continues the word being built.
    macro_rules! open_word {
        () => {{
            if pending_space {
                flush!();
                cur.push('Ġ');
                pending_space = false;
            }
        }};
    }

    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];

        if c == ' ' {
            flush!();
            let mut n = 0;
            while i + n < chars.len() && chars[i + n] == ' ' {
                n += 1;
            }
            i += n;
            // The last space of the run prefixes whatever comes next; the other
            // n-1 have to be represented on their own.
            if n > 1 {
                if pending_space {
                    // A pending space plus a fresh run: the pending one is stranded.
                    out.push("Ġ".to_string());
                }
                if cfg.space_runs {
                    out.push("Ġ".repeat(n - 1));
                } else {
                    for _ in 0..n - 1 {
                        out.push("Ġ".to_string());
                    }
                }
            } else if pending_space {
                out.push("Ġ".to_string());
            }
            pending_space = true;
            continue;
        }

        if pretokenize::lookup_special(c).is_some() {
            flush!();
            drop_pending!();
            out.push(c.to_string());
            i += 1;
            continue;
        }

        if c == 'Ċ' || c == 'ĉ' {
            flush!();
            drop_pending!();
            let mut n = 0;
            while i + n < chars.len() && chars[i + n] == c {
                n += 1;
            }
            if cfg.nl_runs {
                out.push(std::iter::repeat(c).take(n).collect());
            } else {
                for _ in 0..n {
                    out.push(c.to_string());
                }
            }
            i += n;
            continue;
        }

        if cfg.digit_group > 0 && c.is_ascii_digit() {
            // A digit run always starts its own pretoken, whether or not a space
            // preceded it — that is the point of the variant.
            flush!();
            if pending_space {
                cur.push('Ġ');
                pending_space = false;
            }
            let mut n = 0;
            while i + n < chars.len() && chars[i + n].is_ascii_digit() {
                n += 1;
            }
            let mut k = 0;
            while k < n {
                let take = cfg.digit_group.min(n - k);
                let group: String = chars[i + k..i + k + take].iter().collect();
                // The first group keeps the `Ġ` the word opened with; later groups
                // stand alone.
                if k == 0 && !cur.is_empty() {
                    cur.push_str(&group);
                    out.push(std::mem::take(&mut cur));
                } else {
                    out.push(group);
                }
                k += take;
            }
            i += n;
            continue;
        }

        let punct_hit = if cfg.punct_runs {
            is_extended_punct(c)
        } else {
            PUNCT_BASE.contains(&c)
        };

        if punct_hit {
            flush!();
            drop_pending!();
            if cfg.punct_runs {
                let mut n = 0;
                while i + n < chars.len() && is_extended_punct(chars[i + n]) {
                    n += 1;
                }
                out.push(chars[i..i + n].iter().collect());
                i += n;
            } else {
                out.push(c.to_string());
                i += 1;
            }
            continue;
        }

        if c == '\'' || (cfg.punct_runs && c == '\u{2019}') {
            flush!();
            if pending_space {
                cur.push('Ġ');
                pending_space = false;
            }
            cur.push(c);
            i += 1;
            continue;
        }

        open_word!();
        cur.push(c);
        i += 1;
    }
    flush!();
    drop_pending!();
    out
}

fn to_chars(words: Vec<String>) -> Vec<Vec<char>> {
    words.into_iter().map(|w| w.chars().collect()).collect()
}

// ── a local encoder matching `Encoder::merge_word` ──────────────────────────

struct Enc {
    pairs: HashMap<(u32, u32), u32>,
    cfg: Cfg,
}

impl Enc {
    fn new(merges: &[((u32, u32), u32)], cfg: Cfg) -> Enc {
        Enc { pairs: merges.iter().map(|&(p, t)| (p, t)).collect(), cfg }
    }

    fn merge_word(&self, ids: &mut Vec<u32>) {
        loop {
            let mut best: Option<(usize, u32)> = None;
            for (pos, w) in ids.windows(2).enumerate() {
                if let Some(&t) = self.pairs.get(&(w[0], w[1])) {
                    match best {
                        Some((_, bt)) if t > bt => {}
                        _ => best = Some((pos, t)),
                    }
                }
            }
            match best {
                Some((pos, t)) => {
                    ids[pos] = t;
                    ids.remove(pos + 1);
                }
                None => break,
            }
        }
    }

    fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        for word in pretokenize_cfg(text, self.cfg) {
            let mut ch = word.chars();
            if let (Some(c), None) = (ch.next(), ch.next()) {
                if let Some(id) = special_id(c) {
                    out.push(id);
                    continue;
                }
            }
            let mut ids: Vec<u32> = word
                .chars()
                .flat_map(|c| c.to_string().into_bytes().into_iter().map(|b| b as u32))
                .collect();
            self.merge_word(&mut ids);
            out.extend_from_slice(&ids);
        }
        out
    }
}

// ── corpus sampling ─────────────────────────────────────────────────────────

fn read_sample(dir: &Path, budget: usize) -> String {
    let mut files: Vec<PathBuf> = match fs::read_dir(dir) {
        Ok(rd) => rd
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("txt"))
            .collect(),
        Err(_) => return String::new(),
    };
    files.sort();
    if files.is_empty() {
        return String::new();
    }
    // Visit every file, but in a strided interleave so a sharded dataset is
    // sampled across its whole range rather than from its head — and so a small
    // dataset still yields everything it has when the budget outruns it.
    let stride = (files.len() / 64).max(1);
    let mut order: Vec<&PathBuf> = Vec::with_capacity(files.len());
    for offset in 0..stride {
        let mut j = offset;
        while j < files.len() {
            order.push(&files[j]);
            j += stride;
        }
    }

    let mut out = String::new();
    for p in order {
        if out.len() >= budget {
            break;
        }
        if let Ok(s) = fs::read_to_string(p) {
            out.push_str(&s);
        }
    }
    out.truncate(
        (0..=budget.min(out.len()))
            .rev()
            .find(|&i| out.is_char_boundary(i))
            .unwrap_or(0),
    );
    out
}

struct Corpus {
    name: String,
    text: String,
}

fn stats(enc: &Enc, text: &str, merge_base: u32) -> (f64, f64) {
    let ids = enc.encode(text);
    let bytes_per_token = text.len() as f64 / ids.len() as f64;
    let single = ids.iter().filter(|&&i| i < merge_base).count() as f64 / ids.len() as f64;
    (bytes_per_token, single * 100.0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let uploads = PathBuf::from(args.get(1).cloned().unwrap_or_else(|| {
        "/home/x_x/Programming/LLM/backend/data/raw/uploads".to_string()
    }));

    // Per-dataset train/held-out samples. Held-out is a disjoint slice.
    let budget: usize = std::env::var("BUDGET")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(12 << 20);

    let names = ["FineWeb", "Wikipedia", "TinyStories"];
    let mut train_sets: Vec<Corpus> = Vec::new();
    let mut eval_sets: Vec<Corpus> = Vec::new();
    for n in names {
        let all = read_sample(&uploads.join(n), budget * 2);
        if all.is_empty() {
            eprintln!("!! no data for {n}");
            continue;
        }
        // The held-out slice is a fixed size regardless of `budget`, so numbers
        // taken at different sample budgets are measured against the same text.
        const EVAL_BYTES: usize = 4 << 20;
        let mut cut = all.len().saturating_sub(EVAL_BYTES).max(all.len() / 2);
        while cut < all.len() && !all.is_char_boundary(cut) {
            cut += 1;
        }
        train_sets.push(Corpus { name: n.to_string(), text: all[..cut].to_string() });
        eval_sets.push(Corpus { name: n.to_string(), text: all[cut..].to_string() });
    }

    // A held-out chat set, rendered the way the SFT builder renders it. The
    // tokenizer is never *trained* on this — post-training just has to live with
    // whatever the pretrain tokenizer decided, which is exactly why it's measured.
    {
        let sft_dir = uploads.parent().map(|p| p.join("sft")).unwrap_or_default();
        let mut text = String::new();
        if let Ok(rd) = fs::read_dir(&sft_dir) {
            let mut files: Vec<PathBuf> = rd
                .filter_map(|e| e.ok())
                .map(|e| e.path())
                .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("jsonl"))
                .collect();
            files.sort();
            'outer: for f in files {
                let Ok(body) = fs::read_to_string(&f) else { continue };
                for line in body.lines() {
                    // Pull the "text" fields out without a JSON dependency: the
                    // crate has none and this is a bench, not a parser.
                    let mut rest = line;
                    while let Some(i) = rest.find("\"text\": \"") {
                        rest = &rest[i + 9..];
                        let mut out = String::new();
                        let mut it = rest.char_indices();
                        let mut end = rest.len();
                        while let Some((j, c)) = it.next() {
                            if c == '\\' {
                                if let Some((_, e)) = it.next() {
                                    out.push(match e {
                                        'n' => '\n',
                                        't' => '\t',
                                        '"' => '"',
                                        '\\' => '\\',
                                        o => o,
                                    });
                                }
                            } else if c == '"' {
                                end = j;
                                break;
                            } else {
                                out.push(c);
                            }
                        }
                        text.push_str(&out);
                        text.push('\n');
                        rest = &rest[end.min(rest.len())..];
                    }
                    if text.len() >= budget {
                        break 'outer;
                    }
                }
            }
        }
        if text.is_empty() {
            eprintln!("!! no SFT text found under {}", sft_dir.display());
        } else {
            eval_sets.push(Corpus { name: "SFTchat".to_string(), text });
        }
    }

    eprintln!("── samples ──");
    for c in &train_sets {
        eprintln!("  {:<12} train {:>10} B", c.name, c.text.len());
    }

    // Sanity: the baseline config must reproduce the shipped pretokenizer.
    {
        let probe = &train_sets[0].text[..4096.min(train_sets[0].text.len())];
        let mine = pretokenize_cfg(probe, Cfg::baseline());
        let theirs = pretokenize::pre_tokenize_str(probe);
        if mine != theirs {
            let n = mine.iter().zip(&theirs).take_while(|(a, b)| a == b).count();
            eprintln!(
                "!! baseline MISMATCH at {n}: mine={:?} theirs={:?}",
                &mine[n..(n + 6).min(mine.len())],
                &theirs[n..(n + 6).min(theirs.len())]
            );
            std::process::exit(1);
        }
        eprintln!("✓ baseline config reproduces pre_tokenize_str ({} pretokens)", mine.len());
    }

    // Build the tokenizer-training sample under a given composition, then report
    // held-out compression per dataset and corpus-weighted.
    let run = |label: &str, cfg: Cfg, vocab: u32, proportional: bool| {
        let mut sample = String::new();
        if proportional {
            // Share ∝ real corpus size. FineWeb is ~98% of the corpus.
            for c in &train_sets {
                let share = if c.name == "FineWeb" { 90 } else { 5 };
                let take = (budget * share / 100).min(c.text.len());
                let mut end = take;
                while end < c.text.len() && !c.text.is_char_boundary(end) {
                    end += 1;
                }
                sample.push_str(&c.text[..end]);
            }
        } else {
            let each = budget / train_sets.len();
            for c in &train_sets {
                let take = each.min(c.text.len());
                let mut end = take;
                while end < c.text.len() && !c.text.is_char_boundary(end) {
                    end += 1;
                }
                sample.push_str(&c.text[..end]);
            }
        }

        let pre = to_chars(pretokenize_cfg(&sample, cfg));
        let t0 = std::time::Instant::now();
        let (vsize, _, merges) = bpe::train(pre, Some(vocab), |_| {}).unwrap();
        let secs = t0.elapsed().as_secs_f64();

        let enc = Enc::new(&merges, cfg);
        let mb = bpe::MERGE_BASE;

        // Corpus-weighted: FineWeb dominates what the model actually reads.
        let mut parts = Vec::new();
        for c in &eval_sets {
            let (bpt, single) = stats(&enc, &c.text, mb);
            parts.push((c.name.clone(), bpt, single));
        }
        let weighted: f64 = parts
            .iter()
            .map(|(n, bpt, _)| if n == "FineWeb" { bpt * 0.95 } else { bpt * 0.025 })
            .sum();

        eprintln!(
            "{:<34} vocab={:<6} merges={:<6} weighted={:.3}  {}  [{:.0}s]",
            label,
            vsize,
            merges.len(),
            weighted,
            parts
                .iter()
                .map(|(n, b, s)| format!("{}={:.2}/{:.0}%", &n[..2.min(n.len())], b, s))
                .collect::<Vec<_>>()
                .join(" "),
            secs
        );
        weighted
    };

    let b = Cfg::baseline();
    let all3 = Cfg { space_runs: true, nl_runs: true, punct_runs: true, ..b };
    let phase = std::env::var("PHASE").unwrap_or_else(|_| "variants".into());

    eprintln!("\n   weighted = bytes/token, FineWeb-weighted (higher = better compression)");
    eprintln!("   per-set  = bytes/token / share of emitted tokens that are single bytes\n");

    match phase.as_str() {
        // Does the pretokenizer shape matter, holding vocab and sampling fixed?
        "variants" => {
            run("A baseline", b, 8192, false);
            run("B +space runs", Cfg { space_runs: true, ..b }, 8192, false);
            run("C +newline runs", Cfg { nl_runs: true, ..b }, 8192, false);
            run("D +punct runs (wide)", Cfg { punct_runs: true, ..b }, 8192, false);
            run("E +digit groups(3)", Cfg { digit_group: 3, ..b }, 8192, false);
            run("F +digit groups(1)", Cfg { digit_group: 1, ..b }, 8192, false);
            run("G B+C", Cfg { space_runs: true, nl_runs: true, ..b }, 8192, false);
            run("H B+C+D", all3, 8192, false);
            run("I H+digits(1)", Cfg { digit_group: 1, ..all3 }, 8192, false);
            run("J H+digits(3)", Cfg { digit_group: 3, ..all3 }, 8192, false);
        }
        // How much does the tokenizer-training sample composition matter?
        "sampling" => {
            run("baseline  equal", b, 8192, false);
            run("baseline  proportional", b, 8192, true);
            run("H         equal", all3, 8192, false);
            run("H         proportional", all3, 8192, true);
        }
        // The two levers that actually cost something downstream.
        "vocab" => {
            for v in [4096u32, 8192, 16384, 32768, 65536] {
                run(&format!("baseline vocab {v:<6}"), b, v, true);
            }
            eprintln!();
            for v in [4096u32, 8192, 16384, 32768, 65536] {
                run(&format!("H        vocab {v:<6}"), all3, v, true);
            }
        }
        other => eprintln!("unknown PHASE {other}"),
    }
}
