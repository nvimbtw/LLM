// A byte-level n-gram language model, used as the benchmark's *floor*.
//
// Bits-per-byte is a real number with no natural zero — "1.9 bits/byte" means
// nothing on its own. What makes it a verdict is a comparison: a 5-gram lookup
// table fit on the same text is a lower bound on "has this model learned anything
// a counting table doesn't already know". A transformer that scores worse than
// this has not learned language, whatever its training loss did.
//
// Bytes, not tokens, on purpose. The baseline must not depend on the model's
// tokenizer — otherwise changing `max_vocab` would move the floor as well as the
// measurement, and the two numbers would stop being comparable.
//
// Smoothing is Witten-Bell interpolation: at each order, blend the observed
// conditional with the next-shorter context's estimate, weighting the observation
// by how often the context was seen relative to how many distinct continuations it
// took. It is parameter-free (nothing to tune, so nothing to tune *against* the
// model under test) and — unlike stupid backoff — properly normalized, which
// bits-per-byte requires to be a real bound rather than a score.

use std::collections::HashMap;

// Bytes of left context. Capped at 5 because a context plus its target byte and
// its order are packed into a single u64 key (5×8 + 8 + 8 = 56 bits), which makes
// lookups collision-free without hashing byte slices.
pub const MAX_ORDER: usize = 5;

// A byte has 256 possible values; the order-(-1) fallback is uniform over them,
// which is what guarantees every byte gets a non-zero probability.
const UNIFORM: f64 = 1.0 / 256.0;

pub struct ByteNgram {
    order: usize,
    // (order, context, next byte) → how often that continuation was seen.
    grams: HashMap<u64, u32>,
    // (order, context) → (total observations, distinct continuations). The second
    // is Witten-Bell's "how surprising is a new continuation here" term.
    contexts: HashMap<u64, (u32, u32)>,
}

// Packs an order and up to `MAX_ORDER` context bytes into one key. Layout:
// bits 0..40 the context bytes, 40..48 the target byte, 48..56 the order. The
// target is left zero for a context key; `gram_key` fills it in.
fn context_key(order: usize, context: &[u8]) -> u64 {
    let mut packed = 0u64;
    for (i, &b) in context.iter().enumerate() {
        packed |= (b as u64) << (i * 8);
    }
    packed | ((order as u64) << 48)
}

fn gram_key(order: usize, context: &[u8], next: u8) -> u64 {
    context_key(order, context) | ((next as u64) << 40)
}

impl ByteNgram {
    // Count every n-gram up to `order` in `text`. One pass, all orders at once.
    pub fn fit(text: &[u8], order: usize) -> ByteNgram {
        let order = order.min(MAX_ORDER);
        let mut model = ByteNgram {
            order,
            grams: HashMap::new(),
            contexts: HashMap::new(),
        };

        for i in 0..text.len() {
            let next = text[i];
            // n = 0 is the unconditional byte distribution; n = order uses the
            // full window. Shorter windows are what the interpolation falls back
            // through, so they all have to be counted, not just the longest.
            for n in 0..=order {
                if i < n {
                    continue;
                }
                let context = &text[i - n..i];
                let ck = context_key(n, context);
                let gk = gram_key(n, context, next);

                let count = model.grams.entry(gk).or_insert(0);
                let first_time = *count == 0;
                *count += 1;

                let stats = model.contexts.entry(ck).or_insert((0, 0));
                stats.0 += 1;
                if first_time {
                    stats.1 += 1;
                }
            }
        }

        model
    }

    // Probability of `next` given the preceding bytes, interpolated from the
    // longest context that fits down to uniform.
    pub fn probability(&self, history: &[u8], next: u8) -> f64 {
        let usable = history.len().min(self.order);
        // Start at the shortest (uniform) and fold longer contexts in on top, so
        // each order's estimate is already blended when the next one uses it.
        let mut p = UNIFORM;
        for n in 0..=usable {
            let context = &history[history.len() - n..];
            let (total, distinct) = match self.contexts.get(&context_key(n, context)) {
                Some(&stats) => stats,
                // Context never seen: nothing to add at this order, and no longer
                // context can have been seen either, so the blend is done.
                None => break,
            };
            let count = self
                .grams
                .get(&gram_key(n, context, next))
                .copied()
                .unwrap_or(0);

            // Witten-Bell: trust the counts in proportion to how much evidence the
            // context has (`total`) against how varied it is (`distinct`).
            let denom = (total + distinct) as f64;
            let lambda = total as f64 / denom;
            let observed = if total > 0 { count as f64 / total as f64 } else { 0.0 };
            p = lambda * observed + (1.0 - lambda) * p;
        }
        // Every order is blended with a strictly positive fallback, so this cannot
        // reach zero — but clamp anyway so a log can never produce -inf.
        p.max(f64::MIN_POSITIVE)
    }

    // Cross-entropy of `text` under this model, in bits per byte — directly
    // comparable to the transformer's own bits-per-byte on the same text.
    pub fn bits_per_byte(&self, text: &[u8]) -> f64 {
        if text.is_empty() {
            return f64::NAN;
        }
        let mut bits = 0.0;
        for i in 0..text.len() {
            let lo = i.saturating_sub(self.order);
            bits -= self.probability(&text[lo..i], text[i]).log2();
        }
        bits / text.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_separate_orders_and_targets() {
        assert_ne!(context_key(2, b"ab"), context_key(3, b"ab"));
        assert_ne!(gram_key(2, b"ab", b'x'), gram_key(2, b"ab", b'y'));
        // A context key must not collide with a gram key that has target 0.
        assert_eq!(context_key(2, b"ab"), gram_key(2, b"ab", 0));
    }

    #[test]
    fn probabilities_over_the_byte_alphabet_sum_to_one() {
        let model = ByteNgram::fit(b"the cat sat on the mat, the cat sat still", 3);
        let total: f64 = (0..=255u8).map(|b| model.probability(b"the ", b)).sum();
        assert!((total - 1.0).abs() < 1e-9, "distribution summed to {total}");
    }

    #[test]
    fn an_unseen_byte_still_gets_positive_probability() {
        let model = ByteNgram::fit(b"aaaaaaaaaa", 3);
        assert!(model.probability(b"aaa", b'z') > 0.0);
    }

    #[test]
    fn repetitive_text_costs_fewer_bits_than_random_bytes() {
        let repetitive = b"abababababababababababababababab";
        let model = ByteNgram::fit(repetitive, 4);
        let learned = model.bits_per_byte(repetitive);
        // Uniform over 256 symbols is exactly 8 bits/byte; a model that has
        // learned the alternation must beat that comfortably.
        assert!(learned < 4.0, "expected well under 4 bits/byte, got {learned}");
    }

    #[test]
    fn an_empty_model_falls_back_to_the_uniform_byte_distribution() {
        let model = ByteNgram::fit(b"", 5);
        let bits = -model.probability(b"", b'q').log2();
        assert!((bits - 8.0).abs() < 1e-9, "expected 8 bits, got {bits}");
    }

    #[test]
    fn order_is_capped_at_the_packable_maximum() {
        let model = ByteNgram::fit(b"hello world", 12);
        assert_eq!(model.order, MAX_ORDER);
    }
}
