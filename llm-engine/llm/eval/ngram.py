"""A byte-level n-gram model, used as the benchmark's *floor*.

Bits-per-byte is a real number with no natural zero — "1.9 bits/byte" means
nothing on its own. What makes it a verdict is a comparison: a 5-gram lookup
table fit on the same text is a lower bound on "has this model learned anything a
counting table doesn't already know". A transformer scoring worse than this has
not learned language, whatever its training loss did.

Bytes, not tokens, on purpose. The baseline must not depend on the model's
tokenizer — otherwise changing `max_vocab` would move the floor as well as the
measurement, and the two numbers would stop being comparable.

Smoothing is Witten-Bell interpolation: at each order, blend the observed
conditional with the next-shorter context's estimate, weighting the observation
by how often the context was seen against how many distinct continuations it
took. Parameter-free — nothing to tune, so nothing to tune *against* the model
under test — and properly normalized, which bits-per-byte requires to be a real
bound rather than a score.
"""

import math
from collections import defaultdict

MAX_ORDER = 5

# A byte has 256 possible values; the order-(-1) fallback is uniform over them,
# which is what guarantees every byte a non-zero probability.
UNIFORM = 1.0 / 256.0


class ByteNgram:
    def __init__(self, order: int = MAX_ORDER):
        self.order = min(order, MAX_ORDER)
        # (order, context, next) -> count
        self.grams: dict[tuple, int] = defaultdict(int)
        # (order, context) -> [observations, distinct continuations]
        self.contexts: dict[tuple, list[int]] = defaultdict(lambda: [0, 0])

    @classmethod
    def fit(cls, text: bytes, order: int = MAX_ORDER) -> "ByteNgram":
        """Count every n-gram up to `order`, all orders in one pass.

        Shorter windows are what the interpolation falls back through, so they all
        have to be counted, not just the longest.
        """
        model = cls(order)
        for i, next_byte in enumerate(text):
            for n in range(model.order + 1):
                if i < n:
                    continue
                context = text[i - n : i]
                stats = model.contexts[(n, context)]
                key = (n, context, next_byte)
                if model.grams[key] == 0:
                    stats[1] += 1
                model.grams[key] += 1
                stats[0] += 1
        return model

    def probability(self, history: bytes, next_byte: int) -> float:
        """Probability of `next_byte`, interpolated from the longest context that
        fits down to uniform."""
        usable = min(len(history), self.order)
        # Start at the shortest (uniform) and fold longer contexts in on top, so
        # each order's estimate is already blended when the next one uses it.
        p = UNIFORM
        for n in range(usable + 1):
            context = history[len(history) - n :] if n else b""
            stats = self.contexts.get((n, context))
            if stats is None:
                # Context never seen, and no longer context can have been either.
                break
            total, distinct = stats
            count = self.grams.get((n, context, next_byte), 0)

            # Witten-Bell: trust the counts in proportion to how much evidence the
            # context has against how varied it is.
            lam = total / (total + distinct)
            observed = count / total if total else 0.0
            p = lam * observed + (1.0 - lam) * p

        # Every order blends with a strictly positive fallback, so this cannot
        # reach zero — clamped anyway so a log can never produce -inf.
        return max(p, 5e-324)

    def bits_per_byte(self, text: bytes) -> float:
        """Cross-entropy of `text`, in bits per byte — directly comparable to the
        transformer's own bits-per-byte on the same text."""
        if not text:
            return math.nan
        bits = 0.0
        for i in range(len(text)):
            lo = max(0, i - self.order)
            bits -= math.log2(self.probability(text[lo:i], text[i]))
        return bits / len(text)
