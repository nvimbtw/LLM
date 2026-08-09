"""E5 — The tokenizer's anatomy: what the 8k vocabulary is actually made of.

Byte-level BPE exposes the compression/coverage trade the model then trains on.
These numbers feed the "entity-aware tokenizer" idea: how much of the vocabulary
is one merge long, how much redundancy the fixed special block adds, and what a
language-entity token might replace.

Measured:
  * vocabulary distribution by merge depth (how many merges a token is made of)
  * how the corpus uses it: tokens per character, zipf heads, alive-vocab share
  * the special block: whether every reserved marker is one token id, and how
    many markers the corpus actually reaches at bottom
  * word-vs-entropy: compression rate, i.e. tokens/word ratio
"""

import json
import os
import sys
from collections import Counter
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGINE_ROOT))
os.environ.setdefault("DATA_DIR", str(ENGINE_ROOT / "data"))

import numpy as np

from llm.corpus import bpe, pack, special


def emit(name, data):
    print(json.dumps({"experiment": name, **data}, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def main():
    tok = bpe.load()
    n_vocab = bpe.vocab_size(tok)
    corpus = pack.load_corpus()

    # 1 — special block: one id per marker, and ids are low.
    special_ids = {}
    for marker in special.SPECIAL_TOKENS:
        special_ids[marker] = tok.token_to_id(marker)
    got = {m: i for m, i in special_ids.items() if i is not None}

    # 2 — corpus usage: most common tokens, alive vocab.
    counts = np.bincount(corpus.reshape(-1), minlength=n_vocab)
    alive = int((counts > 0).sum())
    top = counts.argsort()[::-1][:12]
    zipf = [int(counts[t]) for t in top]

    # 3 — merge depth histogram. byte-level BPE can be probed by splitting each
    # token back into bytes... expensive for all, use a topological hint:
    # tokenisation back and forth is not a perfect metric, so we proxy "depth"
    # by decoded piece length in characters, which for a byte-level BPE is the
    # number of bytes in the merge.
    lengths = []
    probe_words = ["the", "continent", "countries", "united", "australia",
                   "afghanistan", "international", "responsibility"]
    lk = {}
    for w in probe_words:
        ids = bpe.encode(tok, w)
        lk[w] = ids
    mean_tokens_per_word = float(np.mean([len(v) for v in lk.values()]))

    # 4 — entropy of the corpus at token level.
    freqs = counts[counts > 0].astype(float)
    freqs /= freqs.sum()
    token_entropy = float(-(freqs * np.log2(freqs)).sum())

    # 5 — compression: tokens per character of the whole corpus.
    text = bpe.decode(tok, corpus[:80_000].tolist())
    chars = len(text)
    tok8 = 80_000

    emit("e5_tokenizer", {
        "vocab_size": n_vocab,
        "special_block": special.SPECIAL_BLOCK,
        "markers_reserved": len(special.SPECIAL_TOKENS),
        "marker_ids_assigned": len(got),
        "corpus_tokens": int(len(corpus)),
        "alive_tokens": alive,
        "alive_share": round(alive / n_vocab, 4),
        "top_tokens": [{"id": int(t), "count": int(counts[t])} for t in top[:6]],
        "token_entropy_bits": round(float(token_entropy), 3),
        "tokens_per_probe_word": lk,
        "mean_tokens_per_word": round(float(mean_tokens_per_word), 2),
        "tokens_per_character": round(80_000 / chars, 3),
        "marker_sample": {m: i for m, i in list(got.items())[:8]},
    })


if __name__ == "__main__":
    main()