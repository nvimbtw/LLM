"""E7 — Entity prelexing: how many token slots would a tiny "entity vocab"
actually save on the packed corpus?

The BPE sees "australia" as 4 pieces every time it appears. Before proposing any
model-side graph machinery, measure the simplest possible lever: reserve K
single-token slots for the most recurring Capitalised phrases and re-encode the
corpus with exactly those phrases as byte-level single ids. Reports, per budget:
  * phrases covered, mentions captured, live savings
  * corpus token-count reduction (overall),
so the "entity budget" idea from report E5/E4 gets a concrete number.

Run (from repo root):
    llm-engine/.venv/bin/python llm-engine/explore/entity/e7_entity_vocab.py \\
        > llm-engine/explore/entity/e7_entity_vocab.json
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

LLM_ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(LLM_ENGINE_ROOT))
os.environ.setdefault("DATA_DIR", str(LLM_ENGINE_ROOT / "data"))

import numpy as np

from llm import storage
from llm.corpus import bpe, pack, vocab
from llm.eval.surface import words_in

PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
DATES = set(
    "january february march april may june july august september october "
    "november december monday tuesday wednesday thursday friday saturday sunday "
    "summer winter spring autumn today yesterday tomorrow".split())
STOPWORDS = set(
    "the of and to in is on for with by are was en be at that as from or have an "
    "also were has been which that they do been more than it its these so one two "
    "he she we you that what when where how which those their not into over "
    "file of all this at after".split())


def _name_like(found, freq):
    out = []
    for e in found:
        words = [w.lower() for w in e.split()]
        if not words or words[0] in STOPWORDS or words[0] in DATES:
            continue
        if len(words) == 1 and freq[words[0]] > 200:
            continue
        if all(w in DATES for w in words):
            continue
        out.append(e)
    return out


def main():
    tok = vocab.for_corpus(pack.latest_key())
    corpus = pack.load_corpus(pack.latest_key())
    _, docs, _ = pack.load_segments(pack.latest_key())

    freq = Counter()
    ents = Counter()
    for d in docs:
        raw = bpe.decode(tok, corpus[d.start:d.start + d.length].tolist())
        found = _name_like(PROPER.findall(raw), Counter(words_in(raw)))
        for e in found:
            ents[e.lower().replace(" ", "_")] += 1
        for w in words_in(raw):
            freq[w] += 1

    total_tokens = corpus.shape[0]
    total_words = sum(freq.values())
    total_mentions = sum(ents.values())

    cost = {}
    for key, n in ents.items():
        pieces = len(bpe.encode(tok, " " + key.replace("_", " ")))
        cost[key] = n * pieces

    # A single-piece phrase saves nothing by becoming one token; it should not
    # consume a budget slot. Filter to phrases that actually compress.
    compressible = {k for k in ents if len(bpe.encode(tok, " " + k.replace("_", " "))) >= 2}
    cost = {k: v for k, v in cost.items() if k in compressible}

    ranked = sorted(cost, key=cost.__getitem__, reverse=True)
    total = sum(cost.values())

    per_entity_pieces = {}
    for key in ranked[:12]:
        per_entity_pieces[key] = [ents[key], len(bpe.encode(tok, " " + key.replace("_", " ")))]

    budgets = [0, 64, 128, 256, 512, 1024, 2048]
    results = {}
    for budget in budgets:
        covered = set() if budget == 0 else set(ranked[:budget])
        saved = sum(cost[k] for k in covered)
        results[budget] = {
            "covered_entities": len(covered),
            "saved_pieces": saved,
            "corpus_reduction_pct": round(100.0 * saved / total_tokens, 3),
            "per_mention_avg_pieces_saved": round(saved / max(1, sum(ents[k] for k in covered)), 3),
        }

    emit("e7_entity_vocab", {
        "corpus": {"docs": len(docs), "tokens": total_tokens,
                   "words": total_words},
        "entity_stats": {
            "distinct": len(ents),
            "total_mentions": int(total_mentions),
            "mentions_per_1000_tokens": round(1000.0 * total_mentions / total_tokens, 2),
            "piece_total_all": total,
            "piece_share_of_corpus_pct": round(100.0 * total / total_tokens, 2),
        },
        "top_entities": [(k, {"mentions": int(ents[k]), "pieces": int(per_entity_pieces[k][1]),
                              "saved_if_single": int(max(0, per_entity_pieces[k][0] * (per_entity_pieces[k][1] - 1)))},
                          ) for k in ranked[:10]],
        "by_budget": results,
    })


def emit(name, data):
    print(json.dumps({"experiment": name, **data},
                     default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


if __name__ == "__main__":
    main()