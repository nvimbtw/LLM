"""E4 — A knowledge-graph view of the corpus, and whether the trained model
spans it.

A transformer is a view over token streams; a knowledge graph is a view over
entities and relations. Before proposing any graph machinery inside the model,
measure how much graph like-structure the corpus already contains, and whether
the model's own embeddings already agree with the graph:

  1. a weighted word co-occurrence graph over the corpus (per-document windows,
     no cross-document edges)
  2. statistics: degree, hubs, clustering, coverage, modularity communities
  3. whether graph neighbours are closer than non-neighbours in the model's
     embedding space (BPE-averaged word vectors), and how similarity decays
     with hop distance
  4. whether graph communities separate as embedding clusters
  5. the BPE "cost" of entity-like (Capitalised) phrases, and their recurrence
     across documents
"""

import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

ENGINE_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ENGINE_ROOT))
os.environ.setdefault("DATA_DIR", str(ENGINE_ROOT / "data"))

import numpy as np
import torch
import torch.nn.functional as F
import networkx as nx

from llm import storage
from llm.corpus import bpe, pack, vocab
from llm.eval.surface import words_in
from llm.model.checkpoint import load as ckpt_load


def emit(name, data):
    print(json.dumps({"experiment": name, **data},
                      default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))


def trained_model(name):
    return ckpt_load(storage.weights_path(name), "cpu")


WINDOW = 5        # max word span counted as "co-occurring"
MIN_FREQ = 30     # min occurrences for a word to become a node
TOPK = 80         # top-degree nodes in the embedding-correlation probe
PROPER = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*")
# A capitalized phrase is probably a name only if it does not open a sentence
# with a common word. First token must not be a stopword, and a single-word
# "name" must not be one of the 300 most frequent frames (which are always
# capitalized at sentence start).
_WORD_FREQ = None


DATES = set((
    "january february march april may june july august september october "
    "november december monday tuesday wednesday thursday friday saturday sunday "
    "summer winter spring autumn today yesterday tomorrow"
).split())


def _name_like(tok, found, freq):
    """Drop sentence-leading, stopword, and date/phrase-channel structures."""
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
STOPWORDS = set((
    "the of and to in is on for with by are was en be at that as from or have an "
    "also were has been which their they do been more than it its there so one two "
    "he she we you them what when where how his her their those these not into over "
    "file at after"
).split())


def _doc_words(tok, corpus, start, length):
    return words_in(bpe.decode(tok, corpus[start:start + length].tolist()))


def build_graph(per_doc, freq):
    keep = {w for w, n in freq.items() if n >= MIN_FREQ}
    pairs = Counter()
    for ws in per_doc:
        for i, a in enumerate(ws):
            if a not in keep:
                continue
            for b in ws[i + 1:i + WINDOW + 1]:
                if b not in keep or b == a:
                    continue
                key = (a, b) if a < b else (b, a)
                pairs[key] += 1
    g = nx.Graph()
    g.add_nodes_from(keep)
    for (a, b), n in pairs.items():
        if n >= 2:
            g.add_edge(a, b, weight=n)
    return g


def _word_embeddings(model, tok, words):
    """Word vectors: mean of the BPE piece embeddings, cosine-normalized."""
    table = model.embedding.weight
    out = {}
    for w in words:
        ids = bpe.encode(tok, " " + w)
        if not ids:
            continue
        out[w] = F.normalize(table[ids].mean(dim=0), dim=0)
    return out


def embedding_correlation(g, top, emb):
    """Connected vs non-connected cosine, and cosine decaying with hop count."""
    n = len(top)
    if n < 2 or not emb:
        return {"available": False}
    M = torch.stack([emb[w] for w in top]).detach()
    sims = M @ M.T
    adj = nx.to_numpy_array(g, nodelist=top)

    connected, disconnected = [], []
    for i in range(n):
        for j in range(i + 1, n):
            s = float(sims[i, j])
            (connected if adj[i, j] > 0 else disconnected).append(s)

    idx = {w: k for k, w in enumerate(top)}
    sim_by_hop = {}
    for a in top:
        for b, d in nx.single_source_shortest_path_length(g, a, cutoff=3).items():
            if b in idx and d >= 1 and a != b:
                sim_by_hop.setdefault(d, []).append(float(sims[idx[a], idx[b]]))
    mean_by_hop = {d: round(float(np.mean(v)), 4) for d, v in sorted(sim_by_hop.items())}

    return {
        "available": True,
        "nodes": n,
        "pairs": len(connected) + len(disconnected),
        "connected_mean_cos": round(float(np.mean(connected)), 4) if connected else None,
        "disconnected_mean_cos": round(float(np.mean(disconnected)), 4) if disconnected else None,
        "gap": round(float(np.mean(connected) - np.mean(disconnected)), 4)
        if connected and disconnected else 0.0,
        "mean_sim_by_hop": mean_by_hop,
    }


def pagerank_np(g, alpha=0.85, iters=200):
    """Dense power-iteration PageRank — avoids scipy, fine for ~1k nodes."""
    nodes = list(g)
    idx = {w: i for i, w in enumerate(nodes)}
    n = len(nodes)
    if n == 0:
        return {}
    A = np.zeros((n, n))
    for a, b in g.edges():
        A[idx[b], idx[a]] += 1.0                                 # a -> b
    out = A.sum(axis=0)
    out[out == 0] = 1.0
    A = A / out
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        r = (1 - alpha) / n + alpha * (A @ r
)
    return dict(zip(nodes, r.tolist()))


def main():
    model, _ = trained_model("diagfix")
    tok = vocab.for_corpus(pack.latest_key())
    corpus = pack.load_corpus(pack.latest_key())
    _, docs, _ = pack.load_segments(pack.latest_key())

    per_doc = []
    raw_texts = []
    for d in docs:
        raw = bpe.decode(tok, corpus[d.start:d.start + d.length].tolist())
        raw_texts.append(raw)
        per_doc.append(words_in(raw))
    freq = Counter(w for ws in per_doc for w in ws)
    g = build_graph(per_doc, freq)

    deg = dict(g.degree())
    by_deg = sorted(deg, key=deg.get, reverse=True)
    pagerank = pagerank_np(g)
    by_pr = sorted(pagerank, key=pagerank.get, reverse=True)
    comps = list(nx.connected_components(g))
    comms = list(nx.algorithms.community.greedy_modularity_communities(g, weight="weight"))

    top = [w for w in by_deg if w not in STOPWORDS][:TOPK] or by_deg[:TOPK]
    emb = _word_embeddings(model, tok, by_deg[:300])
    corr = embedding_correlation(g, top, emb)

    # Graph communities vs embedding clusters.
    comm_embed = []
    for c in comms:
        rows = [emb[w] for w in c if w in emb]
        if len(rows) >= 4:
            comm_embed.append(rows)
    intra, inter = [], []
    for i in range(len(comm_embed)):
        R = torch.stack(comm_embed[i]).detach()
        intra += (R @ R.T).flatten().tolist()
        for j in range(i + 1, len(comm_embed)):
            S = torch.stack(comm_embed[j]).detach()
            inter += (R @ S.T).flatten().tolist()

    # Entity-like phrase probe (case preserved).
    entity_sets = []
    all_entities = Counter()
    traces = []
    for raw in raw_texts:
        found = PROPER.findall(raw)
        found = _name_like(tok, found, freq)
        ents = {e.lower().replace(" ", "_") for e in found}
        for e in found:
            key = e.lower().replace(" ", "_")
            all_entities[key] += 1
            traces.append(len(bpe.encode(tok, " " + e)))
        entity_sets.append(ents)
    multi_doc = 0
    for i in range(len(entity_sets)):
        for j in range(i + 1, len(entity_sets)):
            multi_doc += len(entity_sets[i] & entity_sets[j])

    emit("e4_knowledge_graph", {
        "corpus": {"documents": len(docs), "words_total": sum(len(w) for w in per_doc),
                   "unique_words": len(freq)},
        "graph": {
            "nodes": int(g.number_of_nodes()),
            "edges": int(g.number_of_edges()),
            "mean_degree": round(mean_degree(g), 3),
            "clustering": round(nx.average_clustering(g), 4) if g.number_of_nodes() else None,
            "components": len(comps),
            "hubs": [(w, int(deg[w])) for w in by_deg[:10]],
            "pagerank_top": [(w, round(pagerank[w], 4)) for w in by_pr[:6]],
        },
        "embedding_correlation": corr,
        "communities": {
            "n": len(comms),
            "sizes_top8": sorted((len(c) for c in comms), reverse=True)[:8],
            "intra_mean_cos": round(float(np.mean(intra)), 4) if intra else None,
            "inter_mean_cos": round(float(np.mean(inter)), 4) if inter else None,
        },
        "entities": {
            "distinct_phrases": len(all_entities),
            "total_mentions": int(sum(all_entities.values())),
            "pieces_mean": round(float(np.mean(traces)), 2) if traces else None,
            "pieces_max": int(max(traces)) if traces else None,
            "repeated_across_docs": int(multi_doc),
            "sample": [(e, all_entities[e]) for e, _ in all_entities.most_common(6)],
        },
    })


def mean_degree(g):
    return sum(dict(g.degree()).values()) / max(1, g.number_of_nodes())


def load_model(name):
    """The trained (pretrained) model. Kept here so `main` reads cleanly."""
    model, arch = trained_model(name)
    model.eval()
    return model, arch


if __name__ == "__main__":
    main()