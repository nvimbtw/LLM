# A corpus is a function of its vocabulary, not just its text

`corpus.bin` is a flat array of token ids. The ids are only meaningful under the
vocabulary that produced them, so a packed corpus is a function of **two**
inputs: which text was selected, and which tokenizer encoded it. Sharing one path
across either produces data that loads perfectly and means something else.

The pretrain build ignored both. There was one `datasets/corpus.bin`, one
`artifacts/tokenizer/tokenizer.json`, and the decision to skip phase 1 was:

```python
def is_built() -> bool:
    return storage.corpus_path().is_file() and storage.tokenizer_file().is_file()
```

Nothing was compared against the request. A run asking for different corpora,
different caps, or a different `max_vocab` skipped the build and trained on
whatever the previous run had left behind, while `meta.json` recorded the config
it *asked* for. `rebuild_dataset=true` was the only way to be sure — a flag
carrying correctness, which is a flag everyone eventually forgets.

## This is the second time

`llm/corpus/sft.py` hit the same wall first and solved it in four parts, and the
fix here is that mechanism applied one directory over:

1. **Resolve before keying.** An empty selection means "every corpus on disk",
   which is a *different* corpus once another is fetched. It is expanded at build
   time, never stored as a wildcard.
2. **Key the directory** from the resolved request — order preserved, never
   sorted, because concatenation order is a real training difference under a
   seeded shuffle.
3. **Write the selection into a sidecar** beside the data, so a build says what
   it is.
4. **Verify against that sidecar**, field by field, rather than trusting the
   directory name. A key collision, a hand-edited directory or a changed key
   scheme all fail closed into a rebuild.

The new input is the vocabulary, and it is keyed by a **fingerprint** — a digest
of the saved `tokenizer.json` — rather than by the tokenizer's id. Rebuilding
under one id renumbers every token, so an id-keyed corpus would be wrong in
exactly the silent way the singleton was.

## What it cost, and what it bought

Disk. The singleton was self-limiting; `datasets/pretrain/<key>/` holds one
corpus per (vocabulary × selection), and a 390M-token corpus is ~745 MiB. So
deleting a vocabulary deletes the corpora keyed on it, and the tokenizers page
shows what each build occupies. There is no eviction policy yet.

What it bought, beyond correctness: a vocabulary became a **thing you can name**.
A run either names a build or gets one derived from its own inputs by `uuid5`, so
two identical runs still share a vocabulary and a corpus — the reuse that made
skipping tokenization worth having, now with the request actually consulted.

## The bug underneath the bug

A reused corpus reported `"dataset": "reused"` — a *string*. `sft.py::_base_special_ids`
reads `result["dataset"]["special_ids_seen"]`, got `None` from a string, and
concluded "unknown", so `untrained_marker_ids` reset nothing. Every model trained
on a reused corpus entered SFT with its ~28 chat-scaffold marker embeddings at
random init, silently — the exact failure `llm/train/warm_start.py` exists to
prevent, and its only symptom was `warm_started: 0` on a base that had plainly
never seen a chat template.

The keyed sidecar records `special_ids_seen`, so reuse can answer the question
instead of dodging it. `result["dataset"]` is now a dict in both branches, which
is what makes the answer reachable at all.
