# Vocabulary layout

Ids are laid out as `[0,256)` raw bytes, then a **fixed-size reserved block** of
`pretokenize::SPECIAL_BLOCK` (64) slots for the protected markers, then BPE
merges from `bpe::MERGE_BASE` upward. Only the first `NUM_SPECIAL` slots are
allocated; the rest are padding, and `special_char_from_id` returns `None` for
them.

The block is padded rather than sized to the marker count because **merge ids are
absolute and are baked into every saved `merges.bin`**. Keying `MERGE_BASE` to
`NUM_SPECIAL` meant adding one marker shifted every learned merge id, silently
reinterpreting every tokenizer and model already on disk (an old `merges.bin`'s
first merges would decode as the newly added markers). With the padding, a new
marker takes the next free slot and nothing below it moves.

Adding a marker to `SPECIAL_TOKENS` is therefore now safe for existing models.
Raising `SPECIAL_BLOCK` is **not** — it moves `MERGE_BASE` and needs a fresh
tokenizer and a re-pretrain. A marker count past the block fails the build via a
`const` assert rather than corrupting a corpus.

**The vocabulary size is the tokenizer's real lever, and the output projection
is tied so you can afford it.** Held-out bytes/token over the real corpora,
proportional sampling, baseline pretokenizer:

| vocab | FineWeb | vs 8k | SFT chat |
|---|---|---|---|
| 4096 | 3.41 | −12% | 3.43 |
| **8192** | **3.89** | — | 3.82 |
| 16384 | 4.28 | **+10%** | 4.13 |
| 32768 | 4.59 | **+18%** | 4.36 |
| 65536 | 4.79 | +23% | 4.49 |

Because `Model::w_output` is tied to `embedding_table` (see the field, and
`MODEL_VERSION` 6), the vocabulary costs `vocab * d` rather than `2 * vocab * d`
— so **tied at vocab 2V is the same parameter count as untied at V**. At the
Large shape (d 512, 8 layers) untied@8k and tied@16k are both 33.58M
parameters, and tied@16k compresses 10% better. It is also near-free in compute:
the output projection doubles but there are proportionally fewer tokens, netting
out around +1% per corpus byte (32k is +13%, which is a real cost).

**Compression is not the objective, though, and this table cannot settle the
vocabulary.** A larger vocabulary means each row is seen less often and the
softmax is harder, and there is a real optimum that bytes/token cannot find.
Settle it by pretraining at each size and reading the **language suite**, which
is denominated in bits per *byte* and is therefore comparable across tokenizers
— the only metric here that is. Do not compare `bits_per_token` across
vocabularies; see "A benchmark must not be denominated in tokens".

**The pretokenizer is not a lever, and this has been tried.** Space runs,
newline runs, wide-set punctuation runs and digit grouping were each measured at
vocab 8192: every variant landed within ±1% of the current splitter on FineWeb,
and GPT-4-style punctuation-run grouping was *worse* (−1.0%) at every vocabulary
size. The only real signal was space+newline runs on chat data (+1.3%). The 78%
that `17767d0` recovered was the tokenizer win; there is not a second one in the
splitting rules. `crates/tokenize/examples/tokexp.rs` is the bench — it asserts
its baseline config reproduces `pre_tokenize_str` before comparing anything.

**Corpus files are `u16`.** `corpus.bin` and its SFT/KTO siblings carry an 8-byte
header (`LLMC`, version, element width) followed by little-endian `u16` ids, and
the corpus stays `u16` in memory through the training loop — halving both the file
and the resident `Vec` on a machine where that is the binding constraint. Ids are
widened to `u32` one chunk at a time at the step boundary, so `train_step` and the
CUDA path are unchanged. `max_vocab` is clamped to `bpe::MAX_VOCAB` (65,536), and
`CorpusWriter` rejects an out-of-range id rather than truncating it. A corpus
written before the header fails to load with a message telling you to rebuild.
