# Corpus ingestion (streaming)

`build_dataset` streams. Peak memory is **constant in corpus size**, measured at
~1.1 GB whether the corpus is 10 MB or 150 MB, because the two phases are separated:

1. **Learn the merges from a bounded sample.** `plan_bpe_sample` takes
   `BPE_SAMPLE_BYTES` (16 MiB) split **in proportion to each dataset's size**,
   with a floor of `MIN_DATASET_SAMPLE_SHARE` (5%) so a small early-curriculum
   dataset can't be cut to near byte level. `bpe::train` deduplicates the sample
   to its distinct pretokens first, so its peak scales with the sample's
   *vocabulary* rather than its length (measured: 267 MB peak to train a full
   8192 table on 10.6 MiB and then encode 8 MiB of held-out text).

   **This was an equal split per dataset, and the change is measured, not
   assumed.** Equal shares gave FineWeb 27.9% of the sample while it was 98.4%
   of the text actually encoded, so the merges were fitted to a distribution the
   corpus does not have. Held-out bytes/token at vocab 8192 over the real
   corpora:

   | split | weighted | FineWeb | Wikipedia | TinyStories | SFT chat |
   |---|---|---|---|---|---|
   | equal | 3.903 | 3.82 | 3.63 | 3.70 | 3.76 |
   | proportional | **3.948** | **3.87** | 3.55 | 3.66 | **3.82** |

   It is a reallocation: Wikipedia and TinyStories each give up 1–2%. They are
   ~2% of the tokens the model reads, and both distributions that matter (web
   text, and the chat data post-training runs on) gain.

   **The sample's *size* is not a lever and raising it is wasted time.** 16 MiB
   against 64 MiB measured 3.967 → 3.945 at vocab 8k and 4.872 → 4.889 at 64k —
   inside noise, at 4x the tokenizer-training cost. What mattered was
   composition, and now the vocabulary size (see "Vocabulary layout").

   **The stride was also broken and is fixed.** `share.div_ceil(1)` is `share`,
   so the stride computation always produced 1 and a sharded dataset was read
   from its first files only — at FineWeb's real shape, 50 of 5,498 shards. The
   stride is now derived from how many files the share actually consumes.
   `a_sharded_dataset_is_sampled_across_its_whole_range` is the guard.
2. **Stream-encode everything** through one reused `bpe::Encoder`, in ~4 MiB chunks,
   into a `bpe::CorpusWriter`.

**Chunks cut only on newlines, and that is exact, not approximate.**
`replace_special_characters` maps `'\n'` to `'Ċ'` and `split_punctuation` splits on
it, so a newline is already a hard pretoken boundary — chunked encoding is
byte-identical to encoding a file whole. `bpe::tests::test_chunking_at_newlines_is_seam_free`
is the guard; if it ever fails the design is invalid. A file with no newline inside
`CHUNK_MAX_BYTES` has no safe cut point and is refused by name.

Builds are cancellable between chunks and finalize atomically: corpus, merges and
segments are written to `*.partial` and renamed, so a cancelled or failed build
leaves the previous dataset untouched.

**The trainer and the encoder apply merges by the same rule, and that is the whole
correctness story of the tokenizer.** `bpe::train` applies each learned merge to
every *word* independently, in rank order, to fixed point — exactly what
`Encoder::merge_word` does when it repeatedly takes the lowest-id applicable merge.
`bpe::every_learned_merge_is_reachable_by_the_encoder` is the guard, and it is a
total check over the table rather than a sample.

It has to be, because the failure is silent and expensive. The trainer used to walk
a flat corpus while tracking pair positions in a side index, and the index went
stale the moment a merge shortened a word: every later position shifted left by one
and was never corrected, so occurrences were skipped and the merges learned after
that point were fitted to a segmentation `Encoder` cannot reproduce. On the real
8192 table that left **3,004 of 7,872 merges (38%) unreachable** — `" the"` was
learned as `" th" + "e"` at rank 6, but the encoder reaches `[" t", "he"]` and
stops, so `" the"` sat in the vocabulary and could never be emitted. The casualties
were the most frequent words in English (`" the"`, `" in"`, `" from"`, `" they"`,
`" have"`, `" were"`), 51% of emitted tokens were single characters, and
bytes-per-token sat at **2.13**. Nothing downstream noticed: the vocabulary looked
full and the models trained fine, just against a third less compression than they
should have had. Rebuilding the same 8192 table on the same data now gives **3.79
bytes/token, 17% single-character, 0 fragmenting merges**.

The invariant is "a merge's text re-encodes to a **single** token", not "to that
exact id". Those differ only for aliases — two ids decoding to the same text, of
which the encoder naturally reaches one. On a real table that is 4 merges, all a
lone space, costing 4 embedding rows and no compression.

Because the two now agree, `bpe::train`'s returned corpus *is* what `Encoder`
produces (`the_returned_corpus_matches_what_the_encoder_produces`). `build_dataset`
still discards it and stream-encodes instead, so there is exactly one code path
producing token ids.

**Any change here needs a fresh tokenizer and a re-pretrain.** Merge ids are
absolute and baked into every `merges.bin`; a model carries the vocabulary it was
trained under.
