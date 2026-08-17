# A build may not scale with its corpus

[0004](0004-corpus-ingestion-streams.md) established that ingestion streams, and
the Python port kept that: `pack.pack` flushes every 2M tokens, `load_corpus`
maps rather than reads, `EpochStream` never materializes an epoch. Measured on
the 4.17B-token corpus, the training loop needs **681 MB of host RSS and 383 MB
of VRAM**.

Three things did not keep it. Each was found by measurement on a 12 GB machine
against a 51 GB input tree, and each is bounded now.

## 1. The tokenizer learned from everything selected

`vocab.build` handed every selected file to the BPE trainer. Peak RSS against
Wikipedia shards, vocab 8192:

| text in | peak RSS | time |
|---|---|---|
| 57 MB | 308 MB | 5.6 s |
| 219 MB | 734 MB | 14.2 s |
| 866 MB | 1,629 MB | 44.5 s |
| 3,445 MB | 3,599 MB | 237.8 s |

Sublinear — distinct pretokens saturate — but it does not flatten, and the fit
extends to **~10 GB for Wikipedia alone (19 GB) and ~15 GB for the whole input
tree (51 GB)**, at about an hour of CPU. The *Chinchilla 1* preset selects
`Wikipedia` uncapped, so this was live rather than hypothetical. The `count` caps
in a config were the only thing preventing it, per run and opt-in.

**Chunking does not help, and this is the part worth remembering.** What the
trainer holds is a count for every distinct pretoken, not the text; feed it the
corpus in a hundred pieces and the map at the end is the same size. Only seeing
less *distinct* text bounds it. That is why 0004's answer was a sample and not a
stream, and it is what `bpe.plan_sample` restores: a byte budget, split across
datasets with a floor, strided within each dataset, and truncated per file.

Re-measured here — same planner, four budgets, held-out bytes/token at vocab
8192:

| budget | train | peak | FineWeb | PG-19 | TinyStories |
|---|---|---|---|---|---|
| 4 MiB | 2.7 s | 275 MB | 3.683 | 3.336 | 3.691 |
| 16 MiB | 4.2 s | 367 MB | 3.714 | 3.346 | 3.703 |
| **64 MiB** | **8.3 s** | **576 MB** | **3.739** | **3.351** | **3.663** |
| 256 MiB | 22.1 s | 1,150 MB | 3.738 | 3.353 | 3.566 |

0004's finding again: composition is the lever, size is not. 64 MiB is the knee —
past it nothing moves and cost is linear. **A build now takes 6 seconds and
365 MB where it took an hour and would have taken 15 GB.**

Two details are load-bearing and both are guarded in `tests/test_build_bounds.py`:

* **The stride.** A sharded corpus is ordered, so its first N shards are one
  region of it and not a sample of it. 0004 records the Rust version of this bug
  (a collapsed `div_ceil` read FineWeb from 50 of 5,498 shards); the same trap
  appears in §3 below, from the other direction.
* **Per-file truncation.** `input/Books` holds a 28 MB file. Taken whole against
  a 2 MB share it is 14x that dataset's budget and re-skews the composition the
  shares exist to control.

## 2. The packer computed the histogram and threw it away

`Collector.__init__` called `frequent_tokens(corpus)` to find the 32 ids the
embedding metrics track. Two full passes over the mapping — `corpus.max()`, then
a blocked `bincount` — **64.1 s and 8.0 GB resident before step 1 of every run**,
against a loop that needs 681 MB. The memory is clean and reclaimable, so it is
not itself an OOM; what it does is evict everything else from the page cache at
the exact moment the run starts reading the corpus through it.

Every id already passes through `pack.pack` as a numpy array, at the one moment
the whole corpus is in hand. It is counted there and written into
`segments.json` (~50 KB), and `sft.build` does the same. Verified byte-exact
against a full scan.

A corpus packed before this existed has no counts, and then
`diagnostics.sampled_histogram` reads 32 blocks spread across the file — 512 MiB
touched whatever the corpus weighs.

**Blocks, not a stride.** `corpus[::16]` looks like it reads a sixteenth and
reads all of it: tokens are 2 bytes and a page is 4096, so every 16th token still
faults every page. And the blocks must span the file — measured on the 4.17B
corpus, the first 200M tokens contain **16 of the true top 32 tokens**, because
`corpus.bin` is laid out in dataset and stage order. Both halves of that are easy
to get wrong in the direction that looks correct.

## 3. It was built whether or not anything read it

The collector was constructed at the top of `loop.run`, unconditionally —
including on `on_progress=None` runs, which never emit a frame and never read a
diagnostic. `bench.py` paid it on every timing it took. It is now built on the
first diagnosed step (`_LazyCollector`), which is at least
`interval * diag_interval` steps into a warm run.

Together: **peak host RSS through a real 718-step run on the 4.17B-token corpus
fell from 8,374 MB to 1,301 MB**, with the diagnostics unchanged — 32 tracked
tokens, PCA rows, per-head entropy, all still populated.

## And the step that was actually slow

Packing, which a rebuild spends nearly all its time in, encoded one file at a
time and used one of six cores. `tok.encode_batch` over ~4 MiB of text at a time
measured **2.3 MB/s → 7.0 MB/s (3.0x)**: about six hours down to two on the full
input tree. Batching by *characters* rather than by file count is deliberate —
the bound is what the batch holds, not how many things it holds, and one file
here is 28 MB.

The output is byte-identical, and that is checked rather than assumed: the same
900 files packed both ways give the same SHA-256 over `corpus.bin`. If it ever
were not, every corpus packed since would silently differ from the text it
claims to be.

**The SFT build is the same story and was fixed the same way.** Rendering
conversations into the corpus/mask pair encoded segment by segment, one
conversation at a time. `chat.template` now splits into `training_segments`
(pure string work) and `assemble_example` (pure list work), so a build can cut
thousands of conversations into pieces first and hand them all to the tokenizer
in one call — **4.4x on the encode, 2.77x end to end** over 75,000 real
conversations, where the remainder is JSON parsing and the buffers.

The byte-identity requirement is *stricter* here, because the mask moves with the
tokens: a batch that tokenized differently would write a different corpus under
the same dataset key **and** shift which positions are trained. Checked at both
levels — same ids and same mask per conversation, and the same SHA-256 over both
`corpus.bin` and `mask.bin` for the whole build. Segments are still encoded
individually inside the batch, which is the property the mask depends on: a
marker glues to the text beside it in a whole-string encoding, so there is no
clean token boundary at one.
