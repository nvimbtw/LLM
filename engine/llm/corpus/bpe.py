"""The tokenizer: training, loading, encoding, decoding.

Wraps HuggingFace `tokenizers` (byte-level BPE) rather than reimplementing the
Rust crate's BPE — it is the same algorithm and is itself Rust underneath. What
this module owns is the *policy* the library has no opinion on: the reserved
special-token block, saving/loading as one self-contained file, and how much text
a vocabulary is allowed to learn from.
"""

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.implementations import ByteLevelBPETokenizer

from llm import storage
from llm.corpus import special

# Merges seen fewer times than this are noise rather than structure.
MIN_FREQUENCY = 2

# Byte-level BPE starts from every byte, so its alphabet is this whatever the
# corpus is.
BYTE_ALPHABET = 256

# How much text a vocabulary is learned from, however much was selected.
#
# This is the ceiling that was missing. The trainer holds a count for every
# distinct pretoken it sees, so its peak scales with the *variety* of the text
# and cannot be streamed away — feeding it in chunks produces the same map at the
# end. Measured on Wikipedia shards: 308 MB peak on 57 MB of text, 1.6 GB on
# 866 MB, 3.6 GB on 3.4 GB, which extrapolates to ~10 GB for Wikipedia alone and
# ~15 GB for the whole input tree. On a 12 GB machine that is the OOM.
#
# ADR 0004 measured the same thing on the Rust engine and set 16 MiB. Re-measured
# here, over the whole input tree at vocab 8192, held-out bytes/token:
#
#     budget   train   peak    FineWeb   PG-19   TinyStories
#      4 MiB    2.7s   275 MB    3.683   3.336     3.691
#     16 MiB    4.2s   367 MB    3.714   3.346     3.703
#     64 MiB    8.3s   576 MB    3.739   3.351     3.663
#    256 MiB   22.1s  1150 MB    3.738   3.353     3.566
#
# Flat above 64 MiB, which is the ADR's finding again: composition is the lever,
# size is not. 64 MiB rather than 16 because the 0.7% it buys on web text is 0.7%
# fewer tokens to train on for the same corpus — worth four seconds once — and
# because 576 MB leaves twenty times the headroom on this machine. Raising it
# further buys nothing and costs linearly.
SAMPLE_BYTES = 64 << 20

# The share of the budget every selected dataset is guaranteed, before the rest
# is split in proportion to size.
#
# Without a floor, one dominant corpus takes nearly the whole sample: FineWeb is
# 98% of the text in a typical selection here, which would leave the merges
# fitted to web text alone. Without proportionality the opposite happens — ADR
# 0004 measured an equal split giving FineWeb 27.9% of the sample while it was
# 98.4% of the tokens actually encoded, and fixing that gained 0.045 weighted
# bytes/token.
MIN_DATASET_SHARE = 0.05

# How much to read from each file the sample touches. Small, because coverage per
# byte is the point: at 64 KiB a 6 MiB share spans ~96 files spread across the
# dataset instead of ~24.
SAMPLE_TAKE_BYTES = 64 << 10


def train(files: list[Path], max_vocab: int, out_dir: Path | None = None) -> Tokenizer:
    """Train a byte-level BPE tokenizer over `files` and save it.

    Saved as a single `tokenizer.json` rather than the `vocab.json`/`merges.txt`
    pair: that pair does NOT carry the added-token configuration, so a tokenizer
    reloaded from it splits every `<|marker|>` back into byte merges — the model
    then never sees the single ids it was trained on.
    """
    out_dir = storage.ensure_dir(out_dir or storage.tokenizer_dir())

    tok = ByteLevelBPETokenizer()
    tok.train(
        files=[str(f) for f in files],
        vocab_size=max_vocab,
        min_frequency=MIN_FREQUENCY,
        special_tokens=special.token_list(),
        show_progress=False,
    )

    tok.save(str(storage.tokenizer_file(out_dir)))
    return load(out_dir)


def plan_sample(files, budget: int = SAMPLE_BYTES) -> list[tuple[Path, int]]:
    """`(path, bytes)` pairs to learn merges from, spread across the selection.

    `files` is anything with `.path` and `.dataset` — duck-typed, so this module
    still imports nothing from `sources`.

    Three rules, all of them from ADR 0004:

    * Each dataset gets `MIN_DATASET_SHARE` of the budget, and the remainder is
      split in proportion to how much text it actually holds. A dataset smaller
      than its share hands the difference back to the others.
    * Within a dataset the sample is *strided* across its files, never taken from
      the front. A sharded corpus is ordered, so the first N shards are one
      region of it and not a sample of it — the same mistake the Rust engine made
      when a `div_ceil(1)` collapsed its stride and FineWeb was read from 50 of
      its 5,498 shards.
    * Files are truncated to a byte count rather than taken whole, so one 28 MB
      book cannot eat a 2 MB share and re-skew the composition the shares exist
      to control.
    """
    by_dataset: dict[str, list[tuple[Path, int]]] = {}
    for source in files:
        by_dataset.setdefault(source.dataset, []).append(
            (source.path, source.path.stat().st_size)
        )

    drawn: list[tuple[Path, int]] = []
    for name, share in _shares({n: sum(s for _, s in f) for n, f in by_dataset.items()}, budget):
        drawn.extend(_draw(by_dataset[name], share))
    return drawn


def _shares(sizes: dict[str, int], budget: int) -> list[tuple[str, int]]:
    """Each dataset's slice of the budget, in the order given.

    Solved by water-filling: a dataset that holds less text than its share takes
    only what it has, and what it could not use is redistributed among the rest.
    Without that, selecting one 400 KB corpus alongside a large one silently
    shrinks the sample by the difference.
    """
    settled: dict[str, int] = {}
    pool = dict(sizes)
    while pool:
        total = sum(pool.values())
        if total <= budget:
            settled.update(pool)
            break
        floor = min(budget * MIN_DATASET_SHARE, budget / len(pool))
        rest = budget - floor * len(pool)
        planned = {name: floor + rest * size / total for name, size in pool.items()}

        short = {name: size for name, size in pool.items() if size <= planned[name]}
        if not short:
            settled.update({name: int(want) for name, want in planned.items()})
            break
        settled.update(short)
        budget -= sum(short.values())
        pool = {name: size for name, size in pool.items() if name not in short}

    return [(name, int(settled.get(name, 0))) for name in sizes]


def _draw(files: list[tuple[Path, int]], share: int) -> list[tuple[Path, int]]:
    """`share` bytes of one dataset, spread across its files.

    The walk is a stride and then the gaps between it: the first pass touches
    files evenly spaced across the whole range, and only if the share is still
    unfilled — because the files are smaller than `SAMPLE_TAKE_BYTES`, which is
    how TinyStories reads — does it come back for the ones in between. Either
    way the sample spans the dataset rather than its first megabytes.
    """
    if share <= 0 or not files:
        return []

    # `SAMPLE_TAKE_BYTES` is a floor on how much to read once a file is opened,
    # not a ceiling: a dataset held in a few large files would otherwise cap at
    # that much per file and hand back most of a share it can easily fill.
    take_each = max(SAMPLE_TAKE_BYTES, -(-share // len(files)))
    wanted = max(1, min(len(files), int(share // take_each)))
    stride = max(1, len(files) // wanted)

    drawn: list[tuple[Path, int]] = []
    spent = 0
    for offset in range(stride):
        for i in range(offset, len(files), stride):
            if spent >= share:
                return drawn
            path, size = files[i]
            take = min(take_each, share - spent, size)
            if take > 0:
                drawn.append((path, take))
                spent += take
    return drawn


def write_sample(drawn: list[tuple[Path, int]], out_dir: Path) -> list[Path]:
    """Materialize a sample plan as files the trainer can be pointed at.

    One file per source file rather than one big concatenation: the trainer
    parallelizes its counting over the files it is given, and this machine has
    six cores to give it.

    Each slice is cut back to its last newline. A byte-level BPE pretokenizer
    splits on whitespace, so a cut mid-word would invent a pretoken that does not
    occur in the text and spend merges on it.
    """
    storage.ensure_dir(out_dir)
    written = []
    for index, (path, take) in enumerate(drawn):
        raw = path.open("rb").read(take)
        cut = raw.rfind(b"\n")
        if cut > 0:
            raw = raw[: cut + 1]
        # A slice can land inside a multi-byte character at either end; replacing
        # is right here, where the text is only ever counted, and it is what
        # `pack` does when it reads the same files for real.
        out = out_dir / f"{index:06d}-{path.stem}.txt"
        out.write_text(raw.decode("utf-8", errors="replace"), encoding="utf-8")
        written.append(out)
    return written


def load(directory: Path | None = None) -> Tokenizer:
    """Load a trained tokenizer. Defaults to the shared build; pass a model's
    directory to load the copy saved beside its weights."""
    path = storage.tokenizer_file(directory)
    if not path.is_file():
        raise FileNotFoundError(
            f"No tokenizer at {path}. Train a model first — the training "
            f"pipeline builds one and copies it next to the weights."
        )
    return Tokenizer.from_file(str(path))


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def encode_batch(tok: Tokenizer, texts: list[str]) -> list[list[int]]:
    """Encode several texts at once, one list of ids each.

    The library parallelizes a batch across cores; encoding one text at a time
    leaves five of this machine's six idle. Measured over 327 MB of Wikipedia
    shards: 2.3 MB/s one at a time against 7.0 MB/s batched — 3.0x on the step
    that dominates a rebuild.

    Identical output to calling `encode` per text: the batch is a scheduling
    detail, and each text is still encoded independently.
    """
    return [encoding.ids for encoding in tok.encode_batch(texts)]


def decode(tok: Tokenizer, ids: list[int]) -> str:
    return tok.decode(ids, skip_special_tokens=False)


def vocab_size(tok: Tokenizer) -> int:
    return tok.get_vocab_size()


def merge_count(tok: Tokenizer) -> int:
    """How many merges the tokenizer learned — everything in the vocabulary that
    is neither a raw byte nor a reserved marker.

    Derived rather than read back out of the saved merge list, which means
    parsing the whole tokenizer file to count its lines.
    """
    return max(0, vocab_size(tok) - BYTE_ALPHABET - len(special.token_list()))


def token_id(tok: Tokenizer, token: str) -> int:
    """The single id for a special token. Raises rather than returning None —
    a missing marker means the tokenizer was trained without the reserved block,
    and every downstream document boundary would be silently wrong."""
    tid = tok.token_to_id(token)
    if tid is None:
        raise ValueError(f"{token!r} is not in the tokenizer's vocabulary")
    return tid
