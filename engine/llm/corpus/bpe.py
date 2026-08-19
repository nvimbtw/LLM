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

# How much to read at each point the sample touches. Small, because coverage per
# byte is the point: at 64 KiB a 6 MiB share is drawn from ~96 places spread
# across the dataset instead of ~24.
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


def shares(sizes: dict[str, int], budget: int) -> list[tuple[str, int]]:
    """Each dataset's slice of the sample budget, in the order given.

    Half of ADR 0004; `remote.sample` is the other half, drawing a slice strided
    across the dataset it belongs to.

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


def write_chunks(chunks, out_dir: Path) -> list[Path]:
    """Materialize drawn text as files the trainer can be pointed at.

    One file per chunk, because the trainer parallelizes its counting over the
    files it is given and this machine has six cores to give it. The text
    arrives already sliced — a chunk is one stop of the stride — so there is
    nothing to truncate here.

    Bounded by whoever drew the chunks (`bpe.SAMPLE_BYTES`), which is why a
    vocabulary build is the one place text still touches this disk: 64 MiB of
    sample against holding it as Python strings and giving up the cores.
    """
    storage.ensure_dir(out_dir)
    written = []
    for index, (dataset, text) in enumerate(chunks):
        out = out_dir / f"{index:06d}-{_slug(dataset)}.txt"
        out.write_text(text, encoding="utf-8")
        written.append(out)
    return written


def _slug(name: str) -> str:
    keep = "".join(c if c.isalnum() or c in "-_" else "-" for c in name)
    return keep.strip("-")[:40] or "dataset"


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
