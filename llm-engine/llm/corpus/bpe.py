"""The tokenizer: training, loading, encoding, decoding.

Wraps HuggingFace `tokenizers` (byte-level BPE) rather than reimplementing the
Rust crate's BPE — it is the same algorithm and is itself Rust underneath. What
this module owns is the *policy* the library has no opinion on: the reserved
special-token block, saving/loading as one self-contained file, and how much text
a vocabulary is allowed to learn from.
"""

import codecs
import functools
import re
from pathlib import Path

from tokenizers import Tokenizer, decoders, models
from tokenizers.implementations import ByteLevelBPETokenizer

from llm import storage
from llm.corpus import special

# Merges seen fewer times than this are noise rather than structure.
MIN_FREQUENCY = 2

# Byte-level BPE starts from every byte, so its alphabet is this whatever the
# corpus is.
BYTE_ALPHABET = 256

# The id of the first merge learned. The trainer lays a vocabulary out as the
# reserved block, then the byte alphabet, then one id per merge *in the order it
# learned them* — which is what makes the merge sequence readable off the
# vocabulary rather than out of the saved merge list.
FIRST_MERGE_ID = BYTE_ALPHABET + len(special.token_list())

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


# How many parsed vocabularies to keep. A serving process answers for the handful
# of models being chatted with, not for the whole registry, and one is a few MB of
# Rust structures — so this bounds the memory while still hitting on every request
# of a conversation.
#
# What the cache is worth, measured on a 539 KB `tokenizer.json` (vocab 8192)
# encoding a 195-character message:
#
#     load, cold (parse)    9.945 ms
#     load, warm (cached)   0.009 ms
#     encode                0.055 ms
#
# So a count used to spend 99% of itself re-reading a file it had just read. That
# is affordable once per stored message and is not affordable per pause in
# typing, which is what the composer's context meter asks for.
CACHED_VOCABULARIES = 8


def _byte_glyphs() -> dict[str, int]:
    """Return the reversible ByteLevel glyph-to-byte table used by our BPE tokenizer."""
    plain = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    glyphs = plain.copy()
    next_glyph = 256
    for byte in range(256):
        if byte not in plain:
            plain.append(byte)
            glyphs.append(next_glyph)
            next_glyph += 1
    return {chr(glyph): byte for byte, glyph in zip(plain, glyphs)}


BYTE_GLYPHS = _byte_glyphs()


class ByteStreamDecoder:
    """Decode ByteLevel token IDs incrementally, buffering incomplete UTF-8 bytes."""

    def __init__(self, tokenizer: Tokenizer):
        """Keep the tokenizer's added tokens and a fresh UTF-8 state for one reply."""
        self.tokenizer = tokenizer
        self.added = tokenizer.get_added_tokens_decoder()
        self.utf8 = codecs.getincrementaldecoder("utf-8")("replace")

    def push(self, token_id: int) -> str:
        """Return the new text for one valid ID, possibly empty for a partial character."""
        added = self.added.get(token_id)
        if added is not None:
            text = self.utf8.decode(b"", final=True) + added.content
            self.utf8 = codecs.getincrementaldecoder("utf-8")("replace")
            return text
        token = self.tokenizer.id_to_token(token_id)
        if token is None:
            raise ValueError(f"unknown token id {token_id}")
        return self.utf8.decode(bytes(BYTE_GLYPHS[char] for char in token))

    def finish(self) -> str:
        """Flush an unfinished UTF-8 character after the final generated ID."""
        return self.utf8.decode(b"", final=True)


def stream_decoder(tokenizer: Tokenizer) -> ByteStreamDecoder | None:
    """Use incremental decoding only for validated ByteLevel BPE vocabularies."""
    if not isinstance(tokenizer.model, models.BPE) or not isinstance(
        tokenizer.decoder, decoders.ByteLevel
    ):
        return None
    added_ids = tokenizer.get_added_tokens_decoder()
    if any(
        char not in BYTE_GLYPHS
        for token, token_id in tokenizer.get_vocab().items()
        if token_id not in added_ids
        for char in token
    ):
        return None
    return ByteStreamDecoder(tokenizer)


@functools.lru_cache(maxsize=CACHED_VOCABULARIES)
def _parse(path: str, _stamp: tuple[int, int]) -> Tokenizer:
    """`_stamp` is never read. It is in the key so that a file rewritten in
    place — an SFT run saving its vocabulary over the base model's — misses
    rather than being served from a copy parsed before it moved."""
    return Tokenizer.from_file(path)


def load(directory: Path | None = None) -> Tokenizer:
    """Load a trained tokenizer. Defaults to the shared build; pass a model's
    directory to load the copy saved beside its weights.

    Parsed once per file and shared thereafter: `/tokenize/count` is called for
    every message stored and again as one is typed, and re-reading a multi-MB
    JSON per call dominated both. Callers get the *same* object, so a tokenizer
    is to be treated as immutable — nothing here mutates one, and the encode
    paths only take `&self`."""
    path = storage.tokenizer_file(directory)
    if not path.is_file():
        raise FileNotFoundError(
            f"No tokenizer at {path}. Train a model first — the training "
            f"pipeline builds one and copies it next to the weights."
        )
    stat = path.stat()
    return _parse(str(path), (stat.st_mtime_ns, stat.st_size))


def encode(tok: Tokenizer, text: str) -> list[int]:
    return tok.encode(text).ids


def encode_batch(tok: Tokenizer, texts: list[str]) -> list[list[int]]:
    """Encode several texts at once, one list of ids each.

    The library parallelizes a batch across cores; encoding one text at a time
    leaves five of this machine's six idle. Measured over 327 MB of Wikipedia
    shards: 2.3 MB/s one at a time against 7.0 MB/s batched — 3.0x on the step
    that dominates a rebuild.

    `encode_batch_fast` rather than `encode_batch`, because the ids are the whole
    of what anything here reads and the difference between the two calls is what
    the library builds *beside* them: every token's own string, the character
    offsets it was trimmed to, the type and attention masks. Measured on 4.4 MB
    of `gutenberg_eng-clean`, that is 1.8x the CPU and 1.2x the peak memory for
    four fields nobody asks for. The ids are identical, and
    `tests/test_encode_chunking.py` is what says so.

    Identical output to calling `encode` per text: the batch is a scheduling
    detail, and each text is still encoded independently.
    """
    return [encoding.ids for encoding in tok.encode_batch_fast(texts)]


#: How much of one document goes into a single item of an encode batch.
#:
#: The library parallelizes a batch across its *items*, so a batch is only as
#: balanced as the documents in it — and a document is not an amount of text.
#: `gutenberg_eng-clean` holds books from a few KB to 27 MB, so a 4 MiB batch is
#: nine of them, five cores finish early and the largest one is the batch. Worse,
#: the pretokenization of a whole document is held at once: one 27.6 MB book cost
#: 30.6 s and 4.5 GiB of peak RSS whole, against 5.9 s and 1.2 GiB in pieces.
#:
#: Throughput is flat from 8 KiB to 256 KiB on the corpora here, so the size is
#: chosen for the tail rather than for the middle: 64 KiB gives six cores
#: something to divide even when the batch is a single 400 KB book.
CHUNK_CHARS = 1 << 16

#: A cut is made *at* the whitespace, so the character before it must be
#: non-whitespace and the whitespace itself must be one this tokenizer's own
#: regex agrees is whitespace. Spelled out rather than `\s` on both sides because
#: Python's `\s` and Rust's differ at the edges (`\x1c`-`\x1f` are whitespace to
#: one and not the other), and this pattern has to be the stricter of the two in
#: both positions or the cut is not a boundary.
_CUT_POINT = re.compile(r"\S[ \t\n\r\f\v]")


def split_for_encoding(text: str, limit: int = CHUNK_CHARS) -> list[str]:
    """`text` in pieces that encode to exactly what the whole of it encodes to.

    A cut lands on a whitespace character whose predecessor is not whitespace,
    and that position is always a pretoken boundary: every alternative in the
    byte-level pretokenizer's pattern either stops at whitespace or matches
    whitespace only, so the pretoken holding the character before the cut ends
    there — and nothing in the pattern looks backwards, so the text after the cut
    pretokenizes the same alone as it did in place. BPE merges never cross a
    pretoken, so concatenating the pieces' ids reproduces the document's.

    A stretch with no such position in it — a megabyte of base64 — is left whole
    rather than cut somewhere that is not a boundary. Correctness is not
    negotiable for a speedup; a pathological document simply keeps the cost it
    already had.
    """
    if len(text) <= limit:
        return [text]
    pieces = []
    start = 0
    while len(text) - start > limit:
        found = _CUT_POINT.search(text, start + limit)
        if found is None:
            break
        cut = found.start() + 1
        pieces.append(text[start:cut])
        start = cut
    pieces.append(text[start:])
    return pieces


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
    return max(0, vocab_size(tok) - FIRST_MERGE_ID)


#: How many merges `merge_sequence` reports. The head is the part that carries
#: the story — the first merges are the language's own bigrams — and the frame
#: this rides on is a WebSocket message, not a file.
MERGE_SEQUENCE_HEAD = 256


def merge_sequence(
    tok: Tokenizer, counts=None, limit: int = MERGE_SEQUENCE_HEAD
) -> list[dict]:
    """The first `limit` merges, in the order BPE learned them.

    `counts` is the packed corpus's own token histogram, so `count` is how often
    the merged token occurs in the corpus this vocabulary encoded. That is
    **not** the pair frequency that earned the merge: the trainer keeps that
    count only while it is choosing, and the `tokenizers` library exposes
    neither it nor a callback to observe it, so recovering it would mean
    reimplementing BPE training beside the library that does it. The corpus
    frequency is the honest thing that can be had for nothing — every id is
    already counted on the way to disk.

    The tokens are decoded rather than spelled the way the vocabulary spells
    them: byte-level BPE writes a leading space as `Ġ`, and a reader should not
    have to know that.
    """
    end = min(vocab_size(tok), FIRST_MERGE_ID + limit)
    return [
        {
            "token": token_text(tok, tid),
            **({"count": int(counts[tid])} if counts is not None else {}),
        }
        for tid in range(FIRST_MERGE_ID, end)
    ]


def token_text(tok: Tokenizer, tid: int) -> str:
    """How one token is spelled on its own, for a reader looking at it.

    Byte-level BPE learns merges that are *half* of a multi-byte character, and
    decoding one of those alone yields U+FFFD — which reads as "this token is
    broken" rather than "this token is a fragment". The vocabulary's own
    spelling (`Ã©`, `Ġthe`) is ugly and true, so it wins in exactly that case.
    """
    text = decode(tok, [tid])
    if text and "�" not in text:
        return text
    return tok.id_to_token(tid) or ""


def token_id(tok: Tokenizer, token: str) -> int:
    """The single id for a special token. Raises rather than returning None —
    a missing marker means the tokenizer was trained without the reserved block,
    and every downstream document boundary would be silently wrong."""
    tid = tok.token_to_id(token)
    if tid is None:
        raise ValueError(f"{token!r} is not in the tokenizer's vocabulary")
    return tid
