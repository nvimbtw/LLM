"""Cutting a document up to encode it, and putting it back together.

Packing hands the encoder pieces rather than whole documents, for two reasons
that arrive together: the library parallelizes across the *items* of a batch, so
a batch of nine books is nine work units for six cores; and a batch built out of
whole documents is as large as the largest document in it, which on
`gutenberg_eng-clean` is 27.6 MB of text and 4.5 GiB inside the encoder.

Both of those are speed and memory, and neither is worth anything if the corpus
that comes out is not the corpus that would have come out. These are the
assertions that say it is — the same claim `bpe.split_for_encoding` makes in
prose, made against real encodings instead.
"""

import numpy as np
import pytest

from llm import storage
from llm.corpus import bpe, pack, spans as spans_mod

# Text chosen for the boundary rule rather than for realism: runs of blank
# lines, a tab, a non-breaking space, CJK with no spaces in it at all, markers
# that must survive as single tokens, and a stretch that offers no cut point.
STRESS = (
    "the cat sat on the mat.\n\n\nthe cat ate the fish.\n"
    "\ta dog sat on a log.  the dog saw the cat.\n"
    "birds sing in the morning and the morning is bright.\n"
    "<|par|>a paragraph the tokenizer keeps whole<|/par|>\n"
    "猫がマットの上に座っていた\n"
    + "0123456789abcdef" * 64
    + "\n   \n \n"
    + "the fish swam under the log at night.\n"
) * 8


@pytest.fixture
def stress_documents() -> list[str]:
    """One list of texts, several shapes of pathology."""
    return [
        STRESS,
        "",
        " ",
        "\n\n\n\n",
        "no-whitespace-at-all-" * 200,
        "trailing whitespace at the end of the document   ",
        "   leading whitespace at the start of the document",
        STRESS[:1] + STRESS,
    ]


# ── the cut ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("limit", [1, 2, 7, 16, 64, 257, 4096])
def test_the_pieces_are_the_document(stress_documents, limit):
    """Before anything about encoding: cutting must not lose or reorder a
    character. Everything below is meaningless if this is not true."""
    for text in stress_documents:
        assert "".join(bpe.split_for_encoding(text, limit)) == text


@pytest.mark.parametrize("limit", [1, 2, 7, 16, 64, 257, 4096])
def test_every_cut_lands_where_a_pretoken_ends(stress_documents, limit):
    """The rule the identity rests on: a piece begins at a whitespace character
    whose predecessor is not whitespace. Nothing in the byte-level
    pretokenizer's pattern matches across such a position, and BPE never merges
    across a pretoken — so the pieces encode to what the whole encodes to."""
    for text in stress_documents:
        pieces = bpe.split_for_encoding(text, limit)
        at = 0
        for piece in pieces[:-1]:
            at += len(piece)
            assert text[at] in " \t\n\r\f\v"
            assert not text[at - 1].isspace()


@pytest.mark.parametrize("limit", [1, 7, 64, 4096])
def test_cut_pieces_encode_to_the_whole_document(tokenizer, stress_documents, limit):
    """The claim itself, against a real vocabulary rather than against the
    pattern it is reasoned from."""
    for text in stress_documents:
        pieces = bpe.split_for_encoding(text, limit)
        rejoined = [tid for ids in bpe.encode_batch(tokenizer, pieces) for tid in ids]

        assert rejoined == bpe.encode(tokenizer, text)


def test_a_document_with_nowhere_safe_to_cut_is_left_whole(tokenizer):
    """A megabyte of base64 has no whitespace to cut at, and a cut anywhere else
    is not a boundary. It keeps the cost it already had rather than being made
    fast and wrong."""
    blob = "abcdefghijklmnop" * 4_000

    assert bpe.split_for_encoding(blob, 64) == [blob]
    # Cut where it can be and whole where it cannot: the last piece is the blob,
    # and what precedes it is the prose that did offer somewhere to cut.
    pieces = bpe.split_for_encoding("nowhere safe " + blob, 4)
    assert pieces[-1] == " " + blob
    assert "".join(pieces) == "nowhere safe " + blob


# ── the batch ───────────────────────────────────────────────────────────────


def _sizes(documents, limit, chunk) -> list[int]:
    return [
        sum(len(piece.text) for piece in batch)
        for batch in pack._batches(iter(documents), limit=limit, chunk=chunk)
    ]


def test_a_batch_is_bounded_by_characters_and_not_by_the_biggest_document():
    """The regression that put an 11 GiB machine into reclaim: a batch filled
    with whole documents is `ENCODE_BATCH_CHARS` *plus the last one*, and the
    last one is up to 27.6 MB on the corpus this was measured on."""
    documents = [
        pack.SourceDoc(dataset="d", text="a word here " * 5_000),  # 60 KB
        pack.SourceDoc(dataset="d", text="short."),
        pack.SourceDoc(dataset="d", text="another word " * 40_000),  # 520 KB
    ]

    sizes = _sizes(documents, limit=1_024, chunk=256)

    assert sizes, "a stream with text in it produces batches"
    # One piece of overshoot, which is what `chunk` bounds and `_Piece` is for.
    assert max(sizes) <= 1_024 + 256
    assert sum(sizes) == sum(len(one.text) for one in documents)


def test_a_document_may_span_batches_and_says_where_it_opens_and_closes():
    """What lets a batch be bounded at all. The writing side cannot record a
    span until the last piece of its document has been encoded, so the flags are
    the only thing that knows where one document ends and the next begins."""
    documents = [
        pack.SourceDoc(dataset="d", text="one word " * 200),
        pack.SourceDoc(dataset="d", text="two words here " * 200),
    ]

    pieces = [
        piece for batch in pack._batches(iter(documents), limit=64, chunk=32) for piece in batch
    ]

    assert sum(piece.opens for piece in pieces) == 2
    assert sum(piece.closes for piece in pieces) == 2
    for source, text in zip(documents, ("one word ", "two words here ")):
        mine = [piece for piece in pieces if piece.source is source]
        assert mine[0].opens and mine[-1].closes
        assert "".join(piece.text for piece in mine) == text * 200


def test_an_empty_document_is_still_one_piece_that_opens_and_closes():
    """`chars` counts what was read, so a document that encodes to nothing has
    to reach the counters — which it does by being a piece like any other."""
    batches = list(pack._batches(iter([pack.SourceDoc(dataset="d", text="")])))

    assert [(piece.text, piece.opens, piece.closes) for batch in batches for piece in batch] == [
        ("", True, True)
    ]


# ── what comes out of the pack ──────────────────────────────────────────────


def _pack(tokenizer, documents, key: str) -> tuple[bytes, np.ndarray, pack.PackStats]:
    stats = pack.pack(
        iter(documents),
        tokenizer,
        key=key,
        stages=[{"name": None, "epochs": 1, "datasets": []}],
        tokenizer_id="tiny",
        fingerprint="tiny",
    )
    corpus = storage.pretrain_corpus_path(key).read_bytes()
    rows = spans_mod.Spans.load(storage.pretrain_spans_path(key), []).rows
    return corpus, rows, stats


def test_the_corpus_does_not_depend_on_where_the_batches_fell(
    data_root, tokenizer, monkeypatch
):
    """The whole of the risk in cutting documents up and reading ahead of the
    encoder: batch and piece sizes are a scheduling detail, and a corpus that
    moved with them would be a corpus nobody could reproduce.

    Sized down rather than fed a corpus scaled to the real constants — 4 MiB
    batches of 64 KiB pieces would need tens of MB of fixture to cross a single
    boundary."""
    documents = [
        pack.SourceDoc(dataset="a", text=STRESS, stage=0),
        pack.SourceDoc(dataset="a", text="", stage=0),
        pack.SourceDoc(dataset="b", text=STRESS[:997], stage=1),
        pack.SourceDoc(dataset="b", text="the cat sat on the mat.", stage=1),
    ]

    whole, whole_rows, whole_stats = _pack(tokenizer, documents, "chunking-whole")

    monkeypatch.setattr(pack, "ENCODE_BATCH_CHARS", 97)
    monkeypatch.setattr(bpe, "CHUNK_CHARS", 31)
    cut, cut_rows, cut_stats = _pack(tokenizer, documents, "chunking-cut")

    assert cut == whole
    assert np.array_equal(cut_rows, whole_rows)
    assert (cut_stats.corpus_len, cut_stats.documents) == (
        whole_stats.corpus_len,
        whole_stats.documents,
    )
    assert cut_stats.chars == whole_stats.chars
    assert cut_stats.fertility == whole_stats.fertility
    assert cut_stats.special_ids_seen == whole_stats.special_ids_seen


def test_a_document_that_encodes_to_nothing_gets_no_span_and_no_separator(
    data_root, tokenizer
):
    """It is read — `chars` counts what was read — and it is not written. The
    empty check used to sit on a whole document's ids and now has to survive a
    document arriving in several pieces."""
    _, rows, stats = _pack(
        tokenizer,
        [
            pack.SourceDoc(dataset="a", text=""),
            pack.SourceDoc(dataset="a", text="the cat sat on the mat."),
        ],
        "chunking-empty",
    )

    assert len(rows) == 1
    assert stats.documents == 1
    assert stats.corpus_len == int(rows[0]["length"]) + 1  # the one separator


# ── reading ahead ───────────────────────────────────────────────────────────


def test_reading_ahead_hands_over_exactly_what_it_was_given():
    """A thread is an optimization, and an optimization that reordered or
    dropped a batch would reorder the corpus."""
    batches = [[object()], [object(), object()], [object()]]

    assert list(pack._read_ahead(iter(batches))) == batches


def test_a_reader_that_fails_fails_the_pack():
    """A dropped page belongs to the pack that asked for it. Swallowed on a
    thread nobody watches, it would truncate the corpus and report success."""

    def breaks():
        yield [object()]
        raise pack.remote.RemoteError("the backend went away")

    read = []
    with pytest.raises(pack.remote.RemoteError, match="went away"):
        for batch in pack._read_ahead(breaks()):
            read.append(batch)

    assert len(read) == 1, "what arrived before the failure still arrived"


def test_a_consumer_that_stops_early_does_not_strand_the_reader():
    """The reader blocks on a full queue, so something has to tell it that
    nobody is going to drain it — otherwise a failed pack leaks a thread and the
    connection it is holding for the life of the process."""
    import threading

    served = 0

    def endless():
        nonlocal served
        while True:
            served += 1
            yield [object()]

    before = threading.active_count()
    for _ in pack._read_ahead(endless()):
        break

    assert threading.active_count() <= before
    assert served < 1_000, "the reader stopped rather than racing ahead"
