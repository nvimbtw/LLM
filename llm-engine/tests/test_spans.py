"""The document table: what it costs, and that moving it off JSON changed
nothing about what a run reads.

The table is the one structure in a build that scales with the *number* of
documents rather than with their size, which is why it is columnar — see
`llm.corpus.spans`. These pin the cost, and pin the two shapes of sidecar
against each other.
"""

import json
import tracemalloc

import numpy as np
import pytest

from llm import storage
from llm.corpus import pack
from llm.corpus.spans import (
    BUILD_BLOCK,
    LENGTH_HIST_BINS,
    MEASURE_BLOCK,
    RIBBON_SEGMENTS,
    RIBBON_WINDOWS,
    SPAN_DTYPE,
    DocSpan,
    SpanBuilder,
    Spans,
    from_sidecar,
    span_ribbon,
    window_fit,
)


def _build(count: int, stages: int = 1, names: list[str] | None = None) -> Spans:
    names = names or ["set0", "set1", "set2"]
    builder = SpanBuilder()
    for i in range(count):
        builder.append(names[i % 3], start=i * 10, length=9, stage=i % stages)
    return builder.build()


def test_a_row_costs_twenty_two_bytes():
    """The whole reason the table is columnar. A `DocSpan` per row is a few
    hundred bytes of interpreter overhead each; at one row per document rather
    than one per shard, that difference is the machine."""
    spans = _build(1000)

    assert spans.rows.nbytes / len(spans) == 22


def _resident(make) -> tuple[float, object]:
    """Bytes still held once `make()` returns, and what it returned. numpy
    reports into tracemalloc's own domain, so the rows are counted alongside the
    objects around them."""
    tracemalloc.start()
    try:
        held = make()
        current, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return current, held


def test_a_document_costs_a_row_and_not_an_object():
    """The regression the columnar table exists to prevent.

    A corpus of files has one span per shard, and at ~10k spans either shape
    would do. A corpus of documents has hundreds of thousands of spans for the
    same tokens, and then the per-row constant is the whole difference — the
    table, the JSON encoding of it, and the per-epoch copy of it in `EpochStream`
    were together enough to lose the machine.

    Names are shared between rows in both shapes, as they are in a real build,
    so what is compared is the row and not the string it points at.
    """
    count = BUILD_BLOCK * 4
    names = ["a", "b", "c"]

    columnar, _table = _resident(lambda: _build(count, names=names))
    objects, _list = _resident(
        lambda: [DocSpan(names[i % 3], i * 10, 9) for i in range(count)]
    )

    # 22 bytes of row plus the name table, against ~145 for the object.
    assert columnar / count < 32
    assert columnar * 4 < objects


def test_the_table_reads_back_as_the_documents_that_went_in():
    """Iteration is what `explore/` and `bench.py` use, and it must survive the
    round trip through disk unchanged."""
    spans = _build(50, stages=2)
    path = storage.datasets_dir() / "spans-roundtrip.npy"
    storage.ensure_dir(path.parent)
    spans.save(path)

    loaded = Spans.load(path, spans.names)

    assert len(loaded) == 50
    assert list(loaded) == list(spans)
    assert loaded[7] == DocSpan(dataset="set1", start=70, length=9, stage=1)


def test_a_stage_selects_its_own_documents_and_keeps_their_names():
    spans = _build(30, stages=3)

    stage = spans.in_stage(2)

    assert len(stage) == 10
    assert {d.stage for d in stage} == {2}
    assert [d.dataset for d in stage] == [f"set{(i * 3 + 2) % 3}" for i in range(10)]


def test_a_sidecar_written_before_the_table_moved_out_still_loads(tmp_path):
    """A corpus packed the old way carries its rows inline in `segments.json`.
    Those builds are hours of packing apiece — they are read, not invalidated."""
    legacy = tmp_path / "segments.json"
    legacy.write_text(
        json.dumps(
            {
                "eot_id": 3,
                "documents": [
                    {"dataset": "TinyStories", "start": 0, "length": 12, "stage": 0},
                    {"dataset": "Wikipedia", "start": 13, "length": 40, "stage": 1},
                ],
            }
        )
    )

    spans = from_sidecar(tmp_path / "spans.npy", json.loads(legacy.read_text()))

    assert len(spans) == 2
    assert list(spans) == [
        DocSpan(dataset="TinyStories", start=0, length=12, stage=0),
        DocSpan(dataset="Wikipedia", start=13, length=40, stage=1),
    ]
    assert list(spans.starts) == [0, 13]


def test_a_missing_sidecar_is_an_empty_table_not_a_failure():
    """`DataPlan.whole_corpus` is the fallback, and it is reached by the caller
    seeing nothing here rather than by an exception."""
    assert len(from_sidecar(storage.datasets_dir() / "absent.npy", {})) == 0
    assert not Spans.empty()


def test_a_packed_corpus_writes_its_table_beside_the_corpus(packed):
    """The sidecar keeps the provenance and the name table; the rows are next to
    the tokens they index."""
    key, stats = packed

    assert storage.pretrain_spans_path(key).is_file()
    raw = json.loads(storage.pretrain_segments_path(key).read_text())
    assert raw["documents"] == stats.documents
    assert raw["datasets"]

    _, spans, _ = pack.load_segments(key)
    assert len(spans) == stats.documents
    assert spans.names == raw["datasets"]
    # Every document lies inside the corpus it claims to index.
    corpus = pack.load_corpus(key)
    assert int(np.max(spans.starts + spans.lengths)) <= len(corpus)


def test_window_fit_counts_the_windows_that_cross_a_document():
    """The straddle is what packing quality actually is: how often a training
    window spans two documents. A separator landing in a window's last slot is
    not one of them — nothing after it is inside the window."""
    aligned = Spans.of([DocSpan("a", start=i * 4, length=3) for i in range(4)])

    fit = window_fit(aligned, ctx_window=4, corpus_len=16)

    # Every document ends exactly where its window does.
    assert fit.straddle == 0.0
    assert fit.over_ctx == 0.0
    assert len(fit.length_hist) == LENGTH_HIST_BINS
    assert sum(fit.length_hist) == pytest.approx(1.0)

    # Boundaries at 2 and 8 fall inside windows 0 and 2; the one at 19 is past
    # the last full window, which `iter_batches` drops anyway.
    ragged = Spans.of([DocSpan("a", 0, 2), DocSpan("a", 3, 5), DocSpan("a", 9, 10)])

    ragged_fit = window_fit(ragged, ctx_window=4, corpus_len=20)

    assert ragged_fit.straddle == 0.5
    assert ragged_fit.over_ctx == pytest.approx(2 / 3)


def test_a_corpus_with_no_windows_measures_nothing():
    """A fraction of zero windows is not zero, it is unmeasured — and the panel
    shows "—" for the difference."""
    assert window_fit(_build(3), ctx_window=1024, corpus_len=30) is None
    assert window_fit(Spans.empty(), ctx_window=4, corpus_len=1000) is None
    # The ribbon answers on exactly the same terms: nothing to draw is None
    # rather than an empty strip, which would read as a corpus of no documents.
    assert span_ribbon(_build(3), ctx_window=1024, corpus_len=30) is None
    assert span_ribbon(Spans.empty(), ctx_window=4, corpus_len=1000) is None
    assert span_ribbon(_build(3), ctx_window=0, corpus_len=30) is None


def _placed(window) -> list[tuple[int, int, bool, bool]]:
    """One window's segments as plain tuples, for comparing against a layout
    worked out by hand."""
    return [(s.start, s.length, s.opens, s.closes) for s in window.segments]


def test_the_ribbon_lays_a_window_out_as_the_documents_inside_it():
    """The picture behind the straddle, on a corpus small enough to check by eye.

    Four documents of three tokens, each closed by its separator, so every
    document fills its own four-token window exactly.
    """
    aligned = Spans.of([DocSpan("a", start=i * 4, length=3) for i in range(4)])

    ribbon = span_ribbon(aligned, ctx_window=4, corpus_len=16)

    assert ribbon.datasets == ["a"]
    # (16 - 1) // 4 windows, all of them sampled because there are fewer than
    # `RIBBON_WINDOWS`.
    assert [w.index for w in ribbon.windows] == [0, 1, 2]
    for window in ribbon.windows:
        assert _placed(window) == [(0, 3, True, True)]
        assert window.overflow == 0


def test_a_document_wider_than_a_window_reads_as_one_unbroken_bar():
    """The books case, and the reason `opens`/`closes` are on the wire.

    Three documents at 0-2, 3-8 and 9-19 against four-token windows. The middle
    one covers window 1 whole and the last covers window 3 whole; drawing either
    without knowing the document continues past the edge would show a reader nine
    unrelated documents where the corpus has one.
    """
    ragged = Spans.of([DocSpan("a", 0, 2), DocSpan("a", 3, 5), DocSpan("a", 9, 10)])

    ribbon = span_ribbon(ragged, ctx_window=4, corpus_len=20)

    assert [w.index for w in ribbon.windows] == [0, 1, 2, 3]
    # The window the second document opens in: its own two tokens, then one token
    # of a document that runs past the edge.
    assert _placed(ribbon.windows[0]) == [(0, 2, True, True), (3, 1, True, False)]
    # Filled by that document, which opened before this window and closes in it.
    assert _placed(ribbon.windows[1]) == [(0, 4, False, True)]
    assert _placed(ribbon.windows[2]) == [(1, 3, True, False)]
    # Neither end is a document boundary: the window is entirely mid-document.
    assert _placed(ribbon.windows[3]) == [(0, 4, False, False)]


def _dense(count: int, length: int, names: list[str]) -> Spans:
    """`count` documents of `length` tokens, one separator apart, laid down
    dataset by dataset the way a build packs them.

    Built straight into the array rather than through `SpanBuilder`, whose
    per-row Python loop would dominate a test written at the scale these
    measurements exist for.
    """
    rows = np.empty(count, dtype=SPAN_DTYPE)
    rows["start"] = np.arange(count, dtype=np.int64) * (length + 1)
    rows["length"] = length
    # Blocked, not interleaved — `orchestrate.train._documents` yields one
    # dataset to exhaustion before the next.
    rows["dataset"] = np.arange(count) * len(names) // count
    rows["stage"] = 0
    return Spans(names, rows)


def test_the_frame_is_bounded_by_the_constants_and_not_by_the_corpus():
    """The whole reason `RIBBON_WINDOWS` and `RIBBON_SEGMENTS` are constants.

    Two full `MEASURE_BLOCK`s of rows and change, which is also what exercises
    the blocked read: a window whose documents fall either side of a block
    boundary has to come out whole.
    """
    length = 3
    count = MEASURE_BLOCK * 2 + 7
    spans = _dense(count, length, ["books", "wiki"])
    corpus_len = count * (length + 1)

    ribbon = span_ribbon(spans, ctx_window=512, corpus_len=corpus_len)

    assert len(ribbon.windows) == RIBBON_WINDOWS
    assert sum(len(w.segments) for w in ribbon.windows) <= RIBBON_WINDOWS * RIBBON_SEGMENTS
    for window in ribbon.windows:
        assert len(window.segments) == RIBBON_SEGMENTS
        # 512 // 4 documents land in a window and 16 are drawn; the rest are
        # counted so the renderer can draw the tail as the texture it is.
        assert window.overflow == 512 // (length + 1) - RIBBON_SEGMENTS
        assert all(0 <= s.start < 512 for s in window.segments)
        assert all(0 < s.start + s.length <= 512 for s in window.segments)


def test_the_sample_is_spread_across_the_corpus_rather_than_taken_off_its_head():
    """A corpus is packed dataset by dataset, so its first windows are all one
    dataset. A head sample would report every multi-dataset build as homogeneous
    — not because it is, but because it is sorted."""
    count = 4000
    spans = _dense(count, 3, ["books", "wiki"])
    ribbon = span_ribbon(spans, ctx_window=512, corpus_len=count * 4)

    indices = [w.index for w in ribbon.windows]
    assert indices == sorted(set(indices))
    windows = (count * 4 - 1) // 512
    assert indices[0] == 0 and indices[-1] == windows - 1
    assert {s.dataset for w in ribbon.windows for s in w.segments} == {0, 1}

    # Deterministic: the run that packs a build and every later run that reuses
    # it report this frame, and two ribbons of one corpus that disagree read as
    # the corpus having changed when only the sample did.
    assert span_ribbon(spans, 512, count * 4).as_dict() == ribbon.as_dict()
