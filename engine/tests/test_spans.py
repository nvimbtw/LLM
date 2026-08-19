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

from llm import storage
from llm.corpus import pack
from llm.corpus.spans import BUILD_BLOCK, DocSpan, SpanBuilder, Spans, from_sidecar


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
