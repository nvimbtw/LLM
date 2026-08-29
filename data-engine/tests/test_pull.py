"""Pulling from the Hub, against a real parquet file and a stubbed Hub.

The network is the one thing not exercised here. Everything downstream of it —
shard listing, column resolution, row shaping, the row cap, batching, and what
the dataset looks like afterwards — runs for real.
"""

import uuid

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import hub
import store
from api import handlers
from store import queries as q
from store.schema import DatasetKind, DatasetOrigin, DatasetStatus


@pytest.fixture
def shard(tmp_path):
    """A parquet shard with a text column and a stray metadata column."""
    table = pa.table(
        {
            "text": [f"document number {i} " + "padding " * 5 for i in range(50)],
            "url": [f"https://example.invalid/{i}" for i in range(50)],
            "score": [float(i) for i in range(50)],
        }
    )
    path = tmp_path / "train-00000.parquet"
    pq.write_table(table, path)
    return path


@pytest.fixture
def stub_hub(monkeypatch, shard):
    """Point the Hub calls at a local file, leaving the parquet reader real."""
    monkeypatch.setattr(hub.fetch, "hf_hub_download", lambda *a, **k: str(shard))

    class Api:
        def list_repo_files(self, repo, repo_type=None):
            return ["README.md", "data/train-00000.parquet"]

    monkeypatch.setattr("huggingface_hub.HfApi", Api)


@pytest.fixture
def dataset(store_session):
    with store.session() as db:
        row = q.create_dataset(
            db,
            name="pulled",
            kind=DatasetKind.pretrain,
            origin=DatasetOrigin.huggingface,
            owner_id=uuid.uuid4(),
            source_repo="fixture/corpus",
        )
        return row.id


def run_pull(dataset_id, **config):
    import threading

    reports = []
    result = handlers.pull(
        {"dataset_id": str(dataset_id), "repo": "fixture/corpus", **config},
        reports.append,
        threading.Event(),
    )
    return result, reports


def test_a_pull_respects_its_row_cap(stub_hub, dataset):
    result, _ = run_pull(dataset, limit_rows=12)

    assert result["documents"] == 12
    with store.session() as db:
        assert q.dataset(db, dataset).document_count == 12


def test_the_cap_is_exact_when_the_source_outruns_a_batch(stub_hub, dataset, monkeypatch):
    """The cap bounds the intake, not the batch that crosses it.

    Checking only at batch boundaries rounds every request up to `WRITE_BATCH`,
    which is invisible whenever the fixture is smaller than one batch — and is
    exactly how asking for 40 documents returned 200.
    """
    monkeypatch.setattr(handlers, "WRITE_BATCH", 10)

    result, _ = run_pull(dataset, limit_rows=13)

    assert result["documents"] == 13
    with store.session() as db:
        rows = q.documents(db, dataset, offset=0, limit=100)
    assert [r.ordinal for r in rows] == list(range(13))


def test_a_pull_publishes_the_dataset_and_counts_its_bytes(stub_hub, dataset):
    run_pull(dataset, limit_rows=5)

    with store.session() as db:
        row = q.dataset(db, dataset)
        assert row.status == DatasetStatus.ready
        assert row.byte_size > 0
        assert row.document_count == 5


def test_documents_are_numbered_from_zero_without_gaps(stub_hub, dataset):
    run_pull(dataset, limit_rows=7)

    with store.session() as db:
        rows = q.documents(db, dataset, offset=0, limit=100)

    assert [r.ordinal for r in rows] == list(range(7))
    assert all(r.content and r.messages is None for r in rows)


def test_the_text_column_is_found_without_being_named(stub_hub, dataset):
    run_pull(dataset, limit_rows=3)

    with store.session() as db:
        first = q.documents(db, dataset, offset=0, limit=1)[0]

    assert first.content.startswith("document number 0")


def test_other_columns_are_kept_as_metadata_not_as_content(stub_hub, dataset):
    run_pull(dataset, limit_rows=3)

    with store.session() as db:
        first = q.documents(db, dataset, offset=0, limit=1)[0]

    assert first.meta["url"] == "https://example.invalid/0"
    assert "text" not in first.meta


def test_a_column_that_does_not_exist_fails_loudly(stub_hub, dataset):
    """A pull that quietly writes nothing is indistinguishable from a dataset
    that is genuinely empty, so it must not be quiet."""
    with pytest.raises(hub.HubError, match="no pretrain column"):
        run_pull(dataset, limit_rows=5, column="nonexistent")


def test_a_cancelled_pull_keeps_what_it_already_had(stub_hub, dataset, monkeypatch):
    import threading

    cancel = threading.Event()
    # Small enough that the cancel lands after the first commit, not before it.
    monkeypatch.setattr(handlers, "WRITE_BATCH", 5)

    def report(payload):
        if isinstance(payload, dict) and payload.get("documents", 0) >= 5:
            cancel.set()

    result = handlers.pull(
        {"dataset_id": str(dataset), "repo": "fixture/corpus", "limit_rows": 50},
        report,
        cancel,
    )

    assert result["cancelled"] is True
    assert 0 < result["documents"] < 50
    with store.session() as db:
        assert q.dataset(db, dataset).document_count == result["documents"]


def test_a_repo_with_no_shards_is_an_error_not_an_empty_pull(monkeypatch, dataset):
    class Api:
        def list_repo_files(self, repo, repo_type=None):
            return ["README.md"]

    monkeypatch.setattr("huggingface_hub.HfApi", Api)

    with pytest.raises(hub.HubError, match="no parquet shards"):
        run_pull(dataset, limit_rows=5)


@pytest.fixture
def nul_shard(tmp_path, monkeypatch):
    """A shard carrying NUL bytes in both its text and its metadata column.

    Not hypothetical: Gutenberg dumps are full of them, and every document in a
    batch containing one is refused by Postgres, not just the offending row.
    """
    table = pa.table(
        {
            "text": ["clean text here", "text with a \x00 in it", "another \x00\x00 one"],
            "url": ["https://example.invalid/a", "https://example.invalid/\x00b", ""],
        }
    )
    path = tmp_path / "train-00000.parquet"
    pq.write_table(table, path)
    monkeypatch.setattr(hub.fetch, "hf_hub_download", lambda *a, **k: str(path))

    class Api:
        def list_repo_files(self, repo, repo_type=None):
            return ["data/train-00000.parquet"]

    monkeypatch.setattr("huggingface_hub.HfApi", Api)


def test_nul_bytes_are_stripped_from_content(nul_shard, dataset):
    """SQLite would store these happily; Postgres cannot represent them at all.

    So this asserts the invariant rather than reproducing the `DataError` — the
    dialect that raises is the one the test database is not.
    """
    run_pull(dataset, limit_rows=3)

    with store.session() as db:
        rows = q.documents(db, dataset, offset=0, limit=10)

    assert [r.content for r in rows] == [
        "clean text here",
        "text with a  in it",
        "another  one",
    ]


def test_nul_bytes_are_stripped_from_metadata_too(nul_shard, dataset):
    """A `jsonb` column refuses them exactly as a `text` one does, so a corpus
    with a clean body and a dirty metadata column must not fail either."""
    run_pull(dataset, limit_rows=3)

    with store.session() as db:
        rows = q.documents(db, dataset, offset=0, limit=10)

    assert rows[1].meta["url"] == "https://example.invalid/b"


def test_a_stripped_document_reports_the_size_it_was_stored_at(nul_shard, dataset):
    run_pull(dataset, limit_rows=3)

    with store.session() as db:
        rows = q.documents(db, dataset, offset=0, limit=10)

    assert all(r.byte_size == len(r.content.encode("utf-8")) for r in rows)


def test_a_pull_says_how_many_documents_it_rewrote(nul_shard, dataset):
    """Silently changing what the source said is the part worth surfacing; two of
    these three rows were altered on the way in."""
    result, _ = run_pull(dataset, limit_rows=3)

    assert result["nul_stripped"] == 2


def test_a_clean_pull_rewrites_nothing(stub_hub, dataset):
    result, _ = run_pull(dataset, limit_rows=5)

    assert result["nul_stripped"] == 0


def test_conversations_are_shaped_from_a_messages_column(monkeypatch, tmp_path, store_session):
    table = pa.table(
        {
            "conversations": [
                [{"from": "human", "value": "hi"}, {"from": "gpt", "value": "hello"}]
            ]
            * 4
        }
    )
    path = tmp_path / "sft-00000.parquet"
    pq.write_table(table, path)
    monkeypatch.setattr(hub.fetch, "hf_hub_download", lambda *a, **k: str(path))

    class Api:
        def list_repo_files(self, repo, repo_type=None):
            return ["data/sft-00000.parquet"]

    monkeypatch.setattr("huggingface_hub.HfApi", Api)

    with store.session() as db:
        row = q.create_dataset(
            db, name="chat", kind=DatasetKind.sft,
            origin=DatasetOrigin.huggingface, owner_id=None,
        )
        dataset_id = row.id

    run_pull(dataset_id, kind="sft", limit_rows=4)

    with store.session() as db:
        first = q.documents(db, dataset_id, offset=0, limit=1)[0]

    assert first.content is None
    # Roles arrive as the source wrote them; mapping them is a processor's job.
    assert [m["role"] for m in first.messages] == ["human", "gpt"]


def test_nul_bytes_are_stripped_from_inside_a_turn(monkeypatch, tmp_path, store_session):
    """The payload is a list of dicts here, so a scrub that only looked at the
    top level of the row would walk straight past this one."""
    table = pa.table(
        {"conversations": [[{"from": "human", "value": "hi\x00 there"}]] * 2}
    )
    path = tmp_path / "sft-00000.parquet"
    pq.write_table(table, path)
    monkeypatch.setattr(hub.fetch, "hf_hub_download", lambda *a, **k: str(path))

    class Api:
        def list_repo_files(self, repo, repo_type=None):
            return ["data/sft-00000.parquet"]

    monkeypatch.setattr("huggingface_hub.HfApi", Api)

    with store.session() as db:
        row = q.create_dataset(
            db, name="chat", kind=DatasetKind.sft,
            origin=DatasetOrigin.huggingface, owner_id=None,
        )
        dataset_id = row.id

    result, _ = run_pull(dataset_id, kind="sft", limit_rows=2)

    with store.session() as db:
        first = q.documents(db, dataset_id, offset=0, limit=1)[0]

    assert first.messages[0]["content"] == "hi there"
    assert result["nul_stripped"] == 2
