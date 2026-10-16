"""Submission hashes and executed hashes remain distinguishable after a restart."""

import threading

import pytest

import processors
import store
from api.handlers import process
from processors.loader import Catalog
from store import queries as q
from store.schema import DatasetKind, DatasetOrigin, DatasetStatus
from tests.test_processor_loader import SOURCE


def test_worker_records_drift_and_runs_current_version(
    tmp_path, monkeypatch, pretrain_dataset, owner, caplog
):
    """A fresh catalogue executes current bytes and persists both version identities."""
    path = tmp_path / "example.py"
    path.write_text(SOURCE)
    submitted = Catalog({}, tmp_path)["example"].version_hash
    config_steps = [{"processor": "example", "params": {}, "version_hash": submitted}]
    with store.session() as db:
        target = q.create_dataset(
            db,
            name="version-drift",
            kind=DatasetKind.pretrain,
            origin=DatasetOrigin.derived,
            owner_id=owner,
            steps=config_steps,
        )
        target_id = target.id
    path.write_text(SOURCE.replace("text.upper()", "text.lower()"))
    current = Catalog({}, tmp_path)
    monkeypatch.setattr(processors, "REGISTRY", current)
    result = process(
        {
            "source_id": str(pretrain_dataset),
            "target_id": str(target_id),
            "steps": config_steps,
        },
        lambda event: None,
        threading.Event(),
    )
    record = result["steps"][0]
    assert record["version_hash"] == submitted
    assert record["executed_version_hash"] == current["example"].version_hash
    assert record["version_mismatch"] is True
    assert "version drift" in caplog.text
    with store.session() as db:
        assert q.dataset(db, target_id).steps == result["steps"]
        assert "hello" in q.documents(db, target_id, offset=0, limit=1)[0].content


@pytest.mark.parametrize("removed", [True, False])
def test_unavailable_current_processor_fails_target_instead_of_leaving_it_pending(
    tmp_path, monkeypatch, pretrain_dataset, owner, removed
):
    """A removed or newly incompatible processor fails a queued target before any writes."""
    path = tmp_path / "example.py"
    path.write_text(SOURCE)
    submitted = Catalog({}, tmp_path)["example"].version_hash
    with store.session() as db:
        target = q.create_dataset(
            db,
            name="unavailable-version",
            kind=DatasetKind.pretrain,
            origin=DatasetOrigin.derived,
            owner_id=owner,
        )
        target_id = target.id
    if removed:
        path.unlink()
    else:
        path.write_text(SOURCE.replace('["pretrain", "sft"]', '["sft"]'))
    monkeypatch.setattr(processors, "REGISTRY", Catalog({}, tmp_path))
    with pytest.raises((KeyError, ValueError)):
        process(
            {
                "source_id": str(pretrain_dataset),
                "target_id": str(target_id),
                "steps": [{"processor": "example", "version_hash": submitted}],
            },
            lambda event: None,
            threading.Event(),
        )
    with store.session() as db:
        assert q.dataset(db, target_id).status == DatasetStatus.failed
        assert q.dataset(db, target_id).document_count == 0
