"""Failure boundaries and retention; execution is deferred with the sprint suite."""
import uuid

import numpy as np
import pytest

from llm.orchestrate import recovery
from llm.train.plan import DataPlan
from store import queries as q
from store.schema import RunKind


def test_failed_publication_preserves_previous_generation(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery.storage, "recovery_staging_dir", lambda: tmp_path / "staging")
    previous = tmp_path / "models" / "previous"
    previous.mkdir(parents=True)
    (previous / "model.pt").write_bytes(b"original")
    destination = previous.parent / "next"
    with pytest.raises(OSError, match="interrupted"):
        with recovery.stage(destination) as staging:
            (staging / "model.pt").write_bytes(b"partial")
            raise OSError("interrupted")
    assert (previous / "model.pt").read_bytes() == b"original"
    assert not destination.exists()
    assert list((tmp_path / "staging").iterdir()) == []


def test_recovery_fingerprint_covers_content_and_data_order():
    tokens = np.arange(32, dtype=np.uint16)
    plan = DataPlan.whole_corpus(len(tokens))
    original = recovery.fingerprint(tokens, plan=plan)
    changed = tokens.copy()
    changed[12] = 30
    assert recovery.fingerprint(changed, plan=plan) != original
    plan.seed += 1
    assert recovery.fingerprint(tokens, plan=plan) != original


def test_retention_spares_published_kept_and_referenced_generations(store_session):
    parent = uuid.uuid4()
    with store_session() as db:
        rows = [q.upsert_model(db, model_id=uuid.uuid4(), name=f"generation {i}",
                              parent_id=parent, checkpoint_index=i, checkpoint_step=i * 10,
                              parameters={"result": {"recovery": True}})
                for i in range(1, 7)]
        rows[1].published = True
        rows[2].parameters = {"recovery_keep": True}
        rows[3].parameters = {"recovery_referenced": True}
        q.enqueue(db, kind=RunKind.train, label="recovery consumer",
                  config={"resume_from": str(rows[4].id)})
        assert q.retire_recovery(db, parent, rows[5].id) == [rows[0].id]
        assert {row.id for row in q.checkpoints_of(db, parent)} == {row.id for row in rows[1:]}
