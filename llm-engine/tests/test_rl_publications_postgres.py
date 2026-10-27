"""Required PostgreSQL RL publication, saved-definition retention and finite lock-wait contracts; synthetic bytes do not certify numerics."""

from concurrent.futures import ThreadPoolExecutor
from uuid import UUID, uuid4

import pytest
from sqlalchemy.exc import OperationalError

import store
from api import rl_handler, training_publications
from store import architecture_queries as catalog, queries as q
from store.schema import Model, Run, RunArchitecturePin, RunKind, RunStatus
from tests.test_architecture_store import sessions, sample
from tests.test_training_publications import granted, published_job
from tests.test_rl_handler import rl_job, rl_output
from tests.test_rl_handler import test_rl_registration_failure_retains_an_adoptable_package_without_false_job_success as _verify_rollback

pytestmark = pytest.mark.postgres


@pytest.fixture(autouse=True)
def parent_database(sessions, data_root, monkeypatch):
    """Bind the real parent transactions to this uniquely owned migrated PostgreSQL database and isolated artifact tree."""
    monkeypatch.setattr(store, "_sessions", sessions)


def test_postgres_rl_failed_registration_preserves_package_for_reconciliation(rl_job, tmp_path, monkeypatch):
    """Real PostgreSQL rolls back RL evidence/model writes, while complete orphan bytes remain adoptable without terminal success."""
    _verify_rollback(rl_job, tmp_path, monkeypatch)


def test_postgres_rl_saved_recovery_survives_history_cascade_and_root_archive(rl_job, tmp_path):
    """A complete RL recovery model retains exact policy/original-reference authority after actual job cascades and catalog archive."""
    job, actor = rl_job
    checked = rl_output(job, tmp_path / "worker")
    notice, artifacts = granted(checked, tmp_path)
    training_publications.adopt(notice, artifacts, job_id=job.job_id)
    source_id = training_publications.metadata(checked).generation.model_id
    with store.session() as db:
        db.get(Run, job.job_id).status = RunStatus.failed
        q.delete_run(db, job.job_id)
    with store.session() as db:
        assert db.get(RunArchitecturePin, job.job_id) is None
        catalog.get_resource(db, "architecture", job.definition.architecture.resource_id).archived = True
        pin = catalog.prepare_saved_training_pin(db, checked.manifest.definition_id)
        cfg = job.config.model_copy(update={"model_name": str(uuid4()), "resume_from": str(source_id)})
        run = q.enqueue(db, kind=RunKind.rl, label="RL saved recovery", owner_id=actor,
            config=cfg.model_dump(mode="json", exclude_unset=True))
        catalog.attach_training_pin(db, run, pin)
        assert db.get(Model, UUID(cfg.base_model)).definition_id == checked.manifest.definition_id
    source, original, step = rl_handler.source_packages(cfg, pin)
    assert source.manifest.definition == original.manifest.definition == job.definition and step == 1


def test_postgres_rl_adoption_has_a_finite_lock_wait(rl_job, tmp_path, sessions):
    """A conflicting actual RL execution row lock times out the parent adopter and cannot acknowledge a new generation."""
    job, _actor = rl_job
    checked = rl_output(job, tmp_path / "worker", final=True)
    notice, artifacts = granted(checked, tmp_path)

    def adopt():
        """Attempt one real bounded RL adoption from an independent transaction and thread."""
        return training_publications.adopt(notice, artifacts, job_id=job.job_id)

    with sessions.begin() as holder:
        q.lock_run(holder, job.job_id)
        with ThreadPoolExecutor(max_workers=1) as executor:
            with pytest.raises(OperationalError, match="lock timeout"):
                executor.submit(adopt).result(timeout=15)
