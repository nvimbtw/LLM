"""Required real PostgreSQL training evidence, registration and post-rename rollback checks."""

import pytest
import store
from uuid import UUID
from llm.architecture.spec import canonical_digest
from store import architecture_queries as catalog, queries as q
from store.schema import Run, RunArchitecturePin, RunKind, RunStatus
from sqlalchemy.exc import OperationalError
from concurrent.futures import ThreadPoolExecutor

from tests.test_architecture_store import sessions, sample
from tests.test_training_publications import published_job
from tests.test_training_publications import (
    test_parent_adopts_registers_retains_and_replays_complete_generations as _verify_lifecycle,
    test_parent_index_failure_preserves_adoptable_package_and_rolls_back_definition_evidence as _verify_rollback,
)

pytestmark = pytest.mark.postgres


@pytest.fixture(autouse=True)
def parent_database(sessions, data_root, monkeypatch):
    """Bind parent registration to this owned test database and isolated artifacts, never application storage."""
    monkeypatch.setattr(store, "_sessions", sessions)


def test_postgres_publication_definition_retention_and_history_pruning(published_job, tmp_path, monkeypatch):
    """Exercise complete package/model/evidence registration and pruning against actual migrated PostgreSQL constraints."""
    _verify_lifecycle(published_job, tmp_path, monkeypatch)


def test_postgres_failed_registration_preserves_package_and_rolls_back_evidence(published_job, tmp_path, monkeypatch):
    """Verify that real PostgreSQL rolls back definition/evidence writes while the complete orphan remains adoptable."""
    _verify_rollback(published_job, tmp_path, monkeypatch)


def test_postgres_saved_input_pin_survives_real_history_cascade_and_root_archive(published_job, tmp_path):
    """Retained definitions supply a new immutable saved-input pin after real PostgreSQL history cascade and catalog archive."""
    from api import training_publications
    from tests.test_training_publications import granted

    checked = published_job.write(tmp_path / "output", final=True)
    notice, artifacts = granted(checked, tmp_path)
    training_publications.adopt(notice, artifacts, job_id=published_job.job_id)
    with store.session() as db:
        db.get(Run, published_job.job_id).status = RunStatus.done
        q.delete_run(db, published_job.job_id)
    with store.session() as db:
        assert db.get(RunArchitecturePin, published_job.job_id) is None
        root = catalog.get_resource(db, "architecture", published_job.definition.architecture.resource_id)
        root.archived = True
    with store.session() as db:
        definition_id = canonical_digest(published_job.definition)
        prepared = catalog.prepare_saved_training_pin(db, definition_id)
        run = q.enqueue(db, kind=RunKind.train, label="saved source", config={"resume_from": published_job.config.model_name})
        pin = catalog.attach_training_pin(db, run, prepared)
        assert pin.parameters == published_job.definition.parameters
        assert prepared.evidence.tokenizer == published_job.definition.tokenizer
        assert q.get_model(db, UUID(published_job.config.model_name)).definition_id == definition_id


def test_postgres_publication_has_a_finite_lock_wait(published_job, tmp_path, sessions):
    """A conflicting actual row lock refuses adoption within the store timeout and cannot authorize another generation."""
    from api import training_publications
    from tests.test_training_publications import granted

    checked = published_job.write(tmp_path / "output", final=True)
    notice, artifacts = granted(checked, tmp_path)

    def adopt():
        """Attempt the actual bounded parent adoption from an independent transaction/thread."""
        return training_publications.adopt(notice, artifacts, job_id=published_job.job_id)

    with sessions.begin() as holder:
        q.lock_run(holder, published_job.job_id)
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(adopt)
            with pytest.raises(OperationalError, match="lock timeout"):
                future.result(timeout=15)
