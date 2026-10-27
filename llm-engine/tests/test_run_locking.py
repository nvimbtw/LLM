"""PostgreSQL row-lock compatibility with eager nullable execution-domain joins."""

from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import postgresql
from sqlalchemy.exc import OperationalError

from store import queries as q
from store.schema import Run, RunKind, RunStatus
from tests.test_architecture_store import sessions


def test_job_lock_sql_targets_only_execution_owner_and_keeps_domain_reads():
    """Compile the actual helper's query for PostgreSQL without pretending to exercise a live database."""
    statements = []

    class StatementRecorder:
        """Capture the store boundary's real SQL statement without replacing compilation."""

        def scalar(self, statement):
            """Retain one requested query and return the absent-row result expected by the caller."""
            statements.append(statement)
            return None

    assert q.lock_run(StatementRecorder(), uuid4()) is None
    assert len(statements) == 1
    sql = str(statements[0].compile(dialect=postgresql.dialect()))
    assert sql.endswith("FOR UPDATE OF execution_jobs")
    assert "LEFT OUTER JOIN training_runs" in sql
    assert "LEFT OUTER JOIN model_validations" in sql
    assert statements[0].get_execution_options()["populate_existing"] is True


@pytest.mark.postgres
@pytest.mark.parametrize("kind", list(RunKind))
def test_postgres_locks_each_job_kind_with_nullable_domains(sessions, kind):
    """Real PostgreSQL must load and transition every kind despite its absent unrelated domain rows."""
    with sessions.begin() as db:
        job_id = q.enqueue(db, kind=kind, label="row lock regression", config={"fixture": kind.value}).id
    with sessions.begin() as db:
        run = q.lock_run(db, job_id)
        assert run.kind == kind and run.config == {"fixture": kind.value}
        run.status = RunStatus.cancelling
    with sessions() as db:
        assert db.get(Run, job_id).status == RunStatus.cancelling


@pytest.mark.postgres
def test_postgres_job_lock_serializes_same_job_without_blocking_other_jobs(sessions):
    """Independent transactions contend on one durable job while another job remains lockable."""
    with sessions.begin() as db:
        first = q.enqueue(db, kind=RunKind.train, label="held", config={}).id
        second = q.enqueue(db, kind=RunKind.model_validation, label="independent", config={}).id
    with sessions.begin() as holder:
        assert q.lock_run(holder, first).id == first
        with sessions.begin() as independent:
            independent.execute(text("SET LOCAL lock_timeout = '200ms'"))
            assert q.lock_run(independent, second).id == second
        with pytest.raises(OperationalError, match="lock timeout"):
            with sessions.begin() as contender:
                contender.execute(text("SET LOCAL lock_timeout = '200ms'"))
                q.lock_run(contender, first)
