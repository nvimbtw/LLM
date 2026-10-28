"""Production PostgreSQL migrations and ORM models must describe one schema."""

import os

import pytest
from sqlalchemy import create_engine

from tests.fixtures.postgres import database

pytestmark = pytest.mark.postgres


@pytest.fixture
def scratch_database():
    """Own one disposable PostgreSQL database without colliding with concurrent test runs."""
    import store

    previous_url = os.environ.get("LLM_ENGINE_DATABASE_URL")
    previous_sessions = store._sessions
    with database() as url:
        os.environ["LLM_ENGINE_DATABASE_URL"] = url
        try:
            yield url
        finally:
            if previous_url is None:
                os.environ.pop("LLM_ENGINE_DATABASE_URL", None)
            else:
                os.environ["LLM_ENGINE_DATABASE_URL"] = previous_url
            if store._sessions is not None and store._sessions is not previous_sessions:
                store._sessions.kw["bind"].dispose()
            store._sessions = previous_sessions


def test_migrations_reproduce_the_models(scratch_database):
    """Upgrade from empty, then ask autogenerate what is still missing. A clean
    answer means the two agree."""
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    config = Config("alembic.ini")
    command.upgrade(config, "head")

    from store.schema import Base

    engine = create_engine(scratch_database)
    try:
        with engine.connect() as connection:
            diff = compare_metadata(MigrationContext.configure(connection), Base.metadata)
    finally:
        engine.dispose()

    assert diff == [], f"schema.py and migrations disagree: {diff}"


def test_the_schema_round_trips_through_postgres(scratch_database):
    """The SQLite the other store tests use cannot prove the enums, the JSONB
    columns or the cascades survive the dialect they actually run on."""
    import uuid

    from alembic import command
    from alembic.config import Config

    import store
    from store import queries as q
    from store.schema import RunKind, TrainingRun

    command.upgrade(Config("alembic.ini"), "head")
    store.configure(scratch_database)

    owner = uuid.uuid4()
    with store.session() as db:
        model = q.upsert_model(
            db, model_id=uuid.uuid4(), name="pg", parameters={"nested": {"ok": True}}
        )
        q.enqueue(db, kind=RunKind.train, label="r", config={"a": [1, 2]})
        # The one statement that proves the `ALTER TYPE` ran: Postgres holds
        # `run_kind` as a real enum, and a value the migration failed to add is
        # invisible until something tries to store it.
        q.enqueue(db, kind=RunKind.tokenize, label="v", config={})
        q.upsert_tokenizer(
            db, tokenizer_id=uuid.uuid4(), name="pg-vocab", datasets=[["A", None]]
        )

    with store.session() as db:
        stored = q.find_model_by_name(db, "pg")
        assert stored.parameters == {"nested": {"ok": True}}
        assert q.find_tokenizer_by_name(db, "pg-vocab").datasets == [["A", None]]
        from sqlalchemy import select
        assert db.scalar(select(TrainingRun)).training_type == "pretrain"
