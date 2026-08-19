"""The migrations and the models must describe the same database.

Editing `store/schema.py` without a migration is silent: the tests pass against
a `create_all` database and production is missing the column. This is the check
that says so, and it is the only test that needs a real Postgres.
"""

import os
import uuid

import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.postgres

ADMIN_URL = os.environ.get("DATA_ENGINE_TEST_ADMIN_URL", "postgresql+psycopg://localhost/postgres")
SCRATCH = "slate_data_migration_check"


def _postgres_reachable() -> bool:
    try:
        create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT").connect().close()
        return True
    except Exception:
        return False


@pytest.fixture
def scratch_database():
    if not _postgres_reachable():
        pytest.skip(f"no postgres at {ADMIN_URL}")

    admin = create_engine(ADMIN_URL, isolation_level="AUTOCOMMIT")
    with admin.connect() as c:
        c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH}"'))
        c.execute(text(f'CREATE DATABASE "{SCRATCH}"'))

    url = ADMIN_URL.rsplit("/", 1)[0] + f"/{SCRATCH}"
    previous = os.environ.get("DATA_ENGINE_DATABASE_URL")
    os.environ["DATA_ENGINE_DATABASE_URL"] = url
    try:
        yield url
    finally:
        if previous is None:
            os.environ.pop("DATA_ENGINE_DATABASE_URL", None)
        else:
            os.environ["DATA_ENGINE_DATABASE_URL"] = previous

        # Every pool holding this database has to let go before it can be
        # dropped, and a test that failed mid-way may have left one open.
        import store

        if store._sessions is not None:
            store._sessions.kw["bind"].dispose()
            store._sessions = None
        with admin.connect() as c:
            c.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :name AND pid <> pg_backend_pid()"
                ),
                {"name": SCRATCH},
            )
            c.execute(text(f'DROP DATABASE IF EXISTS "{SCRATCH}"'))


def test_migrations_reproduce_the_models(scratch_database):
    """Upgrade from empty, then ask autogenerate what is still missing. A clean
    answer means the two agree."""
    from alembic import command
    from alembic.autogenerate import compare_metadata
    from alembic.config import Config
    from alembic.migration import MigrationContext

    command.upgrade(Config("alembic.ini"), "head")

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
    from alembic import command
    from alembic.config import Config

    import store
    from store import queries as q
    from store.schema import DatasetKind, DatasetOrigin, DatasetStatus, RunKind

    command.upgrade(Config("alembic.ini"), "head")
    store.configure(scratch_database)

    owner = uuid.uuid4()
    with store.session() as db:
        prose = q.create_dataset(
            db, name="pg-prose", kind=DatasetKind.pretrain,
            origin=DatasetOrigin.huggingface, owner_id=owner, source_repo="a/b",
        )
        q.add_documents(db, prose.id, 0, [{"content": "hello", "byte_size": 5}])
        q.finish_dataset(db, prose.id, DatasetStatus.ready)

        chat = q.create_dataset(
            db, name="pg-chat", kind=DatasetKind.sft, origin=DatasetOrigin.derived,
            owner_id=owner, parent_id=prose.id,
            steps=[{"processor": "min-length", "params": {"characters": 1}}],
        )
        q.add_documents(
            db, chat.id, 0,
            [{"messages": [{"role": "user", "content": "hi"}], "byte_size": 2}],
        )
        q.finish_dataset(db, chat.id, DatasetStatus.ready)
        q.create_run(db, kind=RunKind.pull, label="r", config={"a": [1, 2]}, owner_id=owner)
        # The statement that proves both enum values exist: Postgres holds
        # `run_kind` as a real type, and a value the migration missed is
        # invisible until something tries to store it.
        q.create_run(db, kind=RunKind.process, label="p", config={}, owner_id=owner)
        prose_id, chat_id = prose.id, chat.id

    with store.session() as db:
        assert q.dataset(db, prose_id).document_count == 1
        assert q.dataset(db, chat_id).steps[0]["processor"] == "min-length"
        assert q.documents(db, chat_id, offset=0, limit=1)[0].messages[0]["role"] == "user"

        # `none_as_null` is what makes the content/messages constraint hold, and
        # only the real dialect enforces it.
        assert q.documents(db, prose_id, offset=0, limit=1)[0].messages is None

        # The cascade is declared on the FK, which SQLite does not enforce by
        # default — this is the dialect that actually applies it.
        q.delete_dataset(db, prose_id)

    with store.session() as db:
        assert q.documents(db, prose_id, offset=0, limit=1) == []
        # SET NULL, not cascade: deleting a source must not take its derived
        # datasets with it.
        assert q.dataset(db, chat_id) is not None
        assert q.dataset(db, chat_id).parent_id is None


def test_the_content_constraint_rejects_a_row_carrying_both(scratch_database):
    """The one invariant the types cannot express, on the dialect that enforces it."""
    from alembic import command
    from alembic.config import Config
    from sqlalchemy.exc import IntegrityError

    import store
    from store import queries as q
    from store.schema import DatasetKind, DatasetOrigin

    command.upgrade(Config("alembic.ini"), "head")
    store.configure(scratch_database)

    with store.session() as db:
        row = q.create_dataset(
            db, name="pg-bad", kind=DatasetKind.pretrain,
            origin=DatasetOrigin.huggingface, owner_id=None,
        )
        dataset_id = row.id

    with pytest.raises(IntegrityError):
        with store.session() as db:
            q.add_documents(
                db, dataset_id, 0,
                [{"content": "both", "messages": [{"role": "user", "content": "x"}]}],
            )
