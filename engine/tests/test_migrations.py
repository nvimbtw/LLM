"""The migrations and the models must describe the same database.

Editing `store/schema.py` without a migration is silent: the tests pass against
a `create_all` database and production is missing the column. This is the check
that says so, and it is the only test that needs a real Postgres.
"""

import os

import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.postgres

ADMIN_URL = os.environ.get("ENGINE_TEST_ADMIN_URL", "postgresql+psycopg://localhost/postgres")
SCRATCH = "slate_engine_migration_check"


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
    previous = os.environ.get("ENGINE_DATABASE_URL")
    os.environ["ENGINE_DATABASE_URL"] = url
    try:
        yield url
    finally:
        if previous is None:
            os.environ.pop("ENGINE_DATABASE_URL", None)
        else:
            os.environ["ENGINE_DATABASE_URL"] = previous

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
    from store.schema import Feedback, RunKind, Suite

    command.upgrade(Config("alembic.ini"), "head")
    store.configure(scratch_database)

    owner = uuid.uuid4()
    with store.session() as db:
        model = q.upsert_model(
            db, model_id=uuid.uuid4(), name="pg", parameters={"nested": {"ok": True}}
        )
        chat = q.create_chat(db, owner, "c")
        reply = q.add_message(db, chat_id=chat.id, content="a", model_id=model.id)
        q.set_feedback(db, reply.id, owner, Feedback.up)
        q.open_benchmark(db, model=model, suite=Suite.language, score=42.0)
        q.enqueue(db, kind=RunKind.train, label="r", config={"a": [1, 2]})

    with store.session() as db:
        stored = q.find_model_by_name(db, "pg")
        assert stored.parameters == {"nested": {"ok": True}}
        assert q.messages_of(db, chat.id)[0].feedback is Feedback.up
        assert q.latest_benchmark(db, stored.id, Suite.language, 1).score == 42.0

        # The cascade is declared on the FK, which SQLite does not enforce by
        # default — this is the dialect that actually applies it.
        q.delete_model(db, stored.id)
        assert q.benchmarks_of(db, stored.id) == []
