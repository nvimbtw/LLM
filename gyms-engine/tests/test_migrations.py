"""The migrations and the models must describe the same database.

Editing `store/schema.py` without a migration is silent: the tests pass against
a `create_all` database and production is missing the column. This is the check
that says so, and it is the only test here that needs a real Postgres.

It is also the check that was missing when `0001` was rewritten in place rather
than superseded — a database stamped at the old `0001` was never upgraded, and
the disagreement only surfaced as a 500 the first time a model ran a command.
"""

import os

import pytest
from sqlalchemy import create_engine, text

pytestmark = pytest.mark.postgres

ADMIN_URL = os.environ.get("GYMS_ENGINE_TEST_ADMIN_URL", "postgresql+psycopg://localhost/postgres")
SCRATCH = "slate_gyms_migration_check"


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
    previous = os.environ.get("GYMS_ENGINE_DATABASE_URL")
    os.environ["GYMS_ENGINE_DATABASE_URL"] = url
    try:
        yield url
    finally:
        if previous is None:
            os.environ.pop("GYMS_ENGINE_DATABASE_URL", None)
        else:
            os.environ["GYMS_ENGINE_DATABASE_URL"] = previous

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


def _diff(url: str) -> list:
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext

    from store.schema import Base

    engine = create_engine(url)
    try:
        with engine.connect() as connection:
            return compare_metadata(MigrationContext.configure(connection), Base.metadata)
    finally:
        engine.dispose()


def test_migrations_reproduce_the_models(scratch_database):
    """Upgrade from empty, then ask autogenerate what is still missing. A clean
    answer means the two agree."""
    from alembic import command
    from alembic.config import Config

    command.upgrade(Config("alembic.ini"), "head")
    assert _diff(scratch_database) == []


def test_a_database_left_on_the_first_schema_is_brought_forward(scratch_database):
    """The legacy shape, upgraded, lands on the same database a fresh one does.

    `0001` was rewritten in place instead of superseded, so one revision id
    names two schemas and the id alone cannot say which is present. This builds
    the old one, stamps it, and asserts `0002` reconciles it — the case the
    conditional branches in that migration exist for.
    """
    from alembic import command
    from alembic.config import Config

    engine = create_engine(scratch_database, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as c:
            c.execute(text("CREATE TYPE gym_status AS ENUM ('active', 'closed')"))
            c.execute(
                text(
                    "CREATE TABLE gym_sessions ("
                    " id uuid PRIMARY KEY, gym_id varchar(60) NOT NULL,"
                    " task_type varchar(60) NOT NULL, task_config jsonb,"
                    " status gym_status NOT NULL, owner_id uuid,"
                    " created_at timestamptz NOT NULL, closed_at timestamptz)"
                )
            )
            c.execute(
                text(
                    "CREATE TABLE gym_steps ("
                    " id uuid PRIMARY KEY, session_id uuid NOT NULL,"
                    " step_num integer NOT NULL, action text NOT NULL,"
                    " stdout text, stderr text, exit_code integer,"
                    " reward double precision, created_at timestamptz NOT NULL)"
                )
            )
            c.execute(text("CREATE TABLE alembic_version (version_num varchar(32) NOT NULL)"))
            c.execute(text("INSERT INTO alembic_version VALUES ('0001')"))
    finally:
        engine.dispose()

    command.upgrade(Config("alembic.ini"), "head")

    assert _diff(scratch_database) == []
    engine = create_engine(scratch_database)
    try:
        with engine.connect() as c:
            names = set(sa_table_names(c))
    finally:
        engine.dispose()
    assert "gym_sessions" not in names


def sa_table_names(connection) -> list[str]:
    import sqlalchemy as sa

    return sa.inspect(connection).get_table_names()
