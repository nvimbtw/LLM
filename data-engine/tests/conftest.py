import asyncio
import contextlib
import uuid

import pytest


@pytest.fixture
def store_session():
    """A throwaway database per test.

    SQLite in memory rather than the real Postgres: the schema uses nothing
    dialect-specific, and a test that needs a server is a test nobody runs. The
    Postgres path is covered by the migration check in `test_migrations.py`.
    """
    import store

    factory = store.configure("sqlite+pysqlite:///:memory:", create=True)
    yield store.session
    store._sessions = factory


@pytest.fixture
async def client(store_session):
    """The app with the wiring its lifespan would have done.

    `ASGITransport` never runs lifespan, so the registry binding and the drain
    loop are started here — without the loop a queued run never executes and
    anything waiting on one hangs rather than failing.
    """
    from httpx import ASGITransport, AsyncClient

    import main
    from api.runs import REGISTRY

    REGISTRY.bind_loop(asyncio.get_running_loop())
    drain = asyncio.create_task(main.RUNNER.drain_forever())
    try:
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://data-engine"
        ) as http:
            yield http
    finally:
        # Awaited, not just cancelled: the loop dies with this test, and a task
        # left pending on it keeps a waiter on the runner's wake event that the
        # next test's `set()` would then try to resolve on a closed loop.
        drain.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain


@pytest.fixture
def owner() -> uuid.UUID:
    return uuid.uuid4()


@pytest.fixture
def acting(owner):
    """A bearer header naming an owner, as the backend would mint it.

    Unsigned: with no `DATA_ENGINE_SERVICE_SECRET` set the service reads `act`
    without verifying, which is the development pair these tests run as.
    """
    import jwt

    token = jwt.encode({"act": str(owner), "aud": "data-engine", "iss": "backend"}, "unsigned")
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture
def pretrain_dataset(store_session, owner):
    """A small ready dataset, so tests about processing need not pull one."""
    import store
    from store import queries as q
    from store.schema import DatasetKind, DatasetOrigin, DatasetStatus

    documents = [
        {"content": "<p>Hello   world</p>", "byte_size": 20},
        {"content": "   Plenty of text here, easily long enough to survive a filter.   ",
         "byte_size": 65},
        {"content": "short", "byte_size": 5},
        {"content": "<p>Hello   world</p>", "byte_size": 20},
    ]

    with store.session() as db:
        row = q.create_dataset(
            db,
            name="fixture-pretrain",
            kind=DatasetKind.pretrain,
            origin=DatasetOrigin.huggingface,
            owner_id=owner,
            source_repo="fixture/corpus",
        )
        dataset_id = row.id
        q.add_documents(db, dataset_id, 0, documents)
        q.finish_dataset(db, dataset_id, DatasetStatus.ready)

    return dataset_id
