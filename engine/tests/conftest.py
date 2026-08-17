import pytest

from tests.fixtures import tiny_env
from tests.fixtures.tiny_env import TINY_MODEL  # noqa: F401  (re-exported for tests)


@pytest.fixture(scope="session", autouse=True)
def data_root(tmp_path_factory):
    """Point DATA_DIR at a throwaway tree holding only the `tiny` fixture model.

    Autouse because `storage` resolves every path through this variable at call
    time — a test running before it was set would write into the real
    `engine/data/`.
    """
    with tiny_env.data_root(tmp_path_factory.mktemp("data")) as root:
        yield root


@pytest.fixture(autouse=True)
def _clear_model_cache():
    """`orchestrate.chat` memoizes one model per process, keyed by name alone —
    so a test that rebuilt `tiny` would otherwise get the previous weights."""
    from llm.orchestrate import chat

    chat._CACHE.clear()
    yield
    chat._CACHE.clear()


@pytest.fixture
def tokenizer(tokenizer_module):
    return tokenizer_module


@pytest.fixture
def model(model_module):
    return model_module


@pytest.fixture(scope="module")
def tokenizer_module(data_root):
    from llm import storage
    from llm.corpus import bpe

    return bpe.load(storage.model_dir(TINY_MODEL))


@pytest.fixture(scope="module")
def model_module(data_root):
    from llm import storage
    from llm.model import checkpoint

    loaded, _ = checkpoint.load(storage.weights_path(TINY_MODEL), "cpu")
    return loaded


@pytest.fixture(scope="session")
def recorded_trace() -> dict:
    import json

    from tests.fixtures.record import TRACE

    if not TRACE.is_file():
        pytest.fail(f"no recorded trace at {TRACE}; run tests.fixtures.record")
    return json.loads(TRACE.read_text())


@pytest.fixture
def store_session():
    """A throwaway database per test.

    SQLite in memory rather than the real Postgres: the schema uses nothing
    dialect-specific, and a test that needs a server is a test nobody runs.
    The Postgres path is covered by the migration check in test_migrations.py.
    """
    import store

    factory = store.configure("sqlite+pysqlite:///:memory:", create=True)
    yield store.session
    store._sessions = factory


@pytest.fixture
async def engine_client(data_root, store_session):
    """The app with the wiring its lifespan would have done.

    `ASGITransport` never runs lifespan, so the store binding and the drain
    loop have to be started here — without the loop a queued run never
    executes and anything waiting on one hangs rather than failing.
    """
    import asyncio
    import contextlib

    from httpx import ASGITransport, AsyncClient

    import main
    from api.runs import REGISTRY

    REGISTRY.bind_loop(asyncio.get_running_loop())
    drain = asyncio.create_task(main.RUNNER.drain_forever())
    try:
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://engine"
        ) as client:
            yield client
    finally:
        # Awaited, not just cancelled: the loop dies with this test, and a task
        # left pending on it keeps a waiter on the runner's wake event that the
        # next test's `set()` would then try to resolve on a closed loop.
        drain.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain


@pytest.fixture
def trained_id() -> str:
    """A model's weights live under its id, so a run names the directory it
    will write and the display name is a separate column."""
    import uuid

    return str(uuid.uuid4())
