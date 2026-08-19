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


@pytest.fixture(scope="session", autouse=True)
def backend():
    """Point BACKEND_URL at a stand-in serving the fixture datasets.

    Autouse and session-scoped for the same reason `data_root` is: every build
    path reads its text through this, and a test that ran before it was set
    would reach for whatever is listening on port 8080 — which, on the machine
    this is developed on, is a real backend holding real data.
    """
    import os

    from tests.fixtures.fake_backend import FakeBackend

    server = FakeBackend(tiny_env.datasets())
    previous = os.environ.get("BACKEND_URL")
    os.environ["BACKEND_URL"] = server.url
    try:
        yield server
    finally:
        if previous is None:
            os.environ.pop("BACKEND_URL", None)
        else:
            os.environ["BACKEND_URL"] = previous
        server.close()


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


@pytest.fixture
def data_client(backend):
    """A client of the stand-in backend, acting for nobody in particular."""
    from llm.corpus import remote

    with remote.Client(None) as caller:
        yield caller


@pytest.fixture
def packed(data_root, data_client):
    """The fixture corpus, packed the way a run packs it.

    Shared because three files want a real packed corpus and building one is the
    slowest thing in the suite.
    """
    from llm.corpus import pack, remote, vocab
    from tests.fixtures.tiny_env import TINY_DATASET_ID

    selection = vocab.resolve(data_client, [TINY_DATASET_ID])
    tokenizer_id = vocab.auto_id(selection, 512)
    vocab.build(data_client, tokenizer_id, selection, 512)
    fingerprint = vocab.fingerprint(tokenizer_id)
    key = pack.corpus_key(fingerprint, [(None, selection)])

    stats = pack.pack(
        (
            pack.SourceDoc(dataset=one.name, text=remote.text_of(row))
            for one in selection
            for row in data_client.documents(one.id, cap=one.cap)
        ),
        vocab.load(tokenizer_id),
        key=key,
        stages=[{"name": None, "epochs": 1, "datasets": [s.record() for s in selection]}],
        tokenizer_id=tokenizer_id,
        fingerprint=fingerprint,
    )
    return key, stats
