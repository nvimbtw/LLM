import pytest

from tests.fixtures import tiny_env
from tests.fixtures.tiny_env import TINY_MODEL  # noqa: F401  (re-exported for tests)


@pytest.fixture(scope="session", autouse=True)
def data_root(tmp_path_factory):
    """Point DATA_DIR at a throwaway tree holding only the `tiny` fixture model.

    Autouse because `storage` resolves every path through this variable at call
    time — a test running before it was set would write into the real
    `llm-engine/data/`.
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
def store_session(tmp_path):
    """Bind a disposable SQLite database and restore the previous store on exit.

    A file gives HTTP waiters and worker threads independent connections so their
    transactions cannot commit or roll back each other's work. PostgreSQL-specific
    behavior remains covered by test_migrations.py.
    """
    import store

    previous = store._sessions
    factory = store.configure(f"sqlite+pysqlite:///{tmp_path / 'store.sqlite3'}", create=True)
    try:
        yield store.session
    finally:
        factory.kw["bind"].dispose()
        store._sessions = previous


@pytest.fixture
async def engine_client(data_root, store_session):
    """The app with the wiring its lifespan would have done.

    `ASGITransport` never runs lifespan, so the store binding and the drain
    loop have to be started here — without the loop a queued run never
    executes and anything waiting on one hangs rather than failing. This fixture
    owns pause restoration until its drain loop has stopped, so nested visibility
    fixtures cannot reopen admission during their teardown.
    """
    import asyncio
    import contextlib

    from httpx import ASGITransport, AsyncClient

    import main
    from api.runs import REGISTRY

    REGISTRY.bind_loop(asyncio.get_running_loop())
    paused = main.RUNNER.paused
    drain = asyncio.create_task(main.RUNNER.drain_forever())
    try:
        async with AsyncClient(
            transport=ASGITransport(app=main.app), base_url="http://llm-engine"
        ) as client:
            yield client
    finally:
        main.RUNNER.set_paused(True)
        drain.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await drain
        main.RUNNER.set_paused(paused)


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
