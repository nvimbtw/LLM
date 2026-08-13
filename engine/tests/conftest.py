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
