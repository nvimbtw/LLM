import uuid

import pytest


@pytest.fixture
def store_session():
    """A throwaway database per test.

    SQLite in memory rather than the real Postgres, the same bargain the other
    two services strike: the schema uses nothing dialect-specific beyond the
    JSON variant, and a test that needs a server is a test nobody runs.
    """
    import store

    factory = store.configure("sqlite+pysqlite:///:memory:", create=True)
    yield store.session
    store._sessions = factory


@pytest.fixture
def sandbox(monkeypatch):
    """A just-bash that lives in this process.

    A dict of filesystems keyed by session, and only the handful of commands the
    tests assert on. It is emphatically **not** a shell: what a command does is
    the sandbox's own repository's business. What this stands in for is the wire
    — that a step reaches it, that a seeding failure is seen rather than
    swallowed, that what came back is recorded.
    """
    from gyms import sandbox as module

    class Fake:
        def __init__(self) -> None:
            self.files: dict[str, dict[str, str]] = {}
            self.commands: list[tuple[str, str]] = []
            self.fail_writes = False

        def run(self, session: str, command: str, timeout_ms: int = 0) -> module.Ran:
            self.commands.append((session, command))
            files = self.files.setdefault(session, {})
            if command.startswith("mkdir -p "):
                if self.fail_writes:
                    return module.Ran("", "read-only file system", 1, False)
                path = command.split("cat > ", 1)[1].split(" <<", 1)[0].strip("'")
                files[path] = command.split("\n", 1)[1].rsplit("\n", 2)[0]
                return module.Ran("", "", 0, False)
            if command.startswith("rm -rf "):
                # `sandbox.clear`. Stood in for because a route now depends on
                # it: demonstrations reuse one workspace rather than minting a
                # filesystem per task, so a clear that does nothing is a batch
                # whose second task is seeded over the first.
                files.clear()
                return module.Ran("", "", 0, False)
            if command.startswith("find "):
                return module.Ran(f"{len(files)}\n", "", 0, False)
            if command.startswith("ls"):
                return module.Ran("\n".join(sorted(files)), "", 0, False)
            if command.startswith("grep "):
                needle = command.split(" ")[-1]
                hits = [p for p, body in files.items() if needle in body]
                return module.Ran("\n".join(hits), "", 0 if hits else 1, False)
            return module.Ran("", f"{command.split(' ')[0]}: command not found", 127, False)

    fake = Fake()
    monkeypatch.setattr(module, "run", fake.run)

    return fake


@pytest.fixture
async def client(store_session, sandbox):
    from httpx import ASGITransport, AsyncClient

    import main

    async with AsyncClient(
        transport=ASGITransport(app=main.app), base_url="http://gyms-engine"
    ) as client:
        yield client


@pytest.fixture
def a_run(client):
    """Create a run through the API and hand back what it answered."""

    async def make(**overrides):
        body = {
            "gym_id": "bash",
            "model_name": "tiny",
            "task_types": ["bash_grep"],
            "count": 1,
            "seed_base": 7,
            **overrides,
        }
        response = await client.post("/gym/runs", json=body)
        assert response.status_code == 201, response.text
        return response.json()

    return make


@pytest.fixture
def unknown_id() -> uuid.UUID:
    return uuid.uuid4()
