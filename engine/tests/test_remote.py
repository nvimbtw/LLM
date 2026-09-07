"""Reading documents through the backend.

Run against a stand-in for the backend rather than a mocked client: what is
being asserted is the shape of the conversation — how many rows are asked for,
what happens at a page boundary, what is retried — and a mock would only assert
that the code calls the functions it calls.
"""

import json
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import jwt
import pytest

from llm.corpus import remote

SECRET = "test-inbound-secret"


class Backend:
    """A stand-in for the backend's `/internal` routes.

    Holds a dataset of `count` documents of `size` bytes each, records every
    request it is asked, and can be told to fail the next few in a row.
    """

    def __init__(self, count: int, size: int = 64, failures: int = 0, status: int = 500):
        self.dataset_id = str(uuid.uuid4())
        self.count = count
        self.size = size
        self.failures = failures
        self.status = status
        self.requests: list[dict] = []
        #: The most requests it was ever answering at once — what says whether
        #: the reader is overlapping its pages or taking turns with itself.
        self.peak = 0
        self._live = 0
        self._counting = threading.Lock()

        backend = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                query = {k: int(v[0]) for k, v in parse_qs(url.query).items()}
                with backend._counting:
                    backend._live += 1
                    backend.peak = max(backend.peak, backend._live)
                    backend.requests.append(
                        {
                            "path": url.path,
                            "authorization": self.headers.get("Authorization", ""),
                            **query,
                        }
                    )
                try:
                    return self._answer(url, query)
                finally:
                    with backend._counting:
                        backend._live -= 1

            def _answer(self, url, query):

                if backend.failures > 0:
                    backend.failures -= 1
                    return self._send(backend.status, {"error": "not now"})

                if url.path.endswith("/documents"):
                    return self._send(200, backend._page(query))
                if "/internal/datasets/" in url.path:
                    return self._send(200, backend._dataset())
                return self._send(404, {"error": "no such route"})

            def _send(self, status: int, body: dict):
                payload = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://{}:{}".format(*self._server.server_address)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def _dataset(self) -> dict:
        return {
            "id": self.dataset_id,
            "name": "TinyStories",
            "kind": "pretrain",
            "document_count": self.count,
            "updated_at": "2026-08-19T00:00:00+00:00",
        }

    def _page(self, query: dict) -> dict:
        offset, limit = query.get("offset", 0), query.get("limit", 20)
        rows = [
            {"id": str(uuid.uuid4()), "ordinal": i, "content": "x" * self.size, "messages": None}
            for i in range(offset, min(offset + limit, self.count))
        ]
        return {"documents": rows, "offset": offset, "total": self.count}

    @property
    def pages(self) -> list[dict]:
        return [r for r in self.requests if r["path"].endswith("/documents")]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def backend():
    served = []

    def start(count: int, **kwargs) -> Backend:
        served.append(Backend(count, **kwargs))
        return served[-1]

    yield start
    for server in served:
        server.close()


@pytest.fixture(autouse=True)
def _no_backoff(monkeypatch):
    """The retry delay is real time and the test asserts the retry, not the wait."""
    monkeypatch.setattr(remote, "RETRY_BACKOFF_SECS", 0)


def client(server: Backend, owner: str | None = None) -> remote.Client:
    return remote.Client(owner, url=server.url, service_secret=SECRET)


# ── the conversation ────────────────────────────────────────────────────────


def test_every_document_arrives_once_and_in_order(backend):
    """Paging is where a corpus quietly loses or repeats a document, and neither
    is visible afterwards: the pack would simply be wrong."""
    server = backend(500)

    with client(server) as remote_client:
        ordinals = [d["ordinal"] for d in remote_client.documents(server.dataset_id)]

    assert ordinals == list(range(500))
    assert len(server.pages) > 1


def test_pages_overlap_rather_than_taking_turns(backend):
    """Nearly all of a page is the waiting — an offset scan, a JSON encode, two
    service hops — so a reader that asks for one at a time spends the pack's
    wall clock on a socket. Measured on `cosmopedia` at offset 250,000: 3.12
    MB/s one at a time against 8.10 MB/s four at a time."""
    server = backend(4_000, size=32)

    with client(server) as remote_client:
        list(remote_client.documents(server.dataset_id))

    assert server.peak > 1
    assert server.peak <= remote.PAGE_CONCURRENCY


def test_the_first_page_of_a_dataset_is_asked_for_alone(backend):
    """It is what says how many documents there are, and a window planned
    without that could ask past the end — where a short page shifts every offset
    behind it and the pack silently repeats a stretch of the corpus."""
    server = backend(4_000, size=32)

    with client(server) as remote_client:
        next(iter(remote_client.documents(server.dataset_id)))

    assert len(server.pages) == 1


@pytest.mark.parametrize("count", [1, 17, 199, 200, 201, 999])
def test_a_window_that_runs_off_the_end_reads_every_document_once(backend, count):
    """Where windowing breaks if it breaks: the last page of a dataset is short
    by definition, and it must be the last page of its window too."""
    server = backend(count, size=32)

    with client(server) as remote_client:
        ordinals = [d["ordinal"] for d in remote_client.documents(server.dataset_id)]

    assert ordinals == list(range(count))


@pytest.mark.parametrize("cap", [1, 33, 250, 517])
def test_a_cap_landing_inside_a_window_is_still_the_first_n(backend, cap):
    """A window is planned in whole pages, so a cap falls in the middle of one.
    The rows past it are read and dropped; yielding them would be training on
    data nobody selected."""
    server = backend(2_000, size=32)

    with client(server) as remote_client:
        taken = [d["ordinal"] for d in remote_client.documents(server.dataset_id, cap=cap)]

    assert taken == list(range(cap))


def test_a_cap_takes_the_first_n_documents(backend):
    """What `{"name": ..., "count": N}` means now there are no files to count."""
    server = backend(500)

    with client(server) as remote_client:
        taken = list(remote_client.documents(server.dataset_id, cap=42))

    assert [d["ordinal"] for d in taken] == list(range(42))
    # And it stops asking, rather than reading the dataset and slicing it.
    assert sum(page["limit"] for page in server.pages) <= 42 + remote.FIRST_PAGE_DOCUMENTS


def test_a_sample_can_start_anywhere(backend):
    """`plan_sample` spreads its budget across a dataset rather than taking it
    from the front, which is an offset jump here rather than a scan."""
    server = backend(500)

    with client(server) as remote_client:
        taken = list(remote_client.documents(server.dataset_id, start=300, cap=5))

    assert [d["ordinal"] for d in taken] == [300, 301, 302, 303, 304]


def test_no_page_exceeds_what_the_data_engine_allows(backend):
    """Its router refuses a larger `limit` outright, so asking for one is a 422
    rather than a big page."""
    server = backend(2000, size=4)

    with client(server) as remote_client:
        list(remote_client.documents(server.dataset_id))

    assert max(page["limit"] for page in server.pages) <= remote.MAX_PAGE_DOCUMENTS


def test_big_documents_shrink_the_page(backend):
    """The reason a page is sized in bytes at all: a thousand books is not a
    thousand web snippets, and `limit` counts both the same."""
    small = backend(400, size=32)
    large = backend(400, size=256 * 1024)

    with client(small) as c:
        list(c.documents(small.dataset_id))
    with client(large) as c:
        list(c.documents(large.dataset_id))

    assert max(p["limit"] for p in small.pages) == remote.MAX_PAGE_DOCUMENTS
    # 4 MiB of target against 256 KiB documents is ~16 of them.
    assert max(p["limit"] for p in large.pages) < 32


def test_the_page_after_the_first_is_planned_from_what_the_first_weighed(backend):
    server = backend(300, size=1024)

    with client(server) as remote_client:
        list(remote_client.documents(server.dataset_id, target_bytes=32 * 1024))

    first, second = server.pages[0], server.pages[1]
    assert first["limit"] == remote.FIRST_PAGE_DOCUMENTS
    # ~32 documents of 1 KiB fit in a 32 KiB target, minus the JSON around them.
    assert 8 <= second["limit"] <= 32


# ── failure ─────────────────────────────────────────────────────────────────


def test_a_dropped_page_costs_a_retry_and_not_the_pack(backend):
    """The reason this pages instead of reading one long export: a corpus is
    minutes of tokenizing, and losing all of it to one dropped connection is the
    failure the whole shape is chosen to avoid."""
    server = backend(50, failures=2)

    with client(server) as remote_client:
        ordinals = [d["ordinal"] for d in remote_client.documents(server.dataset_id)]

    assert ordinals == list(range(50))


def test_a_refusal_is_not_retried(backend):
    """A 404 means the dataset is gone. Asking again is a slower way to fail, and
    a 401 asked three times is three lines in the backend's log for one mistake."""
    server = backend(10, failures=5, status=404)

    with client(server) as remote_client, pytest.raises(remote.RemoteError) as raised:
        list(remote_client.documents(server.dataset_id))

    assert raised.value.status == 404
    assert len(server.requests) == 1


def test_a_failure_that_never_clears_gives_up_and_says_what_it_saw(backend):
    server = backend(10, failures=99)

    with client(server) as remote_client, pytest.raises(remote.RemoteError) as raised:
        list(remote_client.documents(server.dataset_id))

    assert raised.value.status == 500
    assert "not now" in str(raised.value)
    assert len(server.requests) == remote.PAGE_ATTEMPTS


# ── who is asking ───────────────────────────────────────────────────────────


def test_the_token_names_this_service_the_backend_and_the_person(backend):
    server = backend(10)
    owner = str(uuid.uuid4())

    with client(server, owner) as remote_client:
        remote_client.dataset(server.dataset_id)

    sent = server.requests[0]["authorization"].removeprefix("Bearer ")
    claims = jwt.decode(sent, SECRET, algorithms=["HS256"], audience="backend", issuer="engine")
    assert claims["act"] == owner
    assert claims["aud"] == "backend"
    assert claims["iss"] == "engine"


def test_without_a_secret_the_claim_still_travels(backend):
    """Unsigned is the development pair — the signature is then worthless, but
    the backend still has to know who the run is for."""
    server = backend(10)
    owner = str(uuid.uuid4())

    with remote.Client(owner, url=server.url, service_secret=None) as remote_client:
        remote_client.dataset(server.dataset_id)

    sent = server.requests[0]["authorization"].removeprefix("Bearer ")
    assert jwt.decode(sent, options={"verify_signature": False})["act"] == owner


def test_a_run_with_no_owner_acts_as_the_system(backend):
    server = backend(10)

    with client(server) as remote_client:
        remote_client.dataset(server.dataset_id)

    sent = server.requests[0]["authorization"].removeprefix("Bearer ")
    assert jwt.decode(sent, options={"verify_signature": False})["act"] == "system"


def test_this_service_is_spelled_the_same_way_on_both_sides_of_the_hop():
    """The backend addresses the engine by one name and reads its `iss` as
    another; a rename that missed one of them would be a 401 with nothing
    obviously wrong anywhere."""
    from api import auth

    assert remote.SERVICE_NAME == auth.AUDIENCE


# ── what a key needs ────────────────────────────────────────────────────────


def test_a_dataset_reports_what_makes_a_corpus_key_honest(backend):
    """A dataset that was pulled again is a different corpus under the same id.
    The count and the timestamp are what let a key notice."""
    server = backend(1234)

    with client(server) as remote_client:
        dataset = remote_client.dataset(server.dataset_id)

    assert dataset.id == server.dataset_id
    assert dataset.name == "TinyStories"
    assert dataset.documents == 1234
    assert dataset.updated_at.startswith("2026-08-19")


# ── the vocabulary's sample ─────────────────────────────────────────────────


def test_a_sample_spans_the_dataset_rather_than_its_front(backend):
    """ADR 0004's guard, carried over from the file-based sampler.

    A corpus is ordered — by shard, by article, by whatever the pull walked — so
    its first documents are one region of it and not a sample of it. The Rust
    engine read FineWeb from 50 of its 5,498 shards this way, and the merges it
    learned were fitted to that region.
    """
    server = backend(4000, size=256)
    one = remote.Selected(
        id=server.dataset_id,
        name="Big",
        kind="pretrain",
        cap=None,
        documents=4000,
        bytes=4000 * 256,
        updated_at="t",
    )

    with client(server) as remote_client:
        drawn = list(remote.sample(remote_client, one, 64 * 1024, take_each=8 * 1024))

    offsets = [page["offset"] for page in server.pages]
    assert len(offsets) > 1
    # Every quarter of the dataset is represented — the property that fails when
    # a sample is taken from the front, however much of it is taken.
    for quarter in range(4):
        lo, hi = quarter * 1000, (quarter + 1) * 1000
        assert any(lo <= o < hi for o in offsets), f"nothing drawn from {lo}-{hi}"
    assert sum(len(text) for _, text in drawn) <= 64 * 1024


def test_a_sample_stays_inside_its_budget_however_large_the_documents(backend):
    """The ceiling that was missing before a budget existed: the trainer holds a
    count per distinct pretoken, so its peak follows the variety of what it is
    shown."""
    server = backend(200, size=128 * 1024)
    one = remote.Selected(
        id=server.dataset_id,
        name="Books",
        kind="pretrain",
        cap=None,
        documents=200,
        bytes=200 * 128 * 1024,
        updated_at="t",
    )
    budget = 64 * 1024

    with client(server) as remote_client:
        drawn = list(remote.sample(remote_client, one, budget, take_each=16 * 1024))

    # One document is twice the whole budget: taken whole it would blow it.
    assert sum(len(text) for _, text in drawn) <= budget
