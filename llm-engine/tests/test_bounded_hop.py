"""Bounded Rust transport behavior, without a live socket or a numerical worker."""

from collections.abc import Iterator

import httpx
import jwt
import pytest

from llm import backend


class Body(httpx.SyncByteStream):
    """Track consumption and closure of an upstream stream, including an injected read failure."""

    def __init__(self, chunks: tuple[bytes, ...], failure: Exception | None = None) -> None:
        """Retain finite test chunks and an optional failure after their delivery."""
        self.chunks = chunks
        self.failure = failure
        self.reads = 0
        self.closed = False

    def __iter__(self) -> Iterator[bytes]:
        """Yield recorded chunks once, then surface the upstream failure if configured."""
        for chunk in self.chunks:
            self.reads += 1
            yield chunk
        if self.failure is not None:
            raise self.failure

    def close(self) -> None:
        """Record upstream retirement on success and every refusal path."""
        self.closed = True


@pytest.fixture
def connection():
    """Provide the actual Hop with a deterministic HTTP transport and retain requests for no-retry assertions."""
    requests = []
    streams = []
    hop = backend.Hop("job-owner", url="http://rust.test", service_secret="test-secret")
    hop._http.close()

    def respond(request: httpx.Request) -> httpx.Response:
        """Record each actual request and consume one configured streamed response."""
        requests.append(request)
        status, headers, body = streams.pop(0)
        return httpx.Response(status, headers=headers, stream=body)

    hop._http = httpx.Client(transport=httpx.MockTransport(respond))
    with hop:
        yield hop, streams, requests


def test_bounded_hop_preserves_actor_and_raw_bytes(connection):
    """The bounded route signs the queued actor, disables compression and returns exactly the finite body."""
    hop, streams, requests = connection
    body = Body((b'{"ok":', b"true}"))
    streams.append((200, {}, body))
    assert hop.request_bytes("POST", "/internal/gym/runs", limit_bytes=11, json={"count": 1}) == b'{"ok":true}'
    claims = jwt.decode(requests[0].headers["Authorization"][7:], "test-secret", algorithms=["HS256"], audience="backend")
    assert (claims["act"], claims["iss"]) == ("job-owner", "llm-engine")
    assert requests[0].headers["Accept-Encoding"] == "identity"
    assert requests[0].url.path == "/internal/gym/runs"
    assert body.closed and len(requests) == 1


@pytest.mark.parametrize("headers", [{"Content-Length": "12"}, {"Content-Length": "bad"}, {"Content-Encoding": "gzip"}])
def test_bounded_hop_refuses_headers_before_consuming_body(connection, headers):
    """Declared overflows, invalid lengths and hidden decompression never enter body accumulation."""
    hop, streams, requests = connection
    body = Body((b"{}",))
    streams.append((200, headers, body))
    with pytest.raises(backend.RemoteError):
        hop.request_bytes("POST", "/internal/gym/runs", limit_bytes=11)
    assert body.closed and body.reads == 0 and len(requests) == 1


def test_bounded_hop_stops_at_overflow_without_reading_the_tail(connection):
    """An undeclared oversized response closes before the rest of its stream can be consumed."""
    hop, streams, requests = connection
    body = Body((b"x" * 12, b"tail"))
    streams.append((200, {}, body))
    with pytest.raises(backend.RemoteError, match="byte budget"):
        hop.request_bytes("POST", "/internal/gym/runs", limit_bytes=11)
    assert body.closed and body.reads == 1 and len(requests) == 1


def test_bounded_hop_does_not_repeat_a_partly_read_write(connection):
    """A lost response remains one issued HTTP write, with its original transport error chained."""
    hop, streams, requests = connection
    failure = httpx.ReadError("response lost")
    body = Body((b"{",), failure)
    streams.append((200, {}, body))
    with pytest.raises(backend.RemoteError) as caught:
        hop.request_bytes("POST", "/internal/gym/episodes/example/step", limit_bytes=11)
    assert caught.value.__cause__ is failure
    assert body.closed and len(requests) == 1


@pytest.mark.parametrize("declared", ["1", "3"])
def test_bounded_hop_requires_exact_declared_length(connection, declared):
    """A body that ends at another length cannot turn an incomplete or inconsistent response into success."""
    hop, streams, requests = connection
    body = Body((b"{}",))
    streams.append((200, {"Content-Length": declared}, body))
    with pytest.raises(backend.RemoteError, match="declared length"):
        hop.request_bytes("POST", "/internal/gym/runs", limit_bytes=11)
    assert body.closed and len(requests) == 1


def test_bounded_hop_retains_bounded_http_error_status(connection):
    """A finite Rust rejection preserves its status and detail, and closes its stream once."""
    hop, streams, requests = connection
    body = Body((b'{"error":"denied"}',))
    streams.append((403, {}, body))
    with pytest.raises(backend.RemoteError, match="denied") as caught:
        hop.request_bytes("POST", "/internal/gym/runs", limit_bytes=64)
    assert caught.value.status == 403
    assert body.closed and len(requests) == 1


def test_bounded_hop_checks_elapsed_time_between_reads(connection, monkeypatch):
    """An elapsed deadline retires the response before another read, without claiming an in-flight call was interrupted."""
    hop, streams, requests = connection
    clock = iter((0.0, 0.0, 2.0))
    monkeypatch.setattr(backend.time, "monotonic", lambda: next(clock))
    body = Body((b"x" * 65536, b"tail"))
    streams.append((200, {}, body))
    with pytest.raises(backend.RemoteError, match="elapsed deadline"):
        hop.request_bytes("POST", "/internal/gym/runs", limit_bytes=65536, timeout_seconds=1.0)
    assert body.closed and body.reads == 1 and len(requests) == 1


@pytest.mark.parametrize("limit,timeout", [(0, 1.0), (True, 1.0), (1, 0.0), (1, float("inf")), (1, float("nan"))])
def test_bounded_hop_rejects_invalid_limits_before_issuance(connection, limit, timeout):
    """Invalid admission cannot issue any HTTP request."""
    hop, _, requests = connection
    with pytest.raises(ValueError):
        hop.request_bytes("POST", "/internal/gym/runs", limit_bytes=limit, timeout_seconds=timeout)
    assert requests == []
