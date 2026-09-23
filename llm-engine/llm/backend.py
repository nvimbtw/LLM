"""The one hop that runs toward the Rust backend.

Everything the LLM engine needs and does not own arrives through here: the documents
a corpus is packed from, which belong to the data engine, and the gym episodes a
reinforcement-learning batch is rolled out over, which belong to the gyms
engine. The LLM engine cannot ask either directly — the Python services never speak
to each other — so it asks the backend, which re-mints a token toward the other
service carrying the same acting caller.

This module is the token, the base URL and the failure. What each consumer does
with them differs enough that they are not folded in with it:

**A read may be retried and a write may not.** `corpus.remote` retries a dropped
page because the alternative is losing the pack around it, and asking for the
same page twice costs a page. A gym step is a command that ran in a real
filesystem; asking again after a lost response runs it twice, records two steps
and pays the shaping reward twice for one action. So `Hop` makes exactly one
attempt and the retry lives with the caller that can afford it.
"""

import os
import time
import uuid

import httpx
import jwt

#: This service's own name, as it appears in the `iss` claim of a token it mints.
#: The same string is `AUDIENCE` in `api/auth.py` (what the backend addresses us
#: by) and `llm_engine::AUDIENCE` in Rust; `tests/test_remote.py` pins them together.
SERVICE_NAME = "llm-engine"

#: Who the token is addressed to. The mirror of the backend's own audiences: a
#: token minted here for the backend cannot open a service, and vice versa.
AUDIENCE = "backend"

#: Signs a token when there is no shared secret, exactly as the Rust proxy does
#: in the other direction. The signature is then worthless, but `act` is not —
#: without it the backend cannot scope the request to anyone.
UNSIGNED = "unsigned"

TOKEN_TTL_SECS = 60

DEFAULT_BASE_URL = "http://127.0.0.1:8080"

#: Per request. Everything behind this hop is local, so a slow one is a wedged
#: service rather than a big response.
REQUEST_TIMEOUT_SECS = 30.0


class RemoteError(RuntimeError):
    """The backend refused or could not answer. Carries the status when there was
    one, so a caller can tell "no such dataset" from "nothing is listening"."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def base_url() -> str:
    return os.environ.get("BACKEND_URL") or DEFAULT_BASE_URL


def secret() -> str | None:
    return os.environ.get("BACKEND_SERVICE_SECRET") or None


def detail(response: httpx.Response) -> str:
    """The backend's own message, when it sent one worth repeating."""
    try:
        body = response.json()
    except ValueError:
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("error") or body.get("detail") or body)[:200]
    return str(body)[:200]


class Hop:
    """One caller's connection to the backend.

    Built per run rather than shared, because `owner` is the person the run was
    queued for and it is the whole of what the backend scopes by.
    """

    def __init__(
        self,
        owner: str | None = None,
        *,
        url: str | None = None,
        service_secret: str | None = None,
        timeout: float = REQUEST_TIMEOUT_SECS,
    ):
        self.owner = owner or "system"
        self.base_url = (url or base_url()).rstrip("/")
        self.secret = service_secret if service_secret is not None else secret()
        self._http = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "Hop":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def token(self) -> str:
        """Minted per request rather than cached: it lives for a minute, and a
        cached one would need refreshing on a schedule to save one HMAC."""
        return jwt.encode(
            {
                "iss": SERVICE_NAME,
                "aud": AUDIENCE,
                "act": self.owner,
                "scope": [],
                "jti": str(uuid.uuid4()),
                "exp": int(time.time()) + TOKEN_TTL_SECS,
            },
            self.secret or UNSIGNED,
            algorithm="HS256",
        )

    def request(self, method: str, path: str, **kwargs) -> tuple[dict, int]:
        """One attempt, and the response's size on the wire.

        The size is what a paging caller plans the next page from — measured
        rather than estimated from the rows, because a thousand books and a
        thousand web snippets are not the same page.
        """
        try:
            response = self._http.request(
                method,
                f"{self.base_url}{path}",
                headers={"Authorization": f"Bearer {self.token()}"},
                **kwargs,
            )
        except httpx.HTTPError as e:
            raise RemoteError(f"{path}: {e}") from e

        if response.status_code >= 400:
            raise RemoteError(
                f"{path}: {response.status_code} {detail(response)}", response.status_code
            )
        return response.json(), len(response.content)
