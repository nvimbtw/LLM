"""Reading documents that belong to the data engine.

The engine owns no corpora. What a run trains on is a dataset in the *data*
engine's database, and the two Python services never speak — so a run asks the
Rust backend, which re-mints toward the other service carrying the same acting
caller. The hop is documented in `docs/backend/README.md`; this is the client
side of it.

Three things shape everything here:

**Nothing is written down.** A page is fetched, encoded, and dropped. There is no
cache and no staging directory, because a corpus the engine keeps a copy of is a
corpus the engine owns.

**Pages, not a stream.** The data engine can export a whole dataset as one NDJSON
response, and that is the wrong seam for this: the backend buffers a forwarded
response, and a connection dropped forty minutes into a pack would cost the whole
pack. A page costs one retry.

**A page is sized in bytes, not rows.** `limit` counts documents, and a thousand
books is not a thousand web snippets. The size of the last page sets the next
one's limit, against a byte target — bounded above by the data engine's own
ceiling of 200 rows, which is what actually binds for small documents. Peak
memory is one page, whatever the corpus turns out to be.
"""

import json
import os
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass

import httpx
import jwt

#: This service's own name, as it appears in the `iss` claim of a token it mints.
#: The same string is `AUDIENCE` in `api/auth.py` (what the backend addresses us
#: by) and `engine::AUDIENCE` in Rust; `tests/test_remote.py` pins them together.
SERVICE_NAME = "engine"

#: Who the token is addressed to. The mirror of the backend's own audiences: a
#: token minted here for the backend cannot open a service, and vice versa.
AUDIENCE = "backend"

#: Signs a token when there is no shared secret, exactly as the Rust proxy does
#: in the other direction. The signature is then worthless, but `act` is not —
#: without it the backend cannot scope the request to anyone.
UNSIGNED = "unsigned"

TOKEN_TTL_SECS = 60

DEFAULT_BASE_URL = "http://127.0.0.1:8080"

#: What a page should weigh. Matches `pack.ENCODE_BATCH_CHARS`, so one page is
#: one encode batch and is released as soon as it has been encoded.
PAGE_TARGET_BYTES = 1 << 22

#: The data engine refuses a larger `limit` (`MAX_PAGE` in its datasets router).
#: For small documents this, not the byte target, is what bounds a page — 200
#: web snippets are well under 4 MiB, and the extra round trips cost less over
#: loopback than one re-tokenized corpus does.
MAX_PAGE_DOCUMENTS = 200

#: Where sizing starts, before any page has been measured. Deliberately small:
#: the first page of a corpus of books should not be the one that hurts.
FIRST_PAGE_DOCUMENTS = 16

#: Per page. A page is small and local, so a slow one is a wedged service rather
#: than a big response.
REQUEST_TIMEOUT_SECS = 30.0

#: A dropped page is retried rather than losing the pack around it. Transport
#: failures and 5xx only — a 404 means the dataset is gone and retrying it is
#: just a slower way to fail.
PAGE_ATTEMPTS = 3
RETRY_BACKOFF_SECS = 0.5


class RemoteError(RuntimeError):
    """The backend refused or could not answer. Carries the status when there was
    one, so a caller can tell "no such dataset" from "nothing is listening"."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


@dataclass(frozen=True)
class Dataset:
    """What a corpus key needs to know about its source.

    `documents` and `updated_at` are here because they are what makes a key
    honest: a dataset that was pulled again is a different corpus under the same
    id, and a key that cannot see that would reuse a packed corpus built from
    less data and report success.
    """

    id: str
    name: str
    kind: str
    documents: int
    bytes: int
    updated_at: str

    @classmethod
    def of(cls, payload: dict) -> "Dataset":
        return cls(
            id=str(payload["id"]),
            name=payload["name"],
            kind=payload["kind"],
            documents=payload.get("document_count") or 0,
            bytes=payload.get("byte_size") or 0,
            updated_at=payload.get("updated_at") or "",
        )


@dataclass(frozen=True)
class Selected:
    """One dataset a run asked for, as it stands right now.

    A request names ids and caps; this is that request resolved against the data
    engine, and it is what everything downstream keys, records and reads. The
    resolution happens once, at the top of a build, for the reason the old
    directory expansion did: a selection compared against a *later* state of the
    world would reuse a corpus that predates the data it asked for.
    """

    id: str
    name: str
    kind: str
    cap: int | None
    #: How many documents the dataset holds — not how many this selection takes.
    documents: int
    bytes: int
    updated_at: str

    @property
    def taken(self) -> int:
        """How many documents this selection actually reads."""
        return min(self.cap, self.documents) if self.cap else self.documents

    def identity(self) -> list:
        """What a key is a function of.

        The id and the cap say what was asked for; the count and the timestamp
        say what answered. Without the last two a dataset that was pulled again
        would key identically to the smaller one it replaced. The *name* is
        deliberately absent — renaming a dataset changes no token, and a re-pack
        because someone fixed a typo is the expensive kind of mistake.
        """
        return [self.id, self.cap, self.documents, self.updated_at]

    def record(self) -> dict:
        """What the sidecar keeps, so a built corpus says what it is made of."""
        return {
            "id": self.id,
            "name": self.name,
            "count": self.cap,
            "documents": self.documents,
            "updated_at": self.updated_at,
        }

    def __str__(self) -> str:
        return f"{self.name}[:{self.cap}]" if self.cap else self.name


def resolve(client: "Client", selections) -> list["Selected"]:
    """What a request selects right now, in the order it named.

    Order is preserved and never sorted: concatenation order is a real training
    difference under a seeded shuffle.

    There is no "everything". A directory of corpora a person curated could
    reasonably mean all of it; a shared database that anyone can pull into
    cannot, and a run that silently widened to whatever was added last week
    would key apart every time somebody else worked.
    """
    if not selections:
        raise ValueError(
            "a run must name the datasets it trains on — datasets live in the data "
            "engine now, and there is no 'everything' to fall back on"
        )

    resolved = []
    for item in selections:
        dataset_id, name, cap = _selection(item)
        found = client.dataset(dataset_id)
        resolved.append(
            Selected(
                id=found.id,
                # The request may carry a name for readability, but the data
                # engine's is the one that is true right now.
                name=found.name or name,
                kind=found.kind,
                cap=cap,
                documents=found.documents,
                bytes=found.bytes,
                updated_at=found.updated_at,
            )
        )
    return resolved


def _selection(item) -> tuple[str, str, int | None]:
    """`(id, name, cap)` from a bare id, a mapping, or a `DatasetSelection`.

    Duck-typed rather than importing the config model, so this module stays
    usable from a script holding a list of ids and nothing else.
    """
    if isinstance(item, str):
        return item, "", None
    if isinstance(item, dict):
        return str(item["id"]), item.get("name") or "", item.get("count") or None
    return str(item.id), getattr(item, "name", "") or "", getattr(item, "count", None) or None


def base_url() -> str:
    return os.environ.get("BACKEND_URL") or DEFAULT_BASE_URL


def secret() -> str | None:
    return os.environ.get("BACKEND_SERVICE_SECRET") or None


class Client:
    """One caller's view of the datasets it may read.

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

    def __enter__(self) -> "Client":
        return self

    def __exit__(self, *_) -> None:
        self.close()

    # ── the seam ────────────────────────────────────────────────────────────

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

    def get(self, path: str, params: dict | None = None) -> tuple[dict, int]:
        """One call, with its response size. The size is what the next page is
        planned from, measured on the wire rather than estimated from the rows."""
        last: Exception | None = None
        for attempt in range(PAGE_ATTEMPTS):
            try:
                response = self._http.get(
                    f"{self.base_url}{path}",
                    params=params,
                    headers={"Authorization": f"Bearer {self.token()}"},
                )
            except httpx.HTTPError as e:
                last = RemoteError(f"{path}: {e}")
            else:
                if response.status_code < 400:
                    return response.json(), len(response.content)
                # A refusal is the backend's answer, not a hiccup. Only a server
                # fault is worth asking the same question again.
                if response.status_code < 500:
                    raise RemoteError(
                        f"{path}: {response.status_code} {_detail(response)}",
                        response.status_code,
                    )
                last = RemoteError(
                    f"{path}: {response.status_code} {_detail(response)}",
                    response.status_code,
                )

            if attempt + 1 < PAGE_ATTEMPTS:
                time.sleep(RETRY_BACKOFF_SECS * (attempt + 1))

        raise last if last else RemoteError(f"{path}: no answer")

    # ── what a build asks for ───────────────────────────────────────────────

    def dataset(self, dataset_id: str) -> Dataset:
        payload, _ = self.get(f"/internal/datasets/{dataset_id}")
        return Dataset.of(payload)

    def page(self, dataset_id: str, offset: int, limit: int) -> tuple[list[dict], int, int]:
        """One page: its rows, how many the dataset holds, and what it weighed.

        The primitive under `documents`, and what a strided sample uses directly:
        `sample` spreads its budget across a dataset rather than taking it from
        the front, which is an offset jump rather than a scan.
        """
        payload, weight = self.get(
            f"/internal/datasets/{dataset_id}/documents",
            {"offset": offset, "limit": _clamp(limit)},
        )
        return payload.get("documents", []), payload.get("total", 0), weight

    def documents(
        self,
        dataset_id: str,
        *,
        cap: int | None = None,
        start: int = 0,
        target_bytes: int = PAGE_TARGET_BYTES,
    ) -> Iterator[dict]:
        """Every document of a dataset, in `ordinal` order, one page at a time.

        `cap` takes the first N — the selection cap, which used to mean "the
        first N files" and now means the first N documents. It is the same
        promise it always made and a better-founded one: `ordinal` is the data
        engine's stable order, where a directory listing was ordering by
        filename and hoping.
        """
        offset = start
        limit = FIRST_PAGE_DOCUMENTS
        served = 0

        while cap is None or served < cap:
            if cap is not None:
                limit = min(limit, cap - served)

            rows, total, weight = self.page(dataset_id, offset, limit)
            if not rows:
                return

            for row in rows:
                yield row
                served += 1

            offset += len(rows)
            if offset >= total:
                return

            limit = _next_limit(limit, len(rows), weight, target_bytes)


def text_of(row: dict) -> str | None:
    """The prose of a document, or None if it is a conversation.

    A document carries content XOR messages — the data engine enforces that with
    a CHECK — so this is how a pretrain tells that it was pointed at chat data.
    """
    return row.get("content")


def messages_of(row: dict) -> list | None:
    return row.get("messages")


def sample(
    client: Client,
    selected: Selected,
    share: int,
    *,
    take_each: int,
) -> Iterator[tuple[str, str]]:
    """`share` bytes of one dataset, spread across it, as `(name, text)` chunks.

    The remote counterpart of `bpe._draw`, and it keeps that function's one
    finding: the sample is **strided across the dataset, never taken from the
    front**. A corpus is ordered — by shard, by article, by whatever the pull
    walked — so its first megabytes are one region of it and not a sample of it.
    The Rust engine read FineWeb from 50 of its 5,498 shards that way.

    A stop reads a page at an evenly spaced offset and takes up to `take_each`
    bytes from it, which is a page rather than a document because a stop of 64
    KiB is one book or two hundred snippets and the request should not care
    which. How many documents that is comes from the dataset's own average, so
    the first page at each stop is usually the only one.
    """
    if share <= 0 or selected.taken <= 0:
        return

    stops = max(1, min(selected.taken, share // take_each))
    stride = max(1, selected.taken // stops)
    average = max(1, selected.bytes // max(1, selected.documents))

    spent = 0
    for stop in range(stops):
        if spent >= share:
            return
        offset = stop * stride
        if offset >= selected.taken:
            return

        want = min(take_each, share - spent)
        limit = _clamp(-(-want // average))
        rows, _, _ = client.page(selected.id, offset, limit)

        held: list[str] = []
        weight = 0
        for row in rows:
            text = text_of(row)
            if not text:
                continue
            # The newline that will join this to the last one counts against the
            # budget too. Left out, a stop of many small documents overshoots by
            # one byte each, and the budget is the whole point of the sample.
            room = want - weight - (1 if held else 0)
            if room <= 0:
                break
            held.append(text[:room])
            weight += len(held[-1]) + (1 if len(held) > 1 else 0)
            if weight >= want:
                break

        if held:
            spent += weight
            yield selected.name, "\n".join(held)


def _next_limit(limit: int, rows: int, weight: int, target: int) -> int:
    """How many documents to ask for next, from what the last page weighed.

    Sized off the response's real byte count, so a dataset whose documents grow
    or shrink partway through is tracked rather than assumed. A page that somehow
    weighed nothing leaves the limit alone rather than dividing by it.
    """
    if weight <= 0 or rows <= 0:
        return _clamp(limit)
    per_document = weight / rows
    return _clamp(round(target / per_document))


def _clamp(limit: int) -> int:
    return max(1, min(int(limit), MAX_PAGE_DOCUMENTS))


def _detail(response: httpx.Response) -> str:
    """What the far side said, when it said anything intelligible."""
    try:
        body = response.json()
    except (json.JSONDecodeError, ValueError):
        return response.text[:200]
    if isinstance(body, dict):
        return str(body.get("error") or body.get("detail") or body)[:200]
    return str(body)[:200]
