"""The cache a `web-search` tool call checks before paying for a live search.

One dataset (`store.queries.CACHE_DATASET_NAME`), created on first use,
`owner_id=None` because a search result is a fact about the world and not
about whoever happened to ask first. A re-fetch appends a fresh document
rather than overwriting the old one — the same shape a re-pulled Hub dataset
already has — so `latest_by_key` is what answers "what do we currently
believe," and nothing here ever mutates a row in place.

Freshness is the caller's call, not this route's: a lookup returns the newest
match and when it was written, and the backend decides whether that is fresh
enough against its own `SEARCH_CACHE_MAX_AGE_DAYS`, which can change without
redeploying this service.
"""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

import store
from api import auth
from store import queries as q

router = APIRouter(tags=["search-cache"])

#: A cache key is a normalized query, not a document — matches the ceiling the
#: tools crate already puts on how long a search argument may run
#: (`MAX_QUERY_CHARS`), with room for normalization to not overflow it.
MAX_KEY_CHARS = 500


class LookupRequest(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_CHARS)


class StoreRequest(BaseModel):
    key: str = Field(min_length=1, max_length=MAX_KEY_CHARS)
    content: str = Field(min_length=1)


def _view(row: store.Document | None) -> dict:
    if row is None:
        return {"content": None, "created_at": None}
    return {"content": row.content, "created_at": row.created_at.isoformat()}


@router.post("/search-cache/lookup")
def lookup(body: LookupRequest, _: str = Depends(auth.require_service)):
    """The freshest cached result for `key`, or an empty hit if there is none.

    A POST rather than a GET with a query string: `key` is arbitrary,
    possibly long, possibly unicode text, and a body is the shape that does
    not need percent-encoding on the way in. A miss is not a 404: absence is
    the ordinary outcome of a cache that has never been asked this before,
    not an error about a resource that should exist.
    """
    with store.session() as db:
        dataset = q.dataset_by_name(db, q.CACHE_DATASET_NAME)
        row = q.latest_by_key(db, dataset.id, body.key) if dataset is not None else None
        return _view(row)


@router.post("/search-cache", status_code=201)
def remember(body: StoreRequest, _: str = Depends(auth.require_service)):
    """Record a search result under `key`, creating the cache dataset the
    first time anything is written to it."""
    with store.session() as db:
        dataset = q.cache_dataset(db)
        row = q.add_document(db, dataset.id, cache_key=body.key, content=body.content)
        return _view(row)
