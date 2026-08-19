"""Browsing the Hub. Read-only, and the only routes that reach the network."""

from fastapi import APIRouter, HTTPException, Query

import hub

router = APIRouter(tags=["hub"])


@router.get("/hub/datasets")
def search(q: str | None = Query(default=None), limit: int = Query(default=20)):
    try:
        return {"results": hub.search(q, limit)}
    except hub.HubError as e:
        # 502, not 500: the Hub failed, not this service, and the difference is
        # what tells the user whether retrying is worth anything.
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/hub/datasets/detail")
def detail(repo: str = Query(...)):
    """Kept off the path because a repo id contains a slash, and a path template
    that swallows one is a route that breaks on `owner/name`."""
    try:
        return hub.detail(repo)
    except hub.HubError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/hub/datasets/sample")
def sample(
    repo: str = Query(...),
    config: str | None = Query(default=None),
    split: str | None = Query(default=None),
    rows: int = Query(default=5),
):
    """The first few rows, so the pull form is filled in by looking rather than
    by guessing. Off the path for the same reason `detail` is."""
    try:
        return hub.sample(repo, config, split, rows)
    except hub.HubError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
