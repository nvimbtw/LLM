"""What one person's conversations have cost, in tokens.

Counted from the turns themselves. There is no usage ledger to keep in step
with them, so the number on the page and the number in the chat cannot disagree.
"""

import uuid
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, Query

import store
from api import auth
from store import queries as q

router = APIRouter(tags=["usage"])


@router.get("/usage")
def usage(
    days: int = Query(default=30, ge=1, le=365),
    owner: uuid.UUID = Depends(auth.require_owner),
):
    since = datetime.now(timezone.utc) - timedelta(days=days)
    with store.session() as db:
        series = q.usage_by_day(db, owner, since)
        models = q.usage_by_model(db, owner, since)

    return {
        "days": days,
        "series": [
            {
                "date": str(day),
                "asked": int(asked),
                "generated": int(generated),
                "replies": int(replies),
            }
            for day, asked, generated, replies in series
        ],
        "by_model": [
            {
                "model_id": str(model_id) if model_id else None,
                "name": name,
                "tokens": int(tokens),
                "replies": int(replies),
            }
            for model_id, name, tokens, replies in models
        ],
    }
