"""Built vocabularies: what exists, and what a run may name.

Disk is the authority on what can be loaded and the row is the authority on what
a person calls it, so the listing is the join of the two — the same arrangement
`models.py` has, for the same reason.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException

import store
from api import auth
from store import queries as q
from store.schema import Tokenizer

router = APIRouter(tags=["tokenizers"])


@router.get("/tokenizers")
def list_tokenizers(include_implicit: bool = True):
    """Every vocabulary, whether it was built by name or derived by a run.

    `has_vocab` is false for a row whose directory has gone: it stays listed, so
    the models that name it are explicable, but nothing should offer it.
    """
    from llm.corpus import vocab

    built = {b["tokenizer_id"]: b for b in vocab.catalogue()}
    with store.session() as db:
        rows = q.list_tokenizers(db, include_implicit=include_implicit)
        listed = [_tokenizer(row, built.pop(str(row.id), None)) for row in rows]
    # A directory with no row is a build made outside the queue — by a script, or
    # by a run whose registration failed. It can still be trained against.
    listed += [_tokenizer(None, build) for build in built.values()]
    return {"tokenizers": listed}


@router.get("/tokenizers/{tokenizer_id}")
def get_tokenizer(tokenizer_id: uuid.UUID):
    from llm.corpus import vocab

    built = vocab.info(str(tokenizer_id))
    with store.session() as db:
        row = q.get_tokenizer(db, tokenizer_id)
    if row is None and built is None:
        raise HTTPException(status_code=404, detail=f"no tokenizer {tokenizer_id}")
    return _tokenizer(row, built)


@router.delete("/tokenizers/{tokenizer_id}")
def delete_tokenizer(tokenizer_id: uuid.UUID, _=Depends(auth.require_service)):
    """Forget a vocabulary, and the corpora that only make sense under it.

    The corpora go too: they are keyed on this vocabulary's fingerprint, so
    without it they are gigabytes of ids nothing can read. Models keep their own
    copy of the vocabulary and are unaffected.
    """
    from llm.corpus import pack, vocab

    built = vocab.info(str(tokenizer_id))
    with store.session() as db:
        name = q.delete_tokenizer(db, tokenizer_id)
    if name is None and built is None:
        raise HTTPException(status_code=404, detail=f"no tokenizer {tokenizer_id}")

    fingerprint = (built or {}).get("fingerprint")
    corpora = [
        build["key"]
        for build in pack.builds()
        if build["tokenizer_id"] == str(tokenizer_id)
        or (fingerprint and build["tokenizer_fingerprint"] == fingerprint)
    ]
    for key in corpora:
        pack.delete(key)
    vocab.delete(str(tokenizer_id))
    return {"tokenizer_id": str(tokenizer_id), "deleted": True, "corpora_deleted": len(corpora)}


def _tokenizer(row: Tokenizer | None, built: dict | None) -> dict:
    """One vocabulary as the API reports it: the row where there is one, filled
    in from the build on disk where there is not."""
    tokenizer_id = str(row.id) if row is not None else built["tokenizer_id"]
    return {
        "tokenizer_id": tokenizer_id,
        "name": row.name if row is not None else f"auto {tokenizer_id[:8]}",
        "description": row.description if row is not None else None,
        "implicit": row.implicit if row is not None else True,
        "vocab_size": (built or {}).get("vocab_size") or (row.vocab_size if row else None),
        "merges": (built or {}).get("merges") or (row.merges if row else None),
        "max_vocab": (built or {}).get("max_vocab") or (row.max_vocab if row else None),
        "fingerprint": (built or {}).get("fingerprint") or (row.fingerprint if row else None),
        "datasets": (built or {}).get("datasets") or (row.datasets if row else None) or [],
        "files": (built or {}).get("files"),
        "bytes": (built or {}).get("bytes", 0),
        "built_at": (built or {}).get("built_at"),
        "created_at": row.created_at.isoformat() if row is not None and row.created_at else None,
        "has_vocab": built is not None,
    }
