"""Built vocabularies: what exists, and what a run may name.

Disk is the authority on what can be loaded and the row is the authority on what
a person calls it, so the listing is the join of the two — the same arrangement
`models.py` has, for the same reason.
"""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

import store
from api import auth
from store import queries as q
from store.schema import Model, Tokenizer

router = APIRouter(tags=["tokenizers"])


class TokenizerEdit(BaseModel):
    """Validated display fields; changing them never rebuilds vocabulary artifacts."""

    model_config = ConfigDict(extra="forbid")
    display_name: str = Field(min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("display_name")
    @classmethod
    def nonblank_name(cls, value: str) -> str:
        """Trim a display name and reject whitespace-only labels."""
        value = value.strip()
        if not value:
            raise ValueError("name cannot be empty")
        return value


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
    """Return vocabulary facts and recorded model references, or 404 for an unknown identity."""
    from llm.corpus import vocab

    built = vocab.info(str(tokenizer_id))
    with store.session() as db:
        row = q.get_tokenizer(db, tokenizer_id)
        if row is None and built is None:
            raise HTTPException(status_code=404, detail=f"no tokenizer {tokenizer_id}")
        return {**_tokenizer(row, built), "models": _using_models(db, tokenizer_id)}


@router.patch("/tokenizers/{tokenizer_id}")
def edit_tokenizer(tokenizer_id: uuid.UUID, body: TokenizerEdit, caller: str = Depends(auth.require_service)):
    """Persist system-authorized display edits atomically; duplicate names return 409.

    Disk-only vocabularies gain an index row. Identity, fingerprint, build provenance
    and model references remain unchanged, and no artifact file is written.
    """
    from llm.corpus import vocab

    if caller != auth.SYSTEM:
        raise HTTPException(status_code=403, detail="tokenizer edits require the system caller")
    built = vocab.info(str(tokenizer_id))
    with store.session() as db:
        row = q.get_tokenizer(db, tokenizer_id)
        if row is None:
            if built is None:
                raise HTTPException(status_code=404, detail=f"no tokenizer {tokenizer_id}")
            row = Tokenizer(id=tokenizer_id, name=body.display_name, implicit=True,
                            vocab_size=built.get("vocab_size"), merges=built.get("merges"),
                            max_vocab=built.get("max_vocab"), fingerprint=built.get("fingerprint"),
                            datasets=built.get("datasets"))
            db.add(row)
        row.name = body.display_name
        if "description" in body.model_fields_set:
            row.description = body.description
        try:
            db.flush()
        except IntegrityError as error:
            raise HTTPException(status_code=409, detail="A vocabulary with that name already exists.") from error
        return {**_tokenizer(row, built), "models": _using_models(db, tokenizer_id)}


def _using_models(db: Session, tokenizer_id: uuid.UUID) -> list[dict]:
    """Resolve recorded vocabulary provenance through at most 25 base/resume ancestors per model."""
    rows = q.list_models(db, include_private=True, checkpoints=True)
    by_id = {str(row.id): row for row in rows}
    return [{"model_id": str(row.id), "name": row.name}
            for row in rows if _uses_tokenizer(row, by_id, str(tokenizer_id))]


def _uses_tokenizer(model: Model | None, by_id: dict[str, Model], tokenizer_id: str) -> bool:
    """Match an indexed model's recorded vocabulary, stopping at missing, cyclic or bounded ancestry."""
    visited = set()
    for _ in range(25):
        if model is None or model.id in visited:
            return False
        visited.add(model.id)
        parameters = model.parameters or {}
        result = parameters.get("result") or {}
        config = parameters.get("config") or {}
        recorded = result.get("tokenizer_id") or config.get("tokenizer_id")
        if recorded:
            return recorded == tokenizer_id
        parent = result.get("base_model") or config.get("base_model") or config.get("resume_from")
        model = by_id.get(parent)
    return False


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
