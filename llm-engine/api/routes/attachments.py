"""Files a person attached to a turn.

Stored, served back, and read by nothing else. The LLM engine holds these because
it holds the messages they hang off — not because a model will ever see one.
The invariant is worth stating twice: no code under `llm/` imports this module
or touches `attachments`, and the only thing `attachment_ids` on a chat request
does is bind rows to the message it creates. See
`docs/decisions/attachments-are-stored-not-read.md`.

Bytes live in Postgres, which only works because of the caps below.
"""

import uuid
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, UploadFile
from fastapi.responses import Response

import store
from api import auth
from store import queries as q
from store.schema import Attachment

router = APIRouter(tags=["attachments"])

#: A row in a table, not a file on a disk — this is the number that keeps a
#: `SELECT` off the wrong side of memory. Mirrored by `MAX_MB` in the backend's
#: route, which refuses an oversized body at the edge.
MAX_BYTES = 50 * 1024 * 1024

#: How many may be bound to a single turn.
#:
#: The product with MAX_BYTES is what actually matters: a turn's worth of files
#: passes whole through three processes, so this came down when the per-file
#: ceiling went up. Five at 50 MB is the same worst case ten at 25 MB would be.
MAX_PER_MESSAGE = 5

#: Read in pieces so an over-sized upload is refused at the point it passes the
#: cap, rather than after the whole of it is in memory.
CHUNK_BYTES = 64 * 1024

UNKNOWN_MEDIA_TYPE = "application/octet-stream"

#: Long enough for any real filename and short enough that the column holds it.
MAX_FILENAME = 255


@router.post("/attachments", status_code=201)
async def upload(file: UploadFile, owner: uuid.UUID = Depends(auth.require_owner)):
    content = await _read_capped(file)
    with store.session() as db:
        stored = q.create_attachment(
            db,
            owner_id=owner,
            filename=_filename(file.filename),
            media_type=file.content_type or UNKNOWN_MEDIA_TYPE,
            content=content,
        )
        return describe(stored)


@router.get("/attachments/{attachment_id}")
def download(attachment_id: uuid.UUID, owner: uuid.UUID = Depends(auth.require_owner)):
    with store.session() as db:
        attachment = q.owned_attachment(db, attachment_id, owner)
        if attachment is None:
            raise HTTPException(status_code=404, detail=f"no attachment {attachment_id}")
        return Response(
            content=attachment.content,
            media_type=attachment.media_type,
            headers={"Content-Disposition": _disposition(attachment.filename)},
        )


@router.delete("/attachments/{attachment_id}", status_code=204)
def discard(attachment_id: uuid.UUID, owner: uuid.UUID = Depends(auth.require_owner)):
    with store.session() as db:
        if not q.delete_attachment(db, attachment_id, owner):
            raise HTTPException(
                status_code=404, detail=f"no unsent attachment {attachment_id}"
            )


def describe(attachment: Attachment) -> dict:
    """The metadata half of an attachment. Every route that mentions one returns
    exactly this — the bytes are only ever served by `download`."""
    return {
        "attachment_id": str(attachment.id),
        "filename": attachment.filename,
        "media_type": attachment.media_type,
        "size": attachment.size,
        "created_at": attachment.created_at.isoformat() if attachment.created_at else None,
    }


async def _read_capped(file: UploadFile) -> bytes:
    """The whole upload, or a 413 the moment it grows past the cap.

    Bounded by that check: each iteration either adds a chunk under the ceiling
    or raises.
    """
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(CHUNK_BYTES):
        total += len(chunk)
        if total > MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"an attachment may not exceed {MAX_BYTES // (1024 * 1024)} MB",
            )
        chunks.append(chunk)
    if total == 0:
        raise HTTPException(status_code=400, detail="an empty file is not an attachment")
    return b"".join(chunks)


def _filename(raw: str | None) -> str:
    """A label, never a path.

    The name a browser sends is a stranger's string that this side stores and
    hands back in a header, so the separators are stripped here rather than
    trusted not to matter to whatever reads it later.
    """
    name = (raw or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    name = "".join(c for c in name if c.isprintable())
    return name[:MAX_FILENAME] or "file"


def _disposition(filename: str) -> str:
    """RFC 5987, because a filename is not necessarily ASCII and a raw one would
    make the header unparseable rather than merely ugly."""
    return f"inline; filename*=UTF-8''{quote(filename, safe='')}"
