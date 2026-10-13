"""Generation with nothing stored.

The same model and the same sampler as `/chats`, without a conversation to
belong to — this is what an evaluation harness or a playground calls.
"""

import asyncio
import anyio
import uuid
import store
from store import queries as q
from api.chat_request import Ask, _new_turns
from llm.config import ChatTurn
from llm.model import device

from fastapi import APIRouter, HTTPException
from pydantic import ValidationError
from fastapi.responses import StreamingResponse

from api import sse
from llm.config import ChatConfig, CountRequest
from llm.orchestrate import chat as chat_pipeline

router = APIRouter(tags=["inference"])


@router.post("/chat")
def chat(cfg: ChatConfig):
    """Generate the next assistant turn for a conversation the caller holds."""
    try:
        return chat_pipeline.run(cfg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except device.Busy as e:
        raise HTTPException(status_code=409, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/chat/stream")
async def chat_stream(cfg: ChatConfig):
    """The same turn, streamed token by token as SSE."""

    def advance(generator):
        try:
            return next(generator), None
        except StopIteration as done:
            return None, done.value or {}

    async def frames():
        generator = chat_pipeline.stream(cfg)
        pending = None
        try:
            while True:
                try:
                    pending = asyncio.create_task(asyncio.to_thread(advance, generator))
                    piece, summary = await asyncio.shield(pending)
                except (FileNotFoundError, ValueError) as e:
                    yield sse.token_frame("error", {"error": str(e)})
                    return
                if summary is not None:
                    yield sse.token_frame("done", summary)
                    return
                yield sse.token_frame(*piece)
        finally:
            # A disconnect does not stop a worker already inside next(). Wait
            # for that token boundary before closing and releasing admission.
            # Starlette cancels its AnyIO scope on disconnect. Shield cleanup
            # from that scope so next() cannot still be running during close().
            with anyio.CancelScope(shield=True):
                try:
                    if pending is not None and not pending.done():
                        await asyncio.shield(pending)
                finally:
                    generator.close()

    return StreamingResponse(frames(), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS)


@router.post("/tokenize/count")
def count_tokens(req: CountRequest):
    """Token length of `text` under a model's tokenizer. Loads no weights."""
    try:
        return chat_pipeline.count_tokens(req.model_name, req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/chat/prepare")
def prepare_turn(body: "Ask"):
    """Validate, render and count turns before Rust stores them."""
    turns = _new_turns(body)
    try:
        ChatConfig(**body.model_dump(exclude={"text", "tool_result", "interject", "attachment_ids", "history_floor"}),
                   turns=[ChatTurn(role=role, text=text) for role, text in turns])
    except ValidationError as error:
        raise HTTPException(status_code=422, detail=str(error))
    try:
        counted = [{"role": role, "text": text,
                    "tokens": chat_pipeline.count_tokens(body.model_name, text)["tokens"]}
                   for role, text in turns]
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
    with store.session() as db:
        try:
            model = q.get_model(db, uuid.UUID(body.model_name))
        except ValueError:
            model = q.find_model_by_name(db, body.model_name)
        if model and model.checkpoint_index is not None:
            model.parameters = {**(model.parameters or {}), "recovery_referenced": True}
        return {"turns": counted, "model_id": str(model.id) if model else None,
                "model_name": model.name if model else body.model_name}


@router.post("/chat/context-floor")
def context_floor(body: ChatConfig):
    try:
        with device.admit():
            return {"start": chat_pipeline.history_floor(body.model_name, body.turns)}
    except device.Busy as error:
        raise HTTPException(status_code=409, detail=str(error))
    except FileNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error))
