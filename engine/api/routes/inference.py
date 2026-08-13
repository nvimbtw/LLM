"""Generation with nothing stored.

The same model and the same sampler as `/chats`, without a conversation to
belong to — this is what an evaluation harness or a playground calls.
"""

import asyncio

from fastapi import APIRouter, HTTPException
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
        while True:
            try:
                piece, summary = await asyncio.to_thread(advance, generator)
            except (FileNotFoundError, ValueError) as e:
                yield sse.token_frame("error", {"error": str(e)})
                return
            if summary is not None:
                yield sse.token_frame("done", summary)
                return
            yield sse.token_frame("token", piece)

    return StreamingResponse(frames(), media_type=sse.MEDIA_TYPE, headers=sse.HEADERS)


@router.post("/tokenize/count")
def count_tokens(req: CountRequest):
    """Token length of `text` under a model's tokenizer. Loads no weights."""
    try:
        return chat_pipeline.count_tokens(req.model_name, req.text)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
