"""Stateless SSE; transcript commit behavior is covered by Rust tool_rounds."""
import pytest
from tests.test_routes import _chat_body, _sse


async def test_stateless_stream_returns_usage_and_complete_text(engine_client):
    async with engine_client.stream('POST','/chat/stream',json=_chat_body()) as response:
        frames = _sse(''.join([chunk async for chunk in response.aiter_text()]))
    done = next(frame for frame in frames if frame['event']=='done')
    assert done['data']['generated_tokens'] >= 0
    assert isinstance(done['data']['text'],str)
