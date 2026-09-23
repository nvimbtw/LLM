"""`POST /models/{name}/attention`: what it returns, and what it leaves behind.

The last of those is the point of half this file. Recording attention is a flag
on a model the chat cache shares with every other request in the process, so a
request that returns a correct matrix and forgets to put the flag back has moved
every later chat turn off the fused kernel — silently, and for the life of the
process.
"""

import base64
from unittest.mock import patch

import numpy as np
import pytest

from tests.fixtures.tiny_env import TINY_MODEL

UNKNOWN = "no-such-model"

# Long enough to be a matrix rather than a row, short enough for the fixture
# model's 128-token window.
SAMPLE = "the cat sat on the mat. the cat ate the fish."


@pytest.fixture
async def client(engine_client):
    return engine_client


async def _ask(client, **body):
    return await client.post(
        f"/models/{body.pop('model', TINY_MODEL)}/attention",
        json={"text": SAMPLE, "layer": 0, "head": 0, **body},
    )


def _decoded(body: dict) -> np.ndarray:
    assert body["dtype"] == "float32"
    assert body["encoding"] == "base64"
    flat = np.frombuffer(base64.b64decode(body["weights"]), dtype="<f4")
    return flat.reshape(body["shape"])


def _counting_fused_kernel():
    """`F.scaled_dot_product_attention`, still doing its job and counting calls.
    Which layers reach it is the only evidence that the fused path is intact —
    the two paths agree numerically, so the output cannot tell them apart."""
    from llm.model.transformer import F

    return patch.object(
        F, "scaled_dot_product_attention", wraps=F.scaled_dot_product_attention
    )


def _cached_model():
    """The model the request just loaded. `conftest` clears this per test, so it
    is the one that answered and not a leftover."""
    from llm.orchestrate import chat

    return chat._CACHE["model"]


async def test_a_matrix_comes_back_labelled_on_both_axes(client):
    response = await _ask(client)
    assert response.status_code == 200
    body = response.json()

    weights = _decoded(body)
    seq = len(body["tokens"])
    assert weights.shape == (seq, seq)
    assert body["layers"] == 2 and body["heads"] == 4
    # The labels are most of the value: a matrix with no tokens on it is a grey
    # square. Every one is a real string, not an id rendered as one.
    assert all(isinstance(t, str) and t for t in body["tokens"])
    assert "cat" in "".join(body["tokens"])


async def test_rows_sum_to_one_and_the_causal_mask_holds(client):
    body = (await _ask(client, layer=1, head=2)).json()
    weights = _decoded(body).astype(np.float64)

    assert np.allclose(weights.sum(axis=1), 1.0, atol=1e-5)
    # Row i attended to key j, so anything above the diagonal is a token
    # attending to its own future — softmax over -inf, exactly zero.
    assert np.count_nonzero(np.triu(weights, k=1)) == 0


async def test_recording_is_off_again_after_a_request(client):
    assert (await _ask(client, layer=1)).status_code == 200

    model = _cached_model()
    assert not any(b.attn.record_attention for b in model.blocks)
    assert all(b.attn.last_attention is None for b in model.blocks)


async def test_recording_is_off_again_after_a_forward_that_raises(client):
    """The `finally` in `_record_one_head`, exercised. Without it a single
    failed request leaves the flag on for the life of the process."""
    from llm.model.transformer import Attention
    from llm.orchestrate import attention

    with patch.object(
        Attention, "_attention_recording", side_effect=RuntimeError("device fell over")
    ):
        with pytest.raises(RuntimeError):
            attention.matrix(TINY_MODEL, SAMPLE, layer=0, head=0)

    model = _cached_model()
    assert not any(b.attn.record_attention for b in model.blocks)


async def test_only_the_asked_for_layer_records(client):
    """The other layers keep the fused kernel, which is why the response costs
    one matrix rather than one per layer."""
    from llm.orchestrate import attention

    with _counting_fused_kernel() as fused:
        attention.matrix(TINY_MODEL, SAMPLE, layer=1, head=0)

    model = _cached_model()
    assert fused.call_count == model.layers - 1


async def test_the_default_path_still_uses_the_fused_kernel(client):
    """Nothing about this endpoint may leave the ordinary forward on the
    explicit softmax — that path materializes a `[batch, heads, seq, seq]` fp32
    matrix per layer for a reading nobody takes."""
    import torch

    from llm.corpus import bpe
    from llm.orchestrate import attention, chat

    attention.matrix(TINY_MODEL, SAMPLE, layer=0, head=0)
    model, tokenizer, device = chat.load(TINY_MODEL)

    ids = torch.tensor([bpe.encode(tokenizer, SAMPLE)], device=device)
    with _counting_fused_kernel() as fused:
        with torch.no_grad():
            model(ids)

    assert fused.call_count == model.layers


async def test_text_past_the_bound_is_refused(client):
    """The fixture model's window is 128, below `MAX_SEQUENCE`, so the model's
    own ceiling is what binds — and the message has to say which number it is."""
    response = await _ask(client, text=SAMPLE * 40)
    assert response.status_code == 400
    assert "128" in response.json()["detail"]


async def test_empty_text_is_refused(client):
    response = await client.post(
        f"/models/{TINY_MODEL}/attention", json={"text": "", "layer": 0, "head": 0}
    )
    assert response.status_code == 422


@pytest.mark.parametrize("field,value", [("layer", 2), ("head", 4)])
async def test_a_layer_or_head_the_model_does_not_have_is_refused(client, field, value):
    response = await _ask(client, **{field: value})
    assert response.status_code == 400
    assert field in response.json()["detail"]


async def test_an_unknown_model_is_404(client):
    assert (await _ask(client, model=UNKNOWN)).status_code == 404
