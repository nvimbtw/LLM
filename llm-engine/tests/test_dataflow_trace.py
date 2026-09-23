"""The dataflow contract: one fixed question, five recorded checkpoints.

Asserting only on the final reply would let two compensating changes pass. A
checkpoint per layer boundary localizes a break to the layer that caused it.

Re-record with `python -m tests.fixtures.record` after a deliberate change.
"""

import json

import pytest

from llm.config import ChatConfig, ChatTurn
from llm.orchestrate import chat as chat_pipeline
from tests.fixtures import trace_spec
from tests.fixtures.record import TRACE
from tests.fixtures.tiny_env import TINY_MODEL


@pytest.fixture(scope="module")
def recorded() -> dict:
    if not TRACE.is_file():
        pytest.fail(f"no recorded trace at {TRACE}; run tests.fixtures.record")
    return json.loads(TRACE.read_text())


@pytest.fixture(scope="module")
def actual(model_module, tokenizer_module) -> dict:
    return trace_spec.checkpoints(model_module, tokenizer_module)


def test_tokenizer_ids_match_trace(actual, recorded):
    assert actual["question_ids"] == recorded["question_ids"]


def test_rendered_prompt_matches_trace(actual, recorded):
    assert actual["prompt_ids"] == recorded["prompt_ids"]


def test_logits_match_trace(actual, recorded):
    assert actual["logits"]["shape"] == recorded["logits"]["shape"]
    assert actual["logits"]["argmax"] == recorded["logits"]["argmax"]
    for (got_id, got), (want_id, want) in zip(
        actual["logits"]["top"], recorded["logits"]["top"]
    ):
        assert got_id == want_id
        assert got == pytest.approx(want, abs=1e-3)


def test_greedy_generation_matches_trace(actual, recorded):
    """Temperature 0 removes the sampler, isolating a change in the model."""
    assert actual["greedy"] == recorded["greedy"]


def test_seeded_sampling_matches_trace(actual, recorded):
    assert actual["sampled"] == recorded["sampled"]


def test_stream_matches_trace(recorded):
    """The pipeline end to end, through the same seam the route calls."""
    cfg = ChatConfig(
        model_name=TINY_MODEL,
        turns=[ChatTurn(role="user", text=trace_spec.QUESTION)],
        system=trace_spec.SYSTEM,
        max_tokens=trace_spec.MAX_TOKENS,
        temperature=0.8,
        top_k=40,
        top_p=0.95,
        seed=trace_spec.SEED,
    )

    pieces = []
    stream = chat_pipeline.stream(cfg)
    while True:
        try:
            kind, data = next(stream)
        except StopIteration as done:
            summary = done.value
            break
        if kind == "token":
            pieces.append(data["token"])

    assert summary["text"] == recorded["sampled"]["text"]
    assert "".join(pieces) == recorded["sampled"]["text"]
    assert summary["prompt_tokens"] == len(recorded["prompt_ids"])
