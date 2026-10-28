"""Real private SFT rendering, exact aligned-output budgets and partial-file cleanup."""

import json
from uuid import uuid4

import numpy as np
import pytest
from tokenizers import Tokenizer

from llm import storage
from llm.chat.template import ASSISTANT_CLOSE, Reasoning
from llm.corpus import bpe, sft
from llm.corpus.baseline import byte_alphabet
from llm.corpus.pack import PackBudget
from llm.train.sft_tokenizer import prepare_tokenizer
from llm.architecture.tokenizer_binding import binding_from_bytes
from tests.fixtures.tiny_env import selected
from tests.test_sft_tokenizer import legacy_bytes


def conversations():
    """Supply two real user/assistant examples in the renderer's canonical input format."""
    return [{"system": None, "turns": [{"role": "user", "text": "question"},
             {"role": "assistant", "text": f"answer {index}"}]} for index in range(2)]


def output_bytes(directory):
    """Read complete private renderer artifacts so refused rebuilds must preserve all earlier bytes."""
    return {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}


def test_private_rendering_matches_existing_cache_tokens_masks_and_metadata(tmp_path, monkeypatch):
    """The same actual renderer produces byte-identical artifacts without changing the global data root."""
    tokenizer = byte_alphabet()
    selection = [selected("private-sft-fixture", documents=2)]
    monkeypatch.setattr(sft, "read_conversations", lambda *args: iter(conversations()))
    stats = sft.build(None, tokenizer, selection)
    key = sft.dataset_key(selection, Reasoning.KEEP)
    prior = output_bytes(storage.sft_dataset_dir(key))
    directory = tmp_path / "private"
    exact = PackBudget(max_corpus_bytes=stats.total_tokens * 3, max_documents=stats.conversations)
    assert sft.build(None, tokenizer, selection, directory=directory, budget=exact) == stats
    assert output_bytes(directory) == prior
    assert output_bytes(storage.sft_dataset_dir(key)) == prior


@pytest.mark.parametrize("refusal", ["bytes", "documents", "cancel", "source"])
def test_refused_private_render_removes_partials_and_keeps_prior_complete_files(tmp_path, monkeypatch, refusal):
    """Exact byte/row limits and failures after a flushed batch cannot replace earlier complete data."""
    tokenizer = byte_alphabet()
    selection = [selected("bounded-sft-fixture", documents=2)]
    monkeypatch.setattr(sft, "read_conversations", lambda *args: iter(conversations()))
    directory = tmp_path / "private"
    stats = sft.build(None, tokenizer, selection, directory=directory)
    prior = output_bytes(directory)
    budget = PackBudget(max_corpus_bytes=stats.total_tokens * 3 - (refusal == "bytes"),
                        max_documents=1 if refusal == "documents" else 2)
    monkeypatch.setattr(sft, "ENCODE_BATCH_CHARS", 1)
    monkeypatch.setattr(sft, "FLUSH_TOKENS", 1)
    calls = 0

    def cancelled():
        """Cancel only after the first real batch has reached the partial output files."""
        nonlocal calls
        calls += 1
        if refusal == "cancel" and calls >= 3:
            raise InterruptedError("fixture render cancellation")

    def failed_source(*args):
        """Fail an actual input iterator after one complete encoded/flushed conversation."""
        yield conversations()[0]
        raise RuntimeError("fixture source failure")

    if refusal == "source":
        monkeypatch.setattr(sft, "read_conversations", failed_source)
    expected = InterruptedError if refusal == "cancel" else RuntimeError if refusal == "source" else ValueError
    with pytest.raises(expected):
        sft.build(None, tokenizer, selection, directory=directory, budget=budget, check_cancelled=cancelled)
    assert output_bytes(directory) == prior
    assert not list(directory.glob("*.partial"))


def test_bounded_conversation_refuses_before_tokenizer_allocation(tmp_path, monkeypatch):
    """A single oversized source cannot force encoding merely because the streamed output is bounded."""
    raw = conversations()[0]
    raw["turns"][1]["text"] = "a" * 4096
    monkeypatch.setattr(sft, "read_conversations", lambda *args: iter([raw]))

    def forbidden(*args):
        """Detect any encoding allocation before the source character ceiling is applied."""
        pytest.fail("oversized conversation reached encoding")

    monkeypatch.setattr(sft.template, "encode_examples", forbidden)
    directory = tmp_path / "private"
    with pytest.raises(ValueError, match="character budget"):
        sft.build(None, byte_alphabet(), [], directory=directory,
                  budget=PackBudget(max_corpus_bytes=128, max_documents=1))
    assert output_bytes(directory) == {}


def test_old_placeholder_vocabulary_records_memory_ids_and_preserves_input_only_targets(tmp_path, monkeypatch):
    """Renderer-owned private adoption reports the actual memory IDs even when the base still names placeholders."""
    raw = legacy_bytes()
    tokenizer = Tokenizer.from_str(raw.decode())
    example = conversations()[0]
    example["turns"].insert(0, {"role": "memory", "text": json.dumps({
        "memory": "the source fact", "thread": str(uuid4()), "message": str(uuid4()),
        "role": "assistant", "age_seconds": 1})})
    monkeypatch.setattr(sft, "read_conversations", lambda *args: iter([example]))
    selection = [selected("memory-fixture", documents=1)]
    directory = tmp_path / "private"
    stats = sft.build(None, tokenizer, selection, directory=directory,
                      budget=PackBudget(max_corpus_bytes=1 << 20, max_documents=1))
    assert {47, 48}.issubset(stats.special_ids_seen)
    derived_raw, derived = prepare_tokenizer(raw, binding_from_bytes(uuid4(), raw), stats.special_ids_seen)
    assert derived.derived_from is not None and tokenizer.to_str().encode() == raw
    corpus = np.fromfile(storage.sft_corpus_path("unused", directory=directory), dtype=np.uint16)
    mask = np.fromfile(storage.sft_mask_path("unused", directory=directory), dtype=np.uint8)
    targets = [int(corpus[index + 1]) for index in range(len(corpus) - 1) if mask[index]]
    assert bpe.decode(Tokenizer.from_str(derived_raw.decode()), targets) == "answer 0" + ASSISTANT_CLOSE
