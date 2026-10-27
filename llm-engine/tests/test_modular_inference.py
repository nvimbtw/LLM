"""Actual generic likelihood/continuation behavior, context bounds and request-state ownership."""

import threading

import pytest
import torch

from llm.corpus import bpe, special
from llm.corpus.baseline import byte_alphabet
from llm.model import continuation, likelihood, sample
from llm.model.decoding import PrefixDecoder
from llm.model.language import logits, evaluating
from llm.model import generation
from llm.config import ChatConfig, ChatTurn
from tests.test_modular_rl_policy import policy_model


@pytest.mark.parametrize("name", ["legacy", "standard", "looped", "nested", "custom"])
def test_likelihood_matches_independent_strided_prefixes_and_preserves_modes_rng(name):
    """Each token is scored once using the existing half-window overlap, even for shared/dropout custom models."""
    tokens = [index % 16 for index in range(27)]
    with policy_model(name) as model:
        model.train()
        list(model.modules())[-1].eval()
        modes = tuple(module.training for module in model.modules())
        rng = torch.get_rng_state().clone()
        expected = []
        with torch.no_grad(), evaluating(model):
            for target in range(1, len(tokens)):
                start = max(0, ((target - 4) // 4) * 4)
                while target >= start + 8:
                    start += 4
                values = logits(model, torch.tensor([tokens[start:target]]))
                expected.append(float(values[0, -1].float().log_softmax(-1)[tokens[target]]))
        actual = likelihood.token_logprobs(model, tokens)
        assert actual.count == len(tokens) - 1
        assert actual.logprobs == pytest.approx(expected, abs=2e-6)
        assert tuple(module.training for module in model.modules()) == modes
        assert torch.equal(torch.get_rng_state(), rng)
        assert all(parameter.grad is None for parameter in model.parameters())


@pytest.mark.parametrize("name", ["legacy", "standard", "looped", "nested", "custom"])
def test_seeded_continuation_matches_independent_full_prefix_sampling(name):
    """Legacy cached and generic full-prefix generation keep sampling, tokenizer and trained-context semantics."""
    tokenizer = byte_alphabet()
    with policy_model(name, vocab=320, context=12) as model:
        model.train()
        modes = tuple(module.training for module in model.modules())
        rng = torch.get_rng_state().clone()
        cfg = continuation.SamplingConfig(max_tokens=20, temperature=0.8, top_k=40, top_p=0.95, seed=19)
        ids = bpe.encode(tokenizer, "abc")
        produced = []
        generator = torch.Generator().manual_seed(cfg.seed)
        with torch.no_grad(), evaluating(model):
            for _ in range(12 - len(ids)):
                values = logits(model, torch.tensor([ids + produced]))
                chosen = sample.pick(values[0, -1], cfg.temperature, cfg.top_k, cfg.top_p, generator)
                if chosen == bpe.token_id(tokenizer, special.END_OF_TEXT):
                    break
                produced.append(chosen)
        actual = continuation.continue_tokens(model, tokenizer, "abc", cfg)
        assert actual == bpe.decode(tokenizer, produced)
        assert tuple(module.training for module in model.modules()) == modes
        assert torch.equal(torch.get_rng_state(), rng)


def test_cancelled_measurements_do_not_return_unearned_scores_or_continuations():
    """Already-set cancellation yields no token work and scoring retains its empty-result semantics."""
    cancelled = threading.Event()
    cancelled.set()
    with policy_model("custom", vocab=320, context=12) as model:
        assert likelihood.token_logprobs(model, [1, 2, 3], cancel=cancelled).count == 0
        assert continuation.continue_tokens(model, byte_alphabet(), "abc", continuation.SamplingConfig(), cancel=cancelled) == ""


def test_decoder_releases_prefix_and_cache_on_error_and_rejects_foreign_inputs():
    """A request owns every prefix/cache reference and invalid IDs cannot reach the custom forward."""
    with policy_model("standard") as model, torch.no_grad(), evaluating(model):
        tokens = torch.tensor([[1, 2]])
        positions = torch.tensor([[0, 1]])
        live = torch.ones_like(tokens, dtype=torch.bool)
        with pytest.raises(RuntimeError, match="consumer stopped"):
            with PrefixDecoder(model, tokens, positions, live, window=8) as decoder:
                decoder.advance(torch.tensor([3]))
                raise RuntimeError("consumer stopped")
        assert decoder.model is decoder.tokens is decoder.values is decoder.cache is None
        with pytest.raises(ValueError, match="vocabulary"):
            PrefixDecoder(model, tokens + 16, positions, live, window=8)
        with pytest.raises(ValueError, match="aligned"):
            PrefixDecoder(model, tokens, positions.float(), live, window=8)
        with pytest.raises(ValueError, match="context"):
            PrefixDecoder(model, tokens, positions, live, window=9)


def test_actual_generation_flushes_a_final_incomplete_utf8_fragment(monkeypatch):
    """A draw ending midway through a UTF-8 character still emits exactly the summary's replacement character at stream completion."""
    tokenizer = byte_alphabet()
    lead = tokenizer.token_to_id("Ã")
    assert lead is not None
    monkeypatch.setattr(generation.sample, "pick", lambda *args, **kwargs: lead)
    with policy_model("custom", vocab=320, context=128) as model:
        cfg = ChatConfig(model_name="fixture", turns=[ChatTurn(role="user", text="one")], max_tokens=1)
        frames = []
        stream = generation.stream_model(cfg, model, tokenizer, "cpu")
        try:
            while True:
                frames.append(next(stream))
        except StopIteration as done:
            summary = done.value
        finally:
            stream.close()
        assert summary["generated_tokens"] == 1
        assert "".join(data["token"] for kind, data in frames if kind == "token") == summary["text"] == "�"
