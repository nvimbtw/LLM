"""Actual eval-with-grad policy updates for standard, shared-loop and administrator-source dropout models."""

import copy
from contextlib import contextmanager
from uuid import uuid4

import pytest
import torch
from torch.nn import functional as F

from llm.architecture.build import build_model
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader
from llm.architecture.spec import FinalTokenizer
from llm.model.language import logits
from llm.train import policy
from llm.train.model_policy import optimizer_groups
from tests.test_architecture_builtins import build_template, legacy
from tests.test_architecture_core import revision
from tests.test_modular_training_loop import VALUES
from tests.test_policy import _roll
from tests.test_token_rows import ROW_SOURCE, row_recipe
from llm.train.rollout import padded
from llm.model.decoding import PrefixDecoder


@contextmanager
def policy_model(name, *, vocab=16, context=8):
    """Retain actual custom source imports through the numerical lifetime, or construct one supplied recipe."""
    values = {**VALUES, "context_window": context}
    if name == "legacy":
        yield legacy(values, vocab=vocab)
        return
    if name != "custom":
        yield build_template(name, values, vocab=vocab)
        return
    source = ROW_SOURCE.replace("self.readout = nn.Linear(width, vocab, bias=False)",
        "self.readout = nn.Linear(width, vocab, bias=False)\n        self.noise = nn.Dropout(0.75)")
    source = source.replace("self.readout(self.lookup(tokens))", "self.readout(self.noise(self.lookup(tokens)))")
    with SourceLoader() as loader:
        block = loader.load(revision(source))
        recipe = row_recipe(block.reference).model_copy(update={"context_window": context})
        tokenizer = FinalTokenizer(kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=vocab)
        yield build_model(resolve_recipe(recipe, {}, {block.reference.revision_id: block}, tokenizer))


def scored_rolls():
    """Supply two real ragged token/action sequences with context and drawn targets in one nonzero-advantage group."""
    first, second = _roll(0, score=1.0), _roll(1, score=0.0)
    first.ids, first.policy = [1, 2, 3, 4, 5], [False, False, True, False, True]
    second.ids, second.policy = [1, 4, 5], [False, True, True]
    rolls = [first, second]
    return rolls, policy.score(rolls)


def independent_loss(model, reference, rolls, batch, beta):
    """Compute unpadded per-episode unconstrained next-token objectives independently of the batched production adapter."""
    weight = sum(sum(roll.policy) for roll in rolls)
    objective = torch.zeros(())
    for roll in rolls:
        ids = torch.tensor([roll.ids])
        target = ids[:, 1:]
        logp = -F.cross_entropy(logits(model, ids)[:, :-1].reshape(-1, 16), target.reshape(-1), reduction="none")
        with torch.no_grad():
            base = -F.cross_entropy(logits(reference, ids)[:, :-1].reshape(-1, 16), target.reshape(-1), reduction="none")
        mask = torch.tensor(roll.policy[1:], dtype=torch.float32)
        ratio = (base - logp).clamp(max=policy.MAX_LOG_RATIO)
        objective = objective + ((-logp * batch.advantages[roll.episode.id] + beta * (ratio.exp() - ratio - 1)) * mask).sum()
    return objective / weight


@pytest.mark.parametrize("name", ["standard", "looped", "custom"])
@pytest.mark.parametrize("micro_batch", [1, 2])
def test_eval_policy_recomputation_matches_unpadded_updates_without_dropout_rng(name, micro_batch):
    """Sampling's evaluation policy retains gradients; left padding, context flags and micro-batching preserve the independent objective."""
    with policy_model(name) as actual:
        actual.train()
        expected = copy.deepcopy(actual).eval()
        reference = copy.deepcopy(actual).eval().requires_grad_(False)
        with torch.no_grad():
            next(reference.parameters()).add_(0.1)
        rolls, batch = scored_rolls()
        expected_loss = independent_loss(expected, reference, rolls, batch, 0.02)
        expected_loss.backward()
        modes = tuple(module.training for module in actual.modules())
        rng = torch.get_rng_state().clone()
        measured = policy.accumulate(actual, reference, rolls, batch, beta=0.02, micro_batch=micro_batch)
        assert torch.equal(torch.get_rng_state(), rng)
        assert tuple(module.training for module in actual.modules()) == modes
        assert all(value.grad is None for value in reference.parameters())
        assert measured.tokens == 4 and measured.kl >= 0
        assert measured.loss + measured.kl == pytest.approx(float(expected_loss.detach()), abs=2e-6)
        assert any(value.grad is not None and value.grad.any() for value in actual.parameters())
        for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-6)
        for model in (actual, expected):
            optimizer = torch.optim.AdamW(optimizer_groups(model, weight_decay=0.1), lr=0.001)
            optimizer.step()
        for left, right in zip(actual.parameters(), expected.parameters(), strict=True):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-6)


def test_failed_policy_recomputation_restores_heterogeneous_modes(monkeypatch):
    """A failed eval-with-grad update restores child modes and propagates its model error without advancing RNG."""
    with policy_model("custom") as model:
        model.train()
        next(iter(model.instances.values())).lookup.eval()
        modes = tuple(module.training for module in model.modules())
        rng = torch.get_rng_state().clone()
        rolls, batch = scored_rolls()

        def failed(active_model, *args):
            """Observe evaluation behavior with autograd enabled before raising a numerical error."""
            assert torch.is_grad_enabled() and not any(module.training for module in active_model.modules())
            raise RuntimeError("policy forward failed")

        monkeypatch.setattr(policy, "_logprobs", failed)
        with pytest.raises(RuntimeError, match="policy forward failed"):
            policy.accumulate(model, None, rolls, batch, beta=0.0, micro_batch=1)
        assert tuple(module.training for module in model.modules()) == modes
        assert torch.equal(torch.get_rng_state(), rng)


@pytest.mark.parametrize("invalid", ["length", "initial", "flag", "micro", "beta"])
def test_policy_update_refuses_invalid_action_provenance_before_forward(monkeypatch, invalid):
    """Invalid masks and update bounds cannot reach an administrator's forward or allocate padded batches."""
    with policy_model("custom") as model:
        rolls, batch = scored_rolls()
        if invalid == "length":
            rolls[0].policy.pop()
        elif invalid == "initial":
            rolls[0].policy[0] = True
        elif invalid == "flag":
            rolls[0].policy[2] = 1

        def forbidden(*args, **kwargs):
            """Detect model work after an invalid interpreted action mask or numerical option."""
            pytest.fail("invalid action provenance reached forward")

        monkeypatch.setattr(policy, "_logprobs", forbidden)
        with pytest.raises(ValueError):
            policy.accumulate(model, None, rolls, batch, beta=float("nan") if invalid == "beta" else 0.0,
                micro_batch=0 if invalid == "micro" else 1)


def test_a_scored_batch_with_no_drawn_targets_does_not_execute_a_policy_forward(monkeypatch):
    """Rewards alone cannot authorize an update when no causal model-produced targets were retained."""
    with policy_model("custom") as model:
        rolls, batch = scored_rolls()
        for roll in rolls:
            roll.policy = [False] * len(roll.ids)

        def forbidden(*args, **kwargs):
            """Detect numerical work for a batch with no model actions."""
            pytest.fail("empty action batch executed forward")

        monkeypatch.setattr(policy, "_logprobs", forbidden)
        measured = policy.accumulate(model, None, rolls, batch, beta=0.0, micro_batch=1)
        assert measured.tokens == 0 and measured.loss == measured.kl == 0
        assert all(value.grad is None for value in model.parameters())


@pytest.mark.parametrize("name", ["legacy", "standard", "looped", "custom"])
def test_ragged_round_decoder_matches_individual_full_prefixes_and_drops_cache(name):
    """Legacy cached decoding and declared complete-prefix fallback agree with each row's independent logical positions."""
    with policy_model(name) as model, torch.no_grad():
        model.eval()
        rows = [[1, 2, 3], [4]]
        ids, positions, live = padded(rows, "cpu")
        rng = torch.get_rng_state().clone()
        with PrefixDecoder(model, ids, positions, live, window=8) as decoder:
            for step in range(3):
                for index, row in enumerate(rows):
                    expected = logits(model, torch.tensor([row]))[:, -1]
                    torch.testing.assert_close(decoder.values[index:index + 1, -1], expected, rtol=2e-5, atol=2e-6)
                if step < 2:
                    drawn = torch.tensor([5 + step, 7 + step])
                    decoder.advance(drawn)
                    for row, token in zip(rows, drawn.tolist(), strict=True):
                        row.append(token)
            assert torch.equal(torch.get_rng_state(), rng)
            if name != "legacy":
                assert decoder.cache is None
        assert decoder.cache is None and decoder.closed
        with pytest.raises(ValueError, match="closed"):
            decoder.advance(torch.tensor([1, 1]))


@pytest.mark.parametrize("name", ["standard", "looped", "custom"])
def test_actual_declared_rollout_and_policy_update_use_the_same_evaluation_model(name):
    """Actual grammar/sampling and a finite trusted gym seam produce aligned targets and a real policy gradient without dropout drift."""
    from llm.corpus.baseline import byte_alphabet
    from llm.config import RlConfig
    from llm.gym.client import Episode
    from llm.train import rollout
    from tests.test_rl_pipeline import FakeGym

    tokenizer = byte_alphabet()
    torch.manual_seed(67)
    with policy_model(name, vocab=320, context=128) as model:
        model.train()
        reference = copy.deepcopy(model).eval().requires_grad_(False)
        modes = tuple(module.training for module in model.modules())
        rng = torch.get_rng_state().clone()
        episodes = [Episode(str(uuid4()), index, "bash", "fixture", 1, "one", "two") for index in range(2)]
        cfg = RlConfig(model_name="fixture", base_model="fixture", task_types=["fixture"], max_tokens=8)
        rolls = rollout.run(model, tokenizer, FakeGym(), episodes, cfg, window=128, max_rounds=1,
            generator=torch.Generator().manual_seed(73))
        assert torch.equal(torch.get_rng_state(), rng)
        assert tuple(module.training for module in model.modules()) == modes
        assert all(not roll.live and len(roll.ids) == len(roll.policy) and not roll.policy[0] for roll in rolls)
        batch = policy.score(rolls)
        update = policy.accumulate(model, reference, rolls, batch, beta=0.02, micro_batch=2)
        assert update.tokens > 0 and any(value.grad is not None and value.grad.any() for value in model.parameters())
        assert torch.equal(torch.get_rng_state(), rng)
        assert tuple(module.training for module in model.modules()) == modes
