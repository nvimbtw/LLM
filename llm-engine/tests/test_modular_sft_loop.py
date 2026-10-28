"""Real masked recipe updates and exact next-batch recovery before the SFT worker adapter is connected."""

import copy
from threading import Event

import numpy as np
import pytest
import torch

from llm.model import checkpoint
from llm.model.checkpoint import TrainState
from llm.train import loop, plan as batches
from llm.train.model_policy import optimizer_groups
from llm.train.plan import DataPlan
from llm.train.progress import Checkpoint
from llm.train.warm_start import initialize_token_rows
from tests.test_architecture_builtins import build_template
from tests.test_modular_training_loop import DropoutModel, VALUES, captured_plan, config


def model_for_sft(name):
    """Construct an actual inherited recipe or a valid stochastic LM without a token-row capability."""
    return DropoutModel() if name == "no_token_rows" else build_template(name, VALUES, vocab=16)


def masked_inputs():
    """Keep two initial batches without assistant targets so cursor position differs from optimizer step."""
    corpus = np.arange(129, dtype=np.int64) % 16
    mask = np.ones(len(corpus), dtype=np.uint8)
    mask[:32] = 0
    mask[40::3] = 0
    return corpus, mask


@pytest.mark.parametrize("name", ["standard", "looped", "no_token_rows"])
def test_real_masked_update_matches_independent_loss_and_leaves_inputs_unchanged(name):
    """Only the mask-weighted targets contribute to the first actual AdamW update after empty target batches."""
    actual = model_for_sft(name)
    status = initialize_token_rows(actual, [1, 2])
    assert status.status == ("initialized" if name == "standard" else "skipped_unsupported")
    expected = copy.deepcopy(actual)
    corpus, mask = masked_inputs()
    original_corpus, original_mask = corpus.copy(), mask.copy()
    cfg = config(ctx_window=8, checkpoint_target=16)
    data = DataPlan.whole_corpus(len(corpus))
    _, total = loop.step_plan(corpus, data, cfg, mask)
    stream = batches.EpochStream(corpus, data, 0, mask)
    inputs, targets, weights, _ = next(batch for batch in batches.iter_batches(stream, 8, 2, True) if batch[2].any())
    optimizer = torch.optim.AdamW(optimizer_groups(expected, weight_decay=cfg.weight_decay),
        lr=loop.lr_at(0, total, cfg), betas=(0.9, 0.95), fused=False)
    torch.manual_seed(51)
    values = expected(torch.from_numpy(inputs.copy())).reshape(-1, 16)
    losses = torch.nn.functional.cross_entropy(values, torch.from_numpy(targets.copy()).reshape(-1), reduction="none")
    selected = torch.from_numpy(weights.copy()).float().reshape(-1)
    ((losses * selected).sum() / selected.sum()).backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), cfg.grad_clip)
    optimizer.step()
    wanted_rng = torch.get_rng_state().clone()
    stop = Event()
    consumed = []

    def save(model, optimizer, *, step, total_steps, epoch, stage, stage_name, loss, cursor):
        """Stop after the first real update and retain the exact next-batch position for this assertion."""
        assert step == 1 and cursor == {"epoch": 0, "batch": 3}
        stop.set()
        return Checkpoint(model_id="fixture", index=1, step=step, total_steps=total_steps,
            epoch=epoch, stage=stage, stage_name=stage_name, path="fixture/masked")

    admission = captured_plan(actual)
    torch.manual_seed(51)
    result = loop.run(actual, corpus, data, cfg, "cpu", on_progress=None, mask=mask,
        on_checkpoint=save, cancel=stop, model_plan=admission, on_batch=consumed.append)
    assert result["steps"] == 1 and result["cancelled"] and consumed == [1, 2, 3]
    for left, right in zip(expected.parameters(), actual.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(wanted_rng, torch.get_rng_state())
    np.testing.assert_array_equal(corpus, original_corpus)
    np.testing.assert_array_equal(mask, original_mask)


@pytest.mark.parametrize("name", ["standard", "looped", "no_token_rows"])
def test_masked_exact_cursor_recovery_preserves_weights_optimizer_and_rng(name):
    """Interrupted SFT resumes the next batch rather than using the smaller optimizer-step count as its cursor."""
    expected = model_for_sft(name)
    initialize_token_rows(expected, [1, 2])
    partial, restored = copy.deepcopy(expected), copy.deepcopy(expected)
    corpus, mask = masked_inputs()
    data = DataPlan.whole_corpus(len(corpus))
    cfg = config(ctx_window=8, checkpoint_target=16)
    stop, states = Event(), []

    def save(model, optimizer, *, step, total_steps, epoch, stage, stage_name, loss, cursor):
        """Retain actual state after two updates and four consumed batches, then request cooperative cancellation."""
        if step == 2:
            assert cursor == {"epoch": 0, "batch": 4}
            states.append(TrainState(arch={}, state_dict=copy.deepcopy(model.state_dict()), step=step,
                epoch=epoch, stage=stage, total_steps=total_steps, optimizer=copy.deepcopy(optimizer.state_dict()),
                rng=copy.deepcopy(checkpoint.rng_state()), extra={"cursor": cursor}))
            stop.set()
        return Checkpoint(model_id="fixture", index=step, step=step, total_steps=total_steps,
            epoch=epoch, stage=stage, stage_name=stage_name, path="fixture/masked")

    admission = captured_plan(expected)
    torch.manual_seed(63)
    reference = loop.run(expected, corpus, data, cfg, "cpu", on_progress=None, mask=mask, model_plan=admission)
    wanted_rng = torch.get_rng_state().clone()
    admission = captured_plan(partial)
    torch.manual_seed(63)
    interrupted = loop.run(partial, corpus, data, cfg, "cpu", on_progress=None, mask=mask,
        model_plan=admission, on_checkpoint=save, cancel=stop)
    assert interrupted["cancelled"] and interrupted["steps"] == 2 and len(states) == 1
    restored.load_state_dict(states[0].state_dict)
    result = loop.run(restored, corpus, data, cfg, "cpu", on_progress=None, mask=mask,
        model_plan=captured_plan(restored), resume=states[0])
    assert result["steps"] == reference["steps"] and not result["cancelled"]
    for left, right in zip(expected.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert torch.equal(wanted_rng, torch.get_rng_state())
