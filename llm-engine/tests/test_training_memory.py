"""Memory changes must preserve updates, including when telemetry runs out of VRAM."""

import copy

import numpy as np
import pytest
import torch

from llm.config import TrainConfig
from llm.model.transformer import Transformer
from llm.train import diagnostics, loop
from llm.train.plan import DataPlan


def _train(model, callback=None):
    corpus = np.arange(257, dtype=np.int64) % model.vocab
    mask = (corpus % 3 != 0).astype(np.float32)
    cfg = TrainConfig(
        model_name="memory-check",
        dimensions=16, heads=2, layers=2, ff_multiplier=2, ctx_window=16,
        batch_size=2, epochs=1, compile_model=False, checkpoint_target=0,
    )
    return loop.run(model, corpus, DataPlan.whole_corpus(len(corpus)), cfg,
                    "cpu", on_progress=callback, mask=mask)


def _model():
    torch.manual_seed(7)
    return Transformer(32, 16, 2, 2, 2, 16)


def test_forward_never_keeps_the_previous_updates_gradients():
    model = _model()
    seen = []

    def before_forward(module, _args):
        seen.append(True)
        assert all(p.grad is None for p in module.parameters())

    handle = model.register_forward_pre_hook(before_forward)
    try:
        result = _train(model, lambda frame: None)
    finally:
        handle.remove()
    assert result["steps"] > 1
    assert len(seen) > result["steps"]  # Includes diagnostic forwards.


def test_cpu_snapshots_do_not_alias_weights_and_measure_the_update():
    model = _model()
    collector = diagnostics.Collector(model, np.arange(32))
    grad = collector.before_step()
    saved = [one.clone() for one in collector.snapshot]
    with torch.no_grad():
        for block in model.blocks:
            for p in block.parameters():
                p.add_(0.25)
    for before, snapshot in zip(saved, collector.snapshot):
        assert snapshot.device.type == "cpu"
        torch.testing.assert_close(before, snapshot)
    measured = collector.after_step(grad)
    for ratio, before, block in zip(measured.update_ratio_layers, saved, model.blocks):
        after = torch.cat([p.detach().reshape(-1) for p in block.parameters()])
        assert ratio == pytest.approx(float((after - before).norm() / after.norm()))
    assert collector.snapshot is None


@pytest.mark.parametrize("phase", ["observe", "before", "after"])
def test_diagnostic_oom_preserves_training_updates(monkeypatch, capsys, phase):
    expected = _model()
    actual = copy.deepcopy(expected)
    reference = _train(expected)
    calls = []

    def oom(*args, **kwargs):
        calls.append(True)
        raise torch.cuda.OutOfMemoryError("injected diagnostic allocation failure")

    if phase == "observe":
        monkeypatch.setattr(diagnostics.Collector, "_reduce_attention", oom)
    elif phase == "before":
        monkeypatch.setattr(diagnostics, "snapshot_weights", oom)
    else:
        monkeypatch.setattr(diagnostics, "_embedding_metrics", oom)
    frames = []
    result = _train(actual, frames.append)
    assert result["steps"] == reference["steps"]
    assert len(calls) == 1  # Stop paying for samples that cannot fit.
    assert all(frame.diag is None for frame in frames)
    assert "diagnostics disabled after CUDA OOM" in capsys.readouterr().out
    for wanted, got in zip(expected.parameters(), actual.parameters()):
        torch.testing.assert_close(wanted, got, rtol=0, atol=0)
    for block in actual.blocks:
        assert block.last_input is None
        assert block.attn.last_attention is None
        assert not block.record_activations
        assert not block.attn.record_attention
        assert not block.attn._forward_hooks


def test_diagnostics_do_not_swallow_programming_errors(monkeypatch):
    def broken(*args):
        raise ValueError("bad measurement")

    monkeypatch.setattr(diagnostics, "snapshot_weights", broken)
    with pytest.raises(ValueError, match="bad measurement"):
        _train(_model(), lambda frame: None)
