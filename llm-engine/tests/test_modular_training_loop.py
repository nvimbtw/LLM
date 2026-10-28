"""Actual shared-loop recipe updates and exact optimizer/RNG/cursor recovery, separate from the supervised API gate."""

import copy
from dataclasses import replace
import threading

import numpy as np
import pytest
import torch

from llm.architecture.capture import CausalCapture, capture_complete
from llm.architecture.interfaces import LanguageModelFacts
from llm.config import TrainConfig
from llm.model import checkpoint
from llm.model.checkpoint import TrainState
from llm.train import loop
from llm.train.model_memory import plan_model
from llm.train.plan import DataPlan
from llm.train import plan as plan_mod
from llm.train.model_policy import optimizer_groups
from llm.train.progress import Checkpoint
from tests.test_architecture_builtins import build_template, legacy

VALUES = {"dimensions":8,"heads":2,"ff_multiplier":2,"context_window":8,
          "rope_base":10000.0,"learned_temperature":True}


class DropoutModel(torch.nn.Module):
    """A minimal stochastic causal LM with explicit facts and no Transformer-specific structure."""

    def __init__(self):
        """Own vocabulary lookup, stochastic training behavior and a position-local vocabulary projection."""
        super().__init__()
        self.tokens = torch.nn.Embedding(16,7)
        self.noise = torch.nn.Dropout(0.25)
        self.output = torch.nn.Linear(7,16)

    @property
    def facts(self) -> LanguageModelFacts:
        """Declare real vocabulary and trained context without exposing guessed layer/head fields."""
        return LanguageModelFacts(16,8)

    def forward(self,tokens,*,positions=None,key_mask=None):
        """Produce independent per-position logits; eval is deterministic and training consumes the owned torch RNG."""
        return self.output(self.noise(self.tokens(tokens)))


def config(**changes) -> TrainConfig:
    """Keep deliberate mismatched legacy shape defaults so recipe context authority is observable."""
    return TrainConfig(**{"model_name":"modular-loop","ctx_window":64,"batch_size":2,"epochs":1,
        "lr":1e-3,"warmup_steps":0,"compile_model":False,"precision":"fp32","checkpoint_target":0,**changes})


def captured_plan(model):
    """Build an actual exact-shape CPU export over this model's Parameter identities for memory admission."""
    tokens = torch.arange(16).reshape(2,8)%16
    positions = torch.arange(8).expand(2,-1)
    captured = capture_complete(CausalCapture(model),(tokens,positions,torch.ones_like(tokens,dtype=torch.bool)))
    return plan_model(model,captured.exported,allocator_bytes=2 << 30)


def run(model, *, cfg=None, resume=None, callback=None, writer=None, cancel=None):
    """Train real bounded corpus windows with a real captured plan; no process containment is simulated here."""
    corpus = np.arange(129,dtype=np.int64)%16
    return loop.run(model,corpus,DataPlan.whole_corpus(len(corpus)),cfg or config(),"cpu",
        on_progress=callback,on_checkpoint=writer,cancel=cancel,resume=resume,model_plan=captured_plan(model))


@pytest.mark.parametrize("name",["standard","looped","nested"])
def test_shared_loop_trains_real_recipes_using_facts_and_captured_memory_without_legacy_formulas(name,monkeypatch):
    """Configured recipes train, keep shared aliases and never use layer/head formulas or mutate caller context defaults."""
    model = build_template(name,VALUES,vocab=16)
    before = {name:value.detach().clone() for name,value in model.named_parameters()}
    def forbidden(*_args,**_kwargs):
        """Ensure recipe admission cannot consult a Transformer width/layer planner."""
        pytest.fail("recipe memory must use actual capture")
    monkeypatch.setattr(loop.memory,"plan",forbidden)
    cfg = config()
    result = run(model,cfg=cfg)
    assert result["steps"] == 8 and result["total_steps"] == 8
    assert result["memory"]["kind"] == "captured_model" and result["final_loss"] > 0
    assert cfg.ctx_window == 64 and model.facts.context_window == 8
    assert any(not torch.equal(before[name],value) for name,value in model.named_parameters())
    assert not hasattr(model,"recompute_depth")


def test_standard_recipe_shared_loop_preserves_legacy_parameter_updates_and_temperature_policy():
    """Identical initial weights/data/schedule produce identical eager CPU legacy and standard-recipe updates."""
    torch.manual_seed(11)
    expected = legacy(VALUES,vocab=16)
    actual = build_template("standard",VALUES,vocab=16)
    actual.instances["model"].model.load_state_dict(expected.state_dict())
    corpus = np.arange(129,dtype=np.int64)%16
    cfg = config(ctx_window=8)
    reference = loop.run(expected,corpus,DataPlan.whole_corpus(len(corpus)),cfg,"cpu",on_progress=None)
    result = run(actual,cfg=cfg)
    assert result["steps"] == reference["steps"]
    assert result["final_loss"] == reference["final_loss"]
    for name,value in expected.state_dict().items():
        torch.testing.assert_close(actual.instances["model"].model.state_dict()[name],value,rtol=0,atol=0)


def test_shared_loop_retains_explicit_occurrence_recomputation_and_full_training_updates():
    """The loop preserves a caller's tested replay policy and fixed sharing/counts without inventing a layer-depth setting."""
    expected = build_template("looped",VALUES,vocab=16)
    actual = copy.deepcopy(expected)
    admission = captured_plan(actual)
    targets = actual.recomputation_targets()
    actual.configure_recomputation(targets)
    reference = run(expected)
    corpus = np.arange(129,dtype=np.int64)%16
    result = loop.run(actual,corpus,DataPlan.whole_corpus(len(corpus)),config(),"cpu",on_progress=None,model_plan=admission)
    assert result["steps"] == reference["steps"] and actual.recomputation_targets() == targets
    for left,right in zip(expected.parameters(),actual.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)


def test_recipe_missing_or_wrong_model_admission_refuses_before_optimizer_creation(monkeypatch):
    """No model update or optimizer allocation occurs without its captured final-profile admission."""
    model = build_template("looped",VALUES,vocab=16)
    before = copy.deepcopy(model.state_dict())
    def forbidden(*_args,**_kwargs):
        """Detect optimizer creation before final admission has passed."""
        pytest.fail("optimizer must not be constructed")
    corpus = np.arange(129,dtype=np.int64)%16
    plan = captured_plan(model)
    monkeypatch.setattr(torch.optim,"AdamW",forbidden)
    for memory_plan in (None,replace(plan,parameter_bytes=plan.parameter_bytes+4),
                        replace(plan,input_shapes=((1,8),)*3)):
        with pytest.raises(ValueError,match="captured"):
            loop.run(model,corpus,DataPlan.whole_corpus(len(corpus)),config(),"cpu",on_progress=None,model_plan=memory_plan)
    for name,value in model.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)


def test_recipe_diagnostics_measure_semantic_owners_without_changing_updates_or_rng():
    """Optional occurrence/owner samples preserve the exact CPU training trajectory and omit unavailable legacy metrics."""
    expected = DropoutModel()
    actual = copy.deepcopy(expected)
    frames = []
    torch.manual_seed(31)
    reference = run(expected)
    wanted_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(31)
    result = run(actual,callback=frames.append)
    assert result["steps"] == reference["steps"]
    for left,right in zip(expected.parameters(),actual.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert torch.equal(wanted_rng,torch.random.get_rng_state())
    measured = [frame.diag for frame in frames if frame.diag is not None]
    assert measured and all("parameter_grad_norm" in value and "occurrences" in value for value in measured)
    assert all("attention_entropy_layers" not in value and "embedding_rms" not in value for value in measured)
    assert not any(module._forward_hooks for module in actual.modules())


@pytest.mark.parametrize("name",["standard","looped","dropout"])
def test_recipe_checkpoint_cursor_optimizer_and_rng_resume_exact_training_trajectory(name):
    """A real interrupted loop resumes from the next saved batch with one optimizer owner per shared Parameter."""
    expected = DropoutModel() if name == "dropout" else build_template(name,VALUES,vocab=16)
    partial = copy.deepcopy(expected)
    restored = copy.deepcopy(expected)
    cancel = threading.Event()
    states = []
    def save(model,optimizer,*,step,total_steps,epoch,stage,stage_name,loss,cursor):
        """Capture the real state at an optimizer boundary, then cancel after the third complete update."""
        if step == 3:
            states.append(TrainState(arch={},state_dict=copy.deepcopy(model.state_dict()),step=step,epoch=epoch,
                stage=stage,total_steps=total_steps,optimizer=copy.deepcopy(optimizer.state_dict()),
                rng=copy.deepcopy(checkpoint.rng_state()),extra={"cursor":cursor}))
            cancel.set()
        return Checkpoint(model_id="fixture",index=step,step=step,total_steps=total_steps,epoch=epoch,
            stage=stage,stage_name=stage_name,path=f"fixture/{step}",loss=loss)
    cfg = config(checkpoint_target=8)
    torch.manual_seed(19)
    reference = run(expected,cfg=cfg)
    reference_rng = torch.random.get_rng_state().clone()
    torch.manual_seed(19)
    interrupted = run(partial,cfg=cfg,writer=save,cancel=cancel)
    assert interrupted["cancelled"] and interrupted["steps"] == 3 and len(states) == 1
    restored.load_state_dict(states[0].state_dict)
    result = run(restored,cfg=cfg,resume=states[0])
    assert result["steps"] == reference["steps"] and not result["cancelled"]
    for left,right in zip(expected.parameters(),restored.parameters()):
        torch.testing.assert_close(left,right,rtol=0,atol=0)
    assert torch.equal(reference_rng,torch.random.get_rng_state())


@pytest.mark.parametrize("name", ["standard", "looped"])
def test_continuation_preserves_moments_but_applies_requested_decay_and_device_options(name):
    """A real resumed update matches independent AdamW moments with the new request's decay, LR and CPU policy."""
    actual = build_template(name, VALUES, vocab=16)
    corpus = np.arange(129, dtype=np.int64) % 16
    plan = DataPlan.whole_corpus(len(corpus))
    inputs, targets, _, _ = next(plan_mod.iter_batches(plan_mod.EpochStream(corpus, plan, 0), 8, 2))
    x, y = torch.from_numpy(inputs.copy()), torch.from_numpy(targets.copy())
    previous = torch.optim.AdamW(optimizer_groups(actual, weight_decay=0.1), lr=0.1, betas=(0.9, 0.95), fused=False)
    torch.nn.functional.cross_entropy(actual(x).reshape(-1, 16), y.reshape(-1)).backward()
    previous.step()
    previous.zero_grad(set_to_none=True)
    saved = TrainState(arch={}, state_dict={}, optimizer=copy.deepcopy(previous.state_dict()), step=0,
        epoch=0, stage=0, total_steps=8, rng={}, extra={"cursor": None})
    expected = copy.deepcopy(actual)
    cfg = config(weight_decay=0.9, lr=0.002, checkpoint_target=8)
    reference = torch.optim.AdamW(optimizer_groups(expected, weight_decay=cfg.weight_decay),
                                lr=loop.lr_at(0, 8, cfg), betas=(0.9, 0.95), fused=False)
    for target, source in zip(expected.parameters(), actual.parameters(), strict=True):
        reference.state[target] = copy.deepcopy(previous.state[source])
    torch.nn.functional.cross_entropy(expected(x).reshape(-1, 16), y.reshape(-1)).backward()
    torch.nn.utils.clip_grad_norm_(expected.parameters(), cfg.grad_clip)
    reference.step()
    stop = threading.Event()
    updates = []

    def saved_update(model, optimizer, *, step, total_steps, epoch, stage, stage_name, loss, cursor):
        """Observe one real resumed update and stop before another batch can change it."""
        updates.append(copy.deepcopy(optimizer.state_dict()))
        stop.set()
        return Checkpoint(model_id="fixture", index=1, step=step, total_steps=total_steps,
            epoch=epoch, stage=stage, stage_name=stage_name, path="fixture/continuation", loss=loss)

    result = run(actual, cfg=cfg, resume=saved, writer=saved_update, cancel=stop)
    assert result["cancelled"] and result["steps"] == 1 and len(updates) == 1
    for target, source in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(target, source, rtol=0, atol=0)
    assert all(group["weight_decay"] in (0, 0.9) and group["fused"] is False for group in updates[0]["param_groups"])
    assert any(group["weight_decay"] == 0.9 for group in updates[0]["param_groups"])
    assert all(value["step"].item() == 2 for value in updates[0]["state"].values())
