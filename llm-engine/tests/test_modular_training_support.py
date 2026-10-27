"""Model-independent optimizer/memory/diagnostic contracts and occurrence RNG replay."""

import copy

import pytest
import torch
from torch import nn

from llm.architecture.build import build_model
from llm.architecture.capture import CausalCapture, capture_complete
from llm.architecture.numerical import equivalence, preserved_model
from llm.architecture.interfaces import ExecutionOccurrence
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader
from llm.model.transformer import Attention
from llm.train import memory
from llm.train.model_diagnostics import DiagnosticBudget, ModelDiagnostics
from llm.train.diagnostics_model import ModelCollector
from llm.train.model_memory import plan_model
from llm.train.model_policy import OptimizerPolicy, ParameterPolicy, optimizer_groups
from llm.train.recompute_plan import RecomputeEstimate, conservative_recomputation, plan_recomputation
from tests.test_architecture_builtins import build_template
from tests.test_architecture_core import SOURCE, tensor_recipe, revision


def test_model_diagnostic_report_retains_the_actual_optimizer_step():
    """Semantic diagnostics carry the supplied update step and omit unavailable Transformer measurements."""
    model = nn.Linear(3, 5)
    collector = ModelCollector(model)
    inputs = torch.ones(2, 3)
    collector.observe(inputs, None, None, None)
    model(inputs).square().mean().backward()
    gradients = collector.before_step()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    optimizer.step()
    report = collector.after_step(gradients, step=7).as_dict()
    assert report["step"] == 7
    assert report["parameter_grad_norm"] == gradients
    assert report["parameter_update_ratio"]
    assert report["occurrences"]
    assert "grad_layers" not in report and "attn_entropy" not in report


def test_optimizer_owns_shared_parameters_once_and_rejects_conflicting_policies():
    """Alias names cannot duplicate optimizer state or silently change another alias's decay policy."""
    model = nn.Module()
    model.first = nn.Linear(3, 5)
    model.second = nn.Linear(3, 5)
    model.second.weight = model.first.weight
    groups = optimizer_groups(model, weight_decay=0.1,
        overrides={"first.weight":OptimizerPolicy(0.0),"second.weight":OptimizerPolicy(0.0)})
    parameters = [value for group in groups for value in group["params"]]
    assert len(parameters) == len({id(value) for value in model.parameters()})
    with pytest.raises(ValueError,match="conflicting"):
        optimizer_groups(model,weight_decay=0.1,
            overrides={"first.weight":OptimizerPolicy(0.0),"second.weight":OptimizerPolicy(0.2)})
    with pytest.raises(ValueError,match="foreign"):
        optimizer_groups(model,weight_decay=0.1,
            capabilities=(ParameterPolicy(nn.Parameter(torch.ones(1)),OptimizerPolicy(0.0)),))
    with pytest.raises(ValueError,match="unknown"):
        optimizer_groups(model,weight_decay=0.1,overrides={"missing":OptimizerPolicy(0.0)})


def test_temperature_exclusions_follow_component_capability_and_frozen_parameters_are_absent():
    """A coincidentally named custom Parameter receives ordinary decay; actual attention temperatures do not."""
    model = nn.Module()
    model.attention = Attention(8,2,True)
    model.log_temperature = nn.Parameter(torch.zeros(2))
    model.frozen = nn.Parameter(torch.ones(1),requires_grad=False)
    groups = optimizer_groups(model,weight_decay=0.2)
    decay = {id(parameter):group["weight_decay"] for group in groups for parameter in group["params"]}
    assert decay[id(model.attention.log_temperature)] == 0
    assert decay[id(model.log_temperature)] == 0.2
    assert id(model.frozen) not in decay


def test_custom_memory_plan_uses_tensor_ownership_reference_and_admission_ceiling():
    """Prime-width non-Transformer components use real tensors/operations and never layer/head formulas."""
    model = nn.Sequential(nn.Linear(3,5),nn.GELU(),nn.Linear(5,7))
    captured = capture_complete(model,(torch.ones(2,3),))
    reference = copy.deepcopy(model)
    reference_capture = capture_complete(reference,(torch.ones(2,3),))
    plain = plan_model(model,captured.exported,allocator_bytes=2 << 30)
    anchored = plan_model(model,captured.exported,allocator_bytes=2 << 30,reference_model=reference,
                          reference_export=reference_capture.exported,resident_bytes=100)
    expected = sum(parameter.numel()*parameter.element_size() for parameter in model.parameters())
    assert plain.parameter_bytes == expected
    assert anchored.estimated_bytes == plain.estimated_bytes + expected + plain.activation_bytes + 100
    shared = plan_model(model,captured.exported,allocator_bytes=2 << 30,reference_model=model,reference_export=captured.exported)
    assert shared.reference_bytes == plain.activation_bytes
    assert plain.estimated_bytes < plain.target_bytes < plain.allocator_bytes
    with pytest.raises(ValueError,match="target"):
        plan_model(model,captured.exported,allocator_bytes=1)
    with pytest.raises(ValueError,match="snapshot"):
        plan_model(reference,captured.exported,allocator_bytes=2 << 30)
    with pytest.raises(ValueError,match="reference"):
        plan_model(model,captured.exported,allocator_bytes=2 << 30,reference_model=reference)


def test_unknown_operation_memory_estimate_fails_without_real_allocation_probe():
    """Capturable custom operators need an explicit workspace policy before training admission."""
    model = nn.Softplus()
    captured = capture_complete(model,(torch.ones(2,3),))
    with pytest.raises(ValueError,match="unavailable"):
        plan_model(model,captured.exported,allocator_bytes=2 << 30)


def test_meta_planning_and_tied_parameter_storage_require_no_real_weight_allocation():
    """Meta modules admit known shapes without materializing huge weights, while shared storage counts once."""
    model = nn.Sequential(nn.Linear(3,3,bias=False,device="meta"),nn.Linear(3,3,bias=False,device="meta"))
    model[1].weight = model[0].weight
    captured = capture_complete(model,(torch.ones(2,3,device="meta"),))
    plan = plan_model(model,captured.exported,allocator_bytes=2 << 30)
    assert plan.parameter_bytes == 9 * 4
    enormous = nn.Linear(100000,100000,device="meta")
    with pytest.raises(ValueError,match="allocation"):
        capture_complete(enormous,(torch.ones(1,100000,device="meta"),))


@pytest.mark.parametrize("name",["standard","looped"])
def test_recipe_memory_and_occurrence_diagnostics_use_actual_bounded_model_calls(name):
    """The standard/shared recipe planners count real tensors and actual attention workspace once per computation."""
    model = build_template(name)
    tokens = torch.tensor([[1,2,3]])
    captured = capture_complete(CausalCapture(model),(tokens,torch.arange(3).reshape(1,3),torch.ones_like(tokens,dtype=torch.bool)))
    plan = plan_model(model,captured.exported,allocator_bytes=2 << 30)
    assert plan.activation_bytes > 0
    assert plan.parameter_bytes == sum(value.numel()*value.element_size() for value in (*model.parameters(),*model.buffers()))
    rows = ModelDiagnostics(model).sample(tokens)
    assert len(rows) == len(model.recomputation_targets())


def test_recomputation_planner_selects_only_supported_exact_profile_occurrence_estimates():
    """A planner selects minimal replay work and refuses stale/unsupported budgets rather than guessing discounts."""
    targets = (ExecutionOccurrence("core",(1,0,0)),ExecutionOccurrence("core",(1,1,0)))
    plain = RecomputeEstimate("a"*64,(),300)
    partial = RecomputeEstimate("a"*64,targets[:1],250)
    full = RecomputeEstimate("a"*64,targets,180)
    assert plan_recomputation(targets,(plain,partial,full),profile_digest="a"*64,target_bytes=400) == plain
    assert plan_recomputation(targets,(plain,partial,full),profile_digest="a"*64,target_bytes=260) == partial
    assert plan_recomputation(targets,(plain,partial,full),profile_digest="a"*64,target_bytes=200) == full
    with pytest.raises(ValueError,match="target"):
        plan_recomputation(targets,(plain,partial,full),profile_digest="a"*64,target_bytes=100)
    with pytest.raises(ValueError,match="stale"):
        plan_recomputation(targets,(plain,),profile_digest="b"*64,target_bytes=400)
    with pytest.raises(ValueError,match="duplicate"):
        plan_recomputation(targets,(plain,plain),profile_digest="a"*64,target_bytes=400)


def test_explicit_replay_reserves_captured_work_and_never_invents_automatic_savings():
    """Replay needs additional admitted workspace; unsupported counts and tighter ceilings fail closed."""
    from dataclasses import replace
    model = build_template("looped")
    tokens = torch.tensor([[1, 2, 3]])
    captured = capture_complete(CausalCapture(model), (tokens, torch.arange(3).reshape(1, 3),
                                                     torch.ones_like(tokens, dtype=torch.bool)))
    memory_plan = plan_model(model, captured.exported, allocator_bytes=2 << 30)
    eligible = model.recomputation_targets()
    ordinary, automatic = conservative_recomputation(eligible, memory_plan, profile_digest="a" * 64, requested=None)
    replay, explicit = conservative_recomputation(eligible, memory_plan, profile_digest="a" * 64, requested=2)
    assert automatic.selected.occurrences == () and ordinary == memory_plan
    assert explicit.selected.occurrences == eligible[:2]
    assert replay.activation_bytes == 2 * memory_plan.activation_bytes
    assert explicit.selected.estimated_bytes == replay.estimated_bytes
    with pytest.raises(ValueError, match="count"):
        conservative_recomputation(eligible, memory_plan, profile_digest="a" * 64, requested=len(eligible) + 1)
    restricted = replace(memory_plan, target_bytes=memory_plan.estimated_bytes)
    with pytest.raises(ValueError, match="target"):
        conservative_recomputation(eligible, restricted, profile_digest="a" * 64, requested=True)


def test_worker_allocator_cap_can_only_be_tightened(monkeypatch):
    """A training planner cannot widen the allocator ceiling already imposed by admission."""
    class Properties:
        """Expose fixed device capacity for a pure cap calculation test."""
        total_memory = 1000
    applied = []
    monkeypatch.setattr(torch.cuda,"is_available",lambda:True)
    monkeypatch.setattr(torch.cuda,"get_device_properties",lambda _:Properties())
    monkeypatch.setattr(torch.cuda,"get_per_process_memory_fraction",lambda:0.3)
    monkeypatch.setattr(torch.cuda,"set_per_process_memory_fraction",applied.append)
    memory.guard("cuda",900)
    memory.guard("cuda",200)
    assert applied == [0.3,0.2]


@pytest.mark.parametrize("name",["standard","looped","nested","expanded"])
@pytest.mark.parametrize("selection",["all","one"])
def test_recomputation_preserves_logits_gradients_and_unique_ownership(name,selection):
    """Actual fixed/shared/nested execution matches its eager trajectory without multiplying parameter owners."""
    eager = build_template(name)
    replay = copy.deepcopy(eager)
    targets = replay.recomputation_targets()
    replay.configure_recomputation(targets if selection == "all" else targets[1:2] or targets[:1])
    eager.train()
    replay.train()
    tokens = torch.tensor([[1,2,3],[3,2,1]])
    original = eager(tokens)
    recomputed = replay(tokens)
    torch.testing.assert_close(original,recomputed,rtol=0,atol=0)
    original.square().mean().backward()
    recomputed.square().mean().backward()
    for expected,actual in zip(eager.parameters(),replay.parameters()):
        torch.testing.assert_close(expected.grad,actual.grad,rtol=0,atol=0)
    assert len(list(eager.parameters())) == len(list(replay.parameters()))
    with pytest.raises(ValueError):
        replay.configure_recomputation((targets[0],targets[0]))
    with pytest.raises(ValueError):
        replay.configure_recomputation((ExecutionOccurrence("missing",()),))


def test_dropout_recomputation_replays_rng_and_diagnostics_preserve_next_training_step():
    """Controlled PyTorch dropout remains identical across replay and optional diagnostic work."""
    source = SOURCE.replace("hidden * self.gain", "torch.nn.functional.dropout(hidden * self.gain, p=0.3, training=self.training)")
    with SourceLoader() as loader:
        definition = loader.load(revision(source))
        model = build_model(resolve_recipe(tensor_recipe(definition.reference),{}, {definition.reference.revision_id:definition}))
        expected = copy.deepcopy(model)
        model.configure_recomputation(model.recomputation_targets())
        inputs = {"hidden":torch.ones(2,3,8)}
        torch.manual_seed(42)
        output = model.forward_named(inputs)["hidden"]
        output.square().mean().backward()
        state = torch.get_rng_state().clone()
        torch.manual_seed(42)
        baseline = expected.forward_named(inputs)["hidden"]
        baseline.square().mean().backward()
        torch.testing.assert_close(output,baseline,rtol=0,atol=0)
        torch.testing.assert_close(next(model.parameters()).grad,next(expected.parameters()).grad,rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),state,rtol=0,atol=0)
    causal = build_template("looped")
    causal.train()
    causal.instances["embedding"].eval()
    modes = [module.training for module in causal.modules()]
    before = {name:value.clone() for name,value in causal.state_dict().items()}
    rng = torch.get_rng_state().clone()
    diagnostics = ModelDiagnostics(causal)
    rows = diagnostics.sample(torch.tensor([[1,2,3]]))
    assert len(rows) == len(causal.recomputation_targets())
    assert len({tuple(row["call_path"]) for row in rows if row["instance_path"] == "core"}) == 2
    assert all("attention" not in row and "embedding" not in row for row in rows)
    assert [module.training for module in causal.modules()] == modes
    torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
    for name,value in causal.state_dict().items():
        torch.testing.assert_close(value,before[name],rtol=0,atol=0)
    assert all(not module._forward_hooks for module in causal.modules())


def test_dropout_replay_matches_strict_capture_outputs_gradients_and_rng():
    """Final-profile replay equivalence checks actual stochastic arithmetic against ordinary strict export."""
    class TensorCapture(nn.Module):
        """Expose the root tensor composition's public named signature for strict export."""

        def __init__(self, model):
            """Retain the same registered Parameter identities in both numerical paths."""
            super().__init__()
            self.model = model

        def forward(self, inputs):
            """Return the root recipe's declared outputs through its actual named executor."""
            return self.model.forward_named(inputs)

    source = SOURCE.replace("hidden * self.gain", "torch.nn.functional.dropout(hidden * self.gain, p=0.3, training=self.training)")
    with SourceLoader() as loader:
        definition = loader.load(revision(source))
        model = build_model(resolve_recipe(tensor_recipe(definition.reference), {},
            {definition.reference.revision_id: definition}))
        wrapper = TensorCapture(model)
        args = ({"hidden": torch.ones(2, 3, 8)},)
        captured = capture_complete(wrapper, args)
        model.configure_recomputation(model.recomputation_targets())
        before = torch.get_rng_state().clone()
        with preserved_model(model):
            equivalence(wrapper, captured.exported.module(), args, "fp32", check_rng=True)
        assert torch.equal(torch.get_rng_state(), before)
        assert all(value.grad is None for value in model.parameters())


def test_equivalence_rejects_different_post_backward_rng_without_changing_caller_state():
    """Equal outputs/gradients cannot authorize replay when extra hidden draws change subsequent updates."""
    model = nn.Linear(3, 5)

    def extra_draw(value):
        """Keep the same parameters and outputs while consuming an additional CPU random draw."""
        torch.rand(())
        return model(value)

    before = torch.get_rng_state().clone()
    with pytest.raises(ValueError, match="RNG draws"):
        equivalence(model, extra_draw, (torch.ones(2, 3),), "fp32", check_rng=True)
    assert torch.equal(torch.get_rng_state(), before)


def test_diagnostic_budget_and_oom_disable_optional_sampling_but_programming_errors_propagate(monkeypatch):
    """Failed optional samples release snapshots/hooks and do not hide model/training errors."""
    model = nn.Linear(3,5)
    limited = ModelDiagnostics(model,DiagnosticBudget(bytes=1))
    assert limited.before_step() is None and limited.disabled_reason
    collector = ModelDiagnostics(model)
    def oom(*_args,**_kwargs):
        """Inject an optional diagnostic allocation failure without performing model work."""
        raise torch.cuda.OutOfMemoryError("diagnostic allocation")
    monkeypatch.setattr(collector,"capture_hook",oom)
    assert collector.sample(torch.ones(2,3)) is None
    assert collector.disabled_reason and not model._forward_hooks
    broken = ModelDiagnostics(model)
    def bug(*_args,**_kwargs):
        """Inject an unexpected diagnostic programming error that must remain actionable."""
        raise RuntimeError("programming error")
    monkeypatch.setattr(broken,"capture_hook",bug)
    with pytest.raises(RuntimeError,match="programming"):
        broken.sample(torch.ones(2,3))


def test_diagnostics_leave_dropout_training_trajectory_and_update_snapshots_unchanged():
    """Sampling and CPU update measurements preserve subsequent randomness/weights in a non-Transformer model."""
    model = nn.Sequential(nn.Linear(3,5),nn.Dropout(0.4),nn.Linear(5,2))
    expected = copy.deepcopy(model)
    inputs = torch.ones(2,3)
    diagnostics = ModelDiagnostics(model)
    torch.manual_seed(51)
    before = torch.get_rng_state().clone()
    assert diagnostics.sample(inputs)
    torch.testing.assert_close(torch.get_rng_state(),before,rtol=0,atol=0)
    expected(inputs).square().mean().backward()
    torch.set_rng_state(before)
    model(inputs).square().mean().backward()
    norms = diagnostics.before_step()
    assert norms and all(value >= 0 for value in norms.values())
    first = torch.optim.AdamW(optimizer_groups(model,weight_decay=0.1),lr=0.01)
    second = torch.optim.AdamW(optimizer_groups(expected,weight_decay=0.1),lr=0.01)
    first.step()
    second.step()
    ratios = diagnostics.after_step()
    assert ratios and all(value >= 0 for value in ratios.values())
    assert not diagnostics.snapshot
    for actual,reference in zip(model.parameters(),expected.parameters()):
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
