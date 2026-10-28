"""Actual strict occurrence partitions, backing-storage lifetimes and minimal replay admission."""

from dataclasses import replace

import pytest
import torch

from llm.architecture.build import build_model
from llm.architecture.capture import CausalCapture, capture_complete
from llm.architecture.numerical import equivalence, preserved_model
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader
from llm.architecture.spec import Recipe
from llm.architecture.validation import make_inputs
from llm.train.model_memory import estimate_model
from llm.train.modular_preflight import training_memory
from llm.train.recompute_capture import capture_replay_storage, output_storage
from llm.train.recompute_plan import captured_recomputation
from tests.test_architecture_builtins import build_template
from tests.test_architecture_core import SOURCE, revision, tensor_recipe


@pytest.mark.parametrize("name", ["standard", "looped", "nested", "expanded"])
def test_actual_capture_partitions_preserve_baseline_and_select_minimal_fitting_replay(name):
    """Each real shared/nested call contributes bounded captured work; selection never changes parameter ownership."""
    model = build_template(name, {"dimensions": 8, "heads": 2, "ff_multiplier": 2, "context_window": 8}, vocab=16)
    args = make_inputs(2, 8, 16, "cpu")
    captured = capture_complete(CausalCapture(model), args)
    memory = estimate_model(model, captured.exported, allocator_bytes=2 << 30)
    with preserved_model(model):
        storage = capture_replay_storage(model, captured.exported, args)
    eligible = model.recomputation_targets()
    assert tuple(region.occurrence for region in storage.regions) == eligible
    assert storage.captured_training_bytes == memory.activation_bytes
    assert storage.activation_bytes(()) == memory.activation_bytes + sum(4 * region.output_bytes for region in storage.regions)
    assert all(region.training_bytes > 0 for region in storage.regions)
    _, full = captured_recomputation(eligible, memory, storage, profile_digest="a" * 64, requested=True)
    assert len(full.candidates) == len(eligible) + 1
    assert full.selected.occurrences == eligible
    if name == "standard":
        assert full.selected.estimated_bytes > memory.estimated_bytes
    else:
        target = (memory.estimated_bytes + min(value.estimated_bytes for value in full.candidates)) // 2
        admitted = replace(memory, target_bytes=target)
        selected_memory, policy = captured_recomputation(eligible, admitted, storage,
            profile_digest="a" * 64, requested=None)
        assert 0 < len(policy.selected.occurrences) < len(eligible)
        assert policy.selected.estimated_bytes == selected_memory.estimated_bytes <= target < memory.estimated_bytes
        assert all(value.estimated_bytes > target for value in policy.candidates
                   if len(value.occurrences) < len(policy.selected.occurrences))
        _, recovered = captured_recomputation(eligible, memory, storage, profile_digest="a" * 64,
            requested=None, retained_policy=policy.as_dict()["selected"])
        assert recovered.selected == policy.selected
        with pytest.raises(ValueError, match="saved replay"):
            captured_recomputation(eligible, memory, storage, profile_digest="b" * 64,
                requested=None, retained_policy=policy.as_dict()["selected"])
        model.configure_recomputation(policy.selected.occurrences)
        with preserved_model(model):
            equivalence(CausalCapture(model), captured.exported.module(), args, "fp32", tokens=args[0], check_rng=True)
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("name", ["standard", "looped", "nested", "expanded"])
def test_meta_admission_observes_complete_named_port_storage_without_allocating_weights(name):
    """All builtin layouts admit shape-only capture, including standard explicit-position checks."""
    model = build_model(build_template(name).plan, device="meta")
    args = make_inputs(2, model.facts.context_window, model.facts.vocabulary_size, "meta")
    captured = capture_complete(CausalCapture(model), args)
    memory, replay = training_memory(model, captured.exported, args, allocator_bytes=2 << 30,
                                    profile_digest="a" * 64, recompute=None)
    assert all(parameter.device.type == "meta" for parameter in model.parameters())
    assert replay.selected.occurrences == ()
    assert replay.selected.estimated_bytes == memory.estimated_bytes
    assert len(replay.candidates) == len(model.recomputation_targets()) + 1
    assert all(not module._forward_hooks for module in model.modules())


def test_automatic_compiled_replay_refuses_in_meta_admission():
    """Automatic selection cannot enter unsupported checkpoint compilation when only replay fits."""
    model = build_model(build_template("looped").plan, device="meta")
    args = make_inputs(2, model.facts.context_window, model.facts.vocabulary_size, "meta")
    captured = capture_complete(CausalCapture(model), args)
    memory, replay = training_memory(model, captured.exported, args, allocator_bytes=2 << 30,
                                    profile_digest="a" * 64, recompute=None)
    target = (memory.estimated_bytes + min(value.estimated_bytes for value in replay.candidates)) // 2
    with pytest.raises(ValueError, match="compile_model=false"):
        training_memory(model, captured.exported, args, allocator_bytes=(512 << 20) + target,
                        profile_digest="a" * 64, recompute=None, compiled=True)
    with pytest.raises(ValueError, match="no supported recomputation"):
        training_memory(model, captured.exported, args, allocator_bytes=512 << 20,
                        profile_digest="a" * 64, recompute=None)


def test_named_outputs_charge_unused_ports_and_full_aliased_backing_storage():
    """A small returned view retains its whole larger backing allocation, including an unused named output."""
    port = 'TensorPort(dtype="floating", shape=("batch", "sequence", "width"))'
    source = SOURCE.replace('outputs={"hidden": ' + port + '}',
        'outputs={"hidden": ' + port + ', "unused": ' + port + '}')
    source = source.replace('return {"hidden": hidden * self.gain}',
        'expanded = hidden * self.gain\n        wide = torch.cat((expanded, expanded, expanded, expanded), dim=-1)\n'
        '        return {"hidden": wide[:, :, :hidden.shape[-1]], "unused": expanded}')
    with SourceLoader() as loader:
        definition = loader.load(revision(source))
        recipe = tensor_recipe(definition.reference).model_dump(mode="python")
        recipe["flow"][0]["body"][0]["outputs"]["unused"] = "unused_hidden"
        model = build_model(resolve_recipe(Recipe.model_validate(recipe), {}, {definition.reference.revision_id: definition}))
        hidden = torch.ones(2, 3, 8)
        sizes = output_storage(model, {"hidden": hidden})
        assert len(sizes) == 4
        assert set(sizes.values()) == {5 * hidden.numel() * hidden.element_size()}
        assert all(not module._forward_hooks for module in model.modules())


def test_training_output_storage_does_not_observe_a_cheaper_no_grad_branch():
    """Same-shaped grad-dependent views charge their actual training backing allocation."""
    source = SOURCE.replace('return {"hidden": hidden * self.gain}',
        'value = hidden * self.gain\n        if torch.is_grad_enabled():\n'
        '            wide = torch.cat((value, value, value, value), dim=-1)\n'
        '            return {"hidden": wide[:, :, :hidden.shape[-1]]}\n'
        '        return {"hidden": value}')
    with SourceLoader() as loader:
        definition = loader.load(revision(source))
        model = build_model(resolve_recipe(tensor_recipe(definition.reference), {},
                            {definition.reference.revision_id: definition}))
        hidden = torch.ones(2, 3, 8)
        with torch.no_grad():
            sizes = output_storage(model, {"hidden": hidden})
        assert set(sizes.values()) == {4 * hidden.numel() * hidden.element_size()}
        assert all(not module._forward_hooks for module in model.modules())


def test_missing_or_stale_region_partition_cannot_authorize_a_smaller_estimate():
    """A partition from a different exact graph/occurrence set fails before budget selection."""
    model = build_template("looped")
    args = make_inputs(2, 8, model.facts.vocabulary_size, "cpu")
    captured = capture_complete(CausalCapture(model), args)
    memory = estimate_model(model, captured.exported, allocator_bytes=2 << 30)
    with preserved_model(model):
        storage = capture_replay_storage(model, captured.exported, args)
    for invalid in (replace(storage, outside_bytes=storage.outside_bytes - 1),
                    replace(storage, regions=storage.regions[:-1])):
        with pytest.raises(ValueError, match="partition"):
            captured_recomputation(model.recomputation_targets(), memory, invalid,
                profile_digest="a" * 64, requested=None)


@pytest.mark.parametrize("expression", ["torch.fx.traceback.annotate({})", "torch.compiler.is_compiling()"])
def test_custom_source_cannot_manipulate_engine_capture_metadata(expression):
    """Compiler internals stay outside the supported custom tensor-source surface."""
    source = SOURCE.replace('return {"hidden": hidden * self.gain}', expression + '\n        return {"hidden": hidden * self.gain}')
    with SourceLoader() as loader, pytest.raises(ValueError, match="unsupported"):
        loader.load(revision(source))
