"""Actual stateless attention values and explicit call identities for standard, nested/shared and custom modules."""

from contextlib import contextmanager
import copy
from uuid import uuid4

import pytest
import torch

from llm.architecture.build import build_model
from llm.architecture.builtins.catalog import builtin_definitions, copyable_source
from llm.architecture.builtins.recipes import recipe_template
from llm.architecture.capture import capture_complete
from llm.architecture.errors import ArchitectureError
from llm.architecture.inspection import inspection_bindings, PreparedInspection
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader
from llm.architecture.spec import FinalTokenizer
from llm.orchestrate.attention import _record_one_head
from tests.test_architecture_builtins import build_template, VALUES
from tests.test_architecture_core import revision


@contextmanager
def custom_attention_model(*, mutate=False):
    """Load an actual version-pinned custom layer wrapper with explicitly delegated inspection, optionally violating input ownership."""
    source = copyable_source("transformer_layer").replace("'capabilities': ()", "'capabilities': ('introspection', 'attention')")
    source += '''
    def inspection_points(self):
        """Declare only the underlying implemented attention point."""
        return self.implementation.inspection_points()

    def inspect_named(self, inputs, *, point_id):
        """Measure the version-pinned layer on its actual caller-owned tensors."""
        return self.implementation.inspect_named(inputs, point_id=point_id)
'''
    if mutate:
        source = source.replace("return self.implementation.inspect_named(inputs, point_id=point_id)",
            'inputs["hidden"].add_(1)\n        return self.implementation.inspect_named(inputs, point_id=point_id)')
    with SourceLoader() as loader:
        block = loader.load(revision(source))
        definitions = {item.reference.revision_id: item for item in builtin_definitions().values()}
        definitions[block.reference.revision_id] = block
        raw = recipe_template("looped").model_dump(mode="json")
        raw["modules"]["core"]["reference"] = block.reference.model_dump(mode="json")
        recipe = type(recipe_template("looped")).model_validate(raw)
        tokenizer = FinalTokenizer(kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=24)
        yield build_model(resolve_recipe(recipe, VALUES, definitions, tokenizer))


@pytest.mark.parametrize("name,count", [("standard", 2), ("expanded", 2), ("looped", 2), ("nested", 4)])
def test_attention_points_preserve_actual_owner_and_repeated_call_identity(name, count):
    """Repeated shared calls have distinct views but retain one instance/parameter owner and produce actual causal normalized matrices."""
    model = build_template(name).train()
    modes = tuple(module.training for module in model.modules())
    rng = torch.get_rng_state().clone()
    points = model.inspection_points()
    assert len(points) == count and len({point.id for point in points}) == count
    assert all(point.heads == 2 for point in points)
    if name in {"looped", "nested"}:
        assert len({point.occurrence.instance_path for point in points}) == 1
        assert len({point.occurrence.call_path for point in points}) == count
    tokens = torch.tensor([[1, 2, 3, 4]])
    values = [model.inspect(tokens, point_id=point.id) for point in points]
    for value in values:
        assert value.shape == (1, 2, 4, 4) and torch.isfinite(value).all()
        torch.testing.assert_close(value.sum(-1), torch.ones_like(value.sum(-1)), rtol=2e-6, atol=2e-6)
        assert value.triu(1).count_nonzero() == 0
    if count > 1:
        assert any(not torch.equal(values[0], value) for value in values[1:])
    assert torch.equal(torch.get_rng_state(), rng)
    assert tuple(module.training for module in model.modules()) == modes
    assert all(parameter.grad is None for parameter in model.parameters())


def test_standard_attention_matches_existing_recording_without_changing_flags_or_buffers():
    """The declared view equals legacy actual-forward attention and leaves heterogeneous mode/recording state untouched."""
    model = build_template("standard").train()
    inner = model.instances["model"].model
    reference = copy.deepcopy(inner).eval()
    tokens = torch.tensor([[1, 2, 3, 4]])
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    for index, point in enumerate(model.inspection_points()):
        actual = model.inspect(tokens, point_id=point.id)[0, 1]
        expected = torch.from_numpy(_record_one_head(reference, tokens, index, 1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert not any(block.attn.record_attention or block.attn.last_attention is not None for block in inner.blocks)
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


@pytest.mark.parametrize("name", ["standard", "nested"])
def test_prepared_attention_path_has_a_complete_strict_dynamic_capture(name):
    """The actual observation path captures complete operations over bounded sequence lengths and agrees with eager outputs."""
    model = build_template(name).eval()
    binding = inspection_bindings(model)[-1]
    inspector = PreparedInspection(model, binding)
    tokens = torch.tensor([[1, 2, 3]])
    args = (tokens, torch.arange(3)[None], torch.ones_like(tokens, dtype=torch.bool))
    axis = {1: torch.export.Dim("sequence", min=2, max=8)}
    with torch.no_grad():
        captured = capture_complete(inspector, args, dynamic_shapes=(axis, axis, axis))
        for sequence in (2, 5, 8):
            ids = torch.arange(sequence)[None]
            values = (ids, torch.arange(sequence)[None], torch.ones_like(ids, dtype=torch.bool))
            torch.testing.assert_close(captured.exported.module()(*values), inspector(*values), rtol=2e-5, atol=2e-6)


def test_actual_custom_delegated_attention_works_and_input_mutation_is_rejected():
    """Eligible administrator source can delegate real views; an inspector that mutates its caller cannot change later model computation."""
    with custom_attention_model() as model:
        assert len(model.inspection_points()) == 2
        value = model.inspect(torch.tensor([[1, 2, 3]]), point_id=model.inspection_points()[1].id)
        assert value.shape == (1, 2, 3, 3)
    with custom_attention_model(mutate=True) as model:
        with pytest.raises(ArchitectureError, match="mutated"):
            model.inspect(torch.tensor([[1, 2, 3]]), point_id=model.inspection_points()[0].id)


def test_unknown_attention_point_and_invalid_tokens_fail_before_observation():
    """Unknown identities and unsupported causal inputs never select another module or perform a measurement."""
    model = build_template("looped")
    with pytest.raises(ArchitectureError, match="no declared matching"):
        model.inspect(torch.tensor([[1, 2]]), point_id="missing")
    with pytest.raises(ArchitectureError, match="rank-two int64"):
        model.inspect(torch.tensor([[1.0, 2.0]]), point_id=model.inspection_points()[0].id)
