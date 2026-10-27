"""Explicit token-table initialization ownership without fixed-layout assumptions."""

from uuid import uuid4

import pytest
import torch

from llm.architecture.build import build_model
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader
from llm.architecture.spec import FinalTokenizer, Recipe
from llm.train.warm_start import initialize_token_rows, untrained_marker_ids, warm_start_embeddings
from tests.test_architecture_builtins import build_template, legacy
from tests.test_architecture_core import revision


ROW_SOURCE = '''from torch import nn
import torch
from llm.architecture.sdk import block, BlockDeclaration, ParameterDeclaration, TensorPort
from llm.architecture.interfaces import TokenRowResult

@block(BlockDeclaration(sdk_version=1, name="owned_rows", title="Owned token rows",
    parameters={"vocab": ParameterDeclaration(type="integer", default=8, minimum=1),
                "width": ParameterDeclaration(type="integer", default=4, minimum=1),
                "tied": ParameterDeclaration(type="boolean", default=False)},
    inputs={"tokens": TensorPort(dtype="int64", shape=("batch", "sequence"))},
    outputs={"logits": TensorPort(dtype="floating", shape=("batch", "sequence", "vocabulary"))},
    capabilities=("token_rows",)))
class OwnedRows(nn.Module):
    """Own either tied or independent input/output tables with no embedding attribute."""

    def __init__(self, vocab, width, tied):
        """Create explicit token tables and an optional exact Parameter alias."""
        super().__init__()
        self.lookup = nn.Embedding(vocab, width)
        self.readout = nn.Linear(width, vocab, bias=False)
        if tied:
            self.readout.weight = self.lookup.weight

    def forward(self, tokens):
        """Return token-local causal vocabulary logits."""
        return {"logits": self.readout(self.lookup(tokens))}

    def initialize_token_rows(self, token_ids):
        """Initialize every owned table once from its own original mean and report unsupported IDs."""
        valid = tuple(token_id for token_id in token_ids if 0 <= token_id < self.lookup.weight.shape[0])
        skipped = tuple(token_id for token_id in token_ids if token_id not in valid)
        tables = {id(table): table for table in (self.lookup.weight, self.readout.weight)}
        with torch.no_grad():
            for table in tables.values():
                mean = table.mean(dim=0)
                for token_id in valid:
                    table[token_id].copy_(mean)
        return TokenRowResult("initialized", valid, skipped)
'''


def row_recipe(reference) -> Recipe:
    """Wrap an explicit custom causal block once, preserving the complete named interface for capability delegation."""
    return Recipe.model_validate({"schema_version": 1, "interface": "causal_lm", "parameters": {},
        "inputs": {"tokens": {"dtype": "int64", "shape": ["batch", "sequence"]}},
        "outputs": {"logits": {"dtype": "floating", "shape": ["batch", "sequence", "vocabulary"]}},
        "result": {"logits": "logits"}, "context_window": 4,
        "modules": {"rows": {"reference": reference, "parameters": {
            "vocab": "${tokenizer.vocabulary_size}", "width": 4, "tied": False}}},
        "flow": [{"call": "rows", "inputs": {"tokens": "tokens"}, "outputs": {"logits": "logits"}}]})


def assert_changed_rows(before, after, rows):
    """Require only requested token rows to use the original table mean; all other values remain bit-identical."""
    expected = before.clone()
    for token_id in rows:
        expected[token_id] = before.mean(dim=0)
    torch.testing.assert_close(after, expected, rtol=0, atol=0)


def test_standard_and_legacy_marker_initialization_preserve_mean_policy_and_aliases():
    """Standard whole-model delegation matches the deliberate legacy adapter and resets duplicate IDs once."""
    reference = legacy()
    model = build_template("standard")
    inner = model.instances["model"].model
    inner.load_state_dict(reference.state_dict())
    before = reference.embedding.weight.detach().clone()
    ids = [3, 3, 4, -1, before.shape[0]]
    first = initialize_token_rows(reference, ids)
    second = initialize_token_rows(model, ids)
    assert first == second
    assert first.initialized_ids == (3, 4) and first.skipped_ids == (-1, before.shape[0])
    assert_changed_rows(before, inner.embedding.weight, [3, 4])
    torch.testing.assert_close(reference.embedding.weight, inner.embedding.weight, rtol=0, atol=0)
    assert warm_start_embeddings(model, []) == 0


@pytest.mark.parametrize("tied", [False, True])
def test_declared_custom_initializer_owns_tied_and_untied_tables(tied):
    """Actual eligible custom source chooses each table's policy without a model.embedding convention."""
    with SourceLoader() as loader:
        definition = loader.load(revision(ROW_SOURCE))
        recipe = row_recipe(definition.reference)
        modules = dict(recipe.modules)
        modules["rows"] = modules["rows"].model_copy(update={"parameters": {
            "vocab": "${tokenizer.vocabulary_size}", "width": 4, "tied": tied}})
        recipe = recipe.model_copy(update={"modules": modules})
        tokenizer = FinalTokenizer(kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=8)
        model = build_model(resolve_recipe(recipe, {}, {definition.reference.revision_id: definition}, tokenizer))
        owner = model.instances["rows"]
        before = [value.detach().clone() for value in (owner.lookup.weight, owner.readout.weight)]
        result = initialize_token_rows(model, [1, 2, 1, 99])
        assert result.initialized_ids == (1, 2) and result.skipped_ids == (99,)
        for original, actual in zip(before, (owner.lookup.weight, owner.readout.weight)):
            assert_changed_rows(original, actual, [1, 2])
        assert (owner.lookup.weight is owner.readout.weight) is tied


@pytest.mark.parametrize("name", ["looped", "nested"])
def test_missing_whole_model_capability_skips_without_changing_weights_or_training(name):
    """Composed models retain weights and usable gradients when no whole-model table initializer is declared."""
    model = build_template(name)
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    result = initialize_token_rows(model, [1, 2, 1])
    assert result.status == "skipped_unsupported" and result.skipped_ids == (1, 2)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    model(torch.tensor([[1, 2, 3]])).square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_initializer_requires_declaration_and_refuses_bad_ids_before_mutation():
    """Method shape alone cannot authorize mutation, and invalid external IDs cannot reach a declared implementation."""
    from torch import nn

    class Undeclared(nn.Module):
        """A method resembling a capability without any accepted declaration."""

        def initialize_token_rows(self, token_ids):
            """Fail if the adapter infers capability authorization from method presence."""
            pytest.fail("undeclared initialization was invoked")

    assert initialize_token_rows(Undeclared(), [1]).status == "skipped_unsupported"
    model = build_template("standard")
    for invalid in ([True], [1.0], ["1"], [0] * 65537):
        with pytest.raises(ValueError, match="integer IDs"):
            initialize_token_rows(model, invalid)
    assert untrained_marker_ids([2, 3, 3], None) == []
    assert untrained_marker_ids([2, 3, 3], [2]) == [3]
