"""Builtin numerical, initializer, sharing and copyable-example acceptance evidence."""

import copy
import hashlib
from pathlib import Path
from uuid import uuid4

import pytest
import torch

from llm.architecture.build import build_model, parameter_groups
from llm.architecture.capture import CausalCapture, capture_complete
from llm.architecture.builtins import Add, CausalAttention, Concat, FeedForwardBlock
from llm.architecture.builtins.catalog import builtin_definitions, compatibility_manifest, copyable_source
from llm.architecture.builtins.recipes import recipe_template
from llm.architecture.errors import ArchitectureError
from llm.architecture.parse import parse_recipe
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader, SourceRevision
from llm.architecture.spec import FinalTokenizer, RevisionRef
from llm.model.transformer import Transformer
from llm.train.model_memory import plan_model

EXAMPLES = Path(__file__).resolve().parents[1] / "examples/architectures"
VALUES = {"dimensions": 16, "heads": 2, "ff_multiplier": 2, "context_window": 12,
          "rope_base": 10000.0, "learned_temperature": True}


def build_template(name: str, values: dict | None = None, vocab: int = 24):
    """Resolve/build a real builtin recipe with explicit tokenizer authority."""
    definitions = builtin_definitions()
    tokenizer = FinalTokenizer(kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=vocab)
    plan = resolve_recipe(recipe_template(name), values or VALUES,
                          {value.reference.revision_id: value for value in definitions.values()}, tokenizer)
    return build_model(plan)


def legacy(values: dict | None = None, vocab: int = 24) -> Transformer:
    """Construct the existing Transformer with the same configured numerical architecture."""
    values = VALUES if values is None else values
    return Transformer(vocab=vocab, dimensions=values["dimensions"], heads=values["heads"],
                       layers=values.get("layers", 2), ff_multiplier=values["ff_multiplier"],
                       ctx_window=values["context_window"], rope_base=values["rope_base"],
                       learned_temperature=values["learned_temperature"])


@pytest.mark.parametrize("temperature", [False, True])
def test_standard_seeded_initialization_logits_intermediates_and_gradients_match_legacy(temperature):
    """Standard construction preserves every initialized tensor, RNG draw and numerical boundary."""
    values = {**VALUES, "learned_temperature": temperature}
    with torch.random.fork_rng():
        torch.manual_seed(84)
        expected = legacy(values)
        rng_after = torch.get_rng_state().clone()
        torch.manual_seed(84)
        actual = build_template("standard", values)
        torch.testing.assert_close(torch.get_rng_state(), rng_after, rtol=0, atol=0)
    inner = actual.instances["model"].model
    for name, tensor in expected.state_dict().items():
        torch.testing.assert_close(inner.state_dict()[name], tensor, rtol=0, atol=0)
    tokens = torch.tensor([[1, 2, 3, 4], [7, 8, 0, 0]])
    positions = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]])
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    reference, _ = expected(tokens, positions=positions, key_mask=mask)
    logits = actual(tokens, positions=positions, key_mask=mask)
    torch.testing.assert_close(logits, reference, rtol=0, atol=0)
    torch.testing.assert_close(inner.embedding(tokens), expected.embedding(tokens), rtol=0, atol=0)
    cos, sin = expected.rope_cos[positions], expected.rope_sin[positions]
    x, y = expected.embedding(tokens), inner.embedding(tokens)
    for left, right in zip(expected.blocks, inner.blocks):
        x, _ = left(x, cos, sin, key_mask=mask)
        y, _ = right(y, cos, sin, key_mask=mask)
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    logits.square().mean().backward()
    reference.square().mean().backward()
    for left, right in zip(expected.parameters(), inner.parameters()):
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    assert len(parameter_groups(actual)) == len(list(expected.parameters()))


def test_standard_recipe_matches_saved_tiny_model_and_pinned_logits(model_module, recorded_trace):
    """Load pinned fixture tensors into the new recipe without changing checkpoint dispatch or trace."""
    source = model_module
    values = {"dimensions": source.dimensions, "heads": source.heads, "layers": source.layers,
              "ff_multiplier": source.ff_multiplier, "context_window": source.ctx_window,
              "rope_base": source.rope_base, "learned_temperature": source.learned_temperature}
    model = build_template("standard", values, source.vocab).eval()
    model.instances["model"].model.load_state_dict(source.state_dict())
    tokens = torch.tensor([recorded_trace["prompt_ids"]])
    logits = model(tokens)
    expected, _ = source(tokens)
    torch.testing.assert_close(logits, expected, atol=2e-6, rtol=1e-5)
    for token_id, value in recorded_trace["logits"]["top"]:
        assert logits[0, -1, token_id].item() == pytest.approx(value, abs=1e-3)


def test_expanded_stack_is_numerically_equivalent_with_identical_parameters():
    """Independent named layers reproduce residual intermediates, logits and gradients of the standard model."""
    source = legacy()
    target = build_template("expanded")
    target.instances["embedding"].weight.data.copy_(source.embedding.weight)
    target.instances["final_norm"].norm.load_state_dict(source.ln_final.state_dict())
    for index, layer in enumerate(source.blocks):
        target.instances[f"layers/{index}"].layer.load_state_dict(layer.state_dict())
    assert target.instances["head"].weight is target.instances["embedding"].weight
    assert target.instances["layers/0"].layer.attn.wqkv.weight is not target.instances["layers/1"].layer.attn.wqkv.weight
    tokens = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]])
    actual = target(tokens)
    expected, _ = source(tokens)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    actual.square().mean().backward()
    expected.square().mean().backward()
    torch.testing.assert_close(target.instances["embedding"].weight.grad, source.embedding.weight.grad)
    for index, layer in enumerate(source.blocks):
        for left, right in zip(target.instances[f"layers/{index}"].layer.parameters(), layer.parameters()):
            torch.testing.assert_close(left.grad, right.grad)
    assert sum(value.numel() for value in target.parameters()) == sum(value.numel() for value in source.parameters())


@pytest.mark.parametrize("name,repetitions", [("looped", 2), ("nested", 4)])
def test_shared_loop_matches_explicit_unroll_values_gradients_and_parameter_counts(name, repetitions):
    """Nested shared repeats retain every gradient and never multiply optimizer parameter ownership."""
    model = build_template(name)
    explicit = copy.deepcopy(model)
    tokens = torch.tensor([[1, 2, 3], [3, 2, 1]])
    positions = torch.arange(3).expand(2, -1)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    actual = model(tokens)
    x = explicit.instances["embedding"](tokens)["hidden"]
    for _ in range(repetitions):
        x = explicit.instances["core"](x, positions, mask)["hidden"]
    x = explicit.instances["final_norm"](x)["hidden"]
    expected = explicit.instances["head"](x)["logits"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.square().mean().backward()
    expected.square().mean().backward()
    for left, right in zip(model.parameters(), explicit.parameters()):
        assert left.grad is not None and torch.isfinite(left.grad).all()
        torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)
    shallow = build_template("looped", {**VALUES, "iterations": 1})
    assert sum(value.numel() for value in model.parameters()) == sum(value.numel() for value in shallow.parameters())
    assert len(parameter_groups(model)) == len(list(model.parameters()))


@pytest.mark.parametrize("name", ["standard", "expanded", "looped", "nested"])
def test_recipes_deterministic_initialization_causality_padding_and_strict_capture(name):
    """Every template preserves deterministic independent sequences and exports the actual full graph."""
    with torch.random.fork_rng():
        torch.manual_seed(126)
        first = build_template(name).eval()
        torch.manual_seed(126)
        second = build_template(name).eval()
    for left, right in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    tokens = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    expected = first(tokens)
    changed = tokens.clone()
    changed[:, -1] = 0
    torch.testing.assert_close(first(changed)[:, :-1], expected[:, :-1])
    for length in (1, 2, 3):
        torch.testing.assert_close(first(tokens[:, :length]), expected[:, :length], atol=2e-6, rtol=1e-5)
    padding = torch.tensor([[True, True, True, True], [True, True, False, False]])
    positions = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]])
    ragged = first(tokens, positions=positions, key_mask=padding)
    torch.testing.assert_close(ragged[1, :2], first(tokens[1:2, :2])[0], atol=2e-6, rtol=1e-5)
    captured = torch.export.export(first, (tokens,), strict=True).module()
    torch.testing.assert_close(captured(tokens), expected)


def test_standard_optional_token_rows_and_recomputation_preserve_legacy_policy():
    """Supported block capabilities preserve mean-row warm-start and exact recomputation gradients."""
    model = build_template("standard")
    block = model.instances["model"]
    table = block.model.embedding.weight
    before = table.detach().clone()
    outcome = block.initialize_token_rows((1, 2, 1, -1, 24))
    assert outcome.initialized_ids == (1, 2) and outcome.skipped_ids == (-1, 24)
    torch.testing.assert_close(table[1], before.mean(0))
    torch.testing.assert_close(table[2], before.mean(0))
    torch.testing.assert_close(table[3:], before[3:])
    reference = copy.deepcopy(model)
    block.configure_recomputation(block.recomputation_targets())
    tokens = torch.tensor([[1, 2, 3]])
    left, right = model(tokens), reference(tokens)
    torch.testing.assert_close(left, right, rtol=0, atol=0)
    left.square().mean().backward()
    right.square().mean().backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)
    with pytest.raises(ArchitectureError):
        block.configure_recomputation((block.recomputation_targets()[0],) * 2)


def test_explicit_add_concat_and_invalid_attention_dimensions():
    """Merge builtins expose explicit operations and do not silently broadcast or accept invalid heads."""
    left, right = torch.randn(2, 3, 4), torch.randn(2, 3, 4)
    torch.testing.assert_close(Add()(left, right)["hidden"], left + right)
    torch.testing.assert_close(Concat()(left, right)["hidden"], torch.cat((left, right), -1))
    with pytest.raises(ArchitectureError):
        Add()(left, right[:1])
    with pytest.raises(ArchitectureError):
        build_template("standard", {**VALUES, "dimensions": 15})


def test_elementary_attention_and_feed_forward_match_legacy_layer_components():
    """Separately copyable primitives retain fused QKV, nonunit temperature, masking and GELU numerics."""
    source = legacy()
    attention = CausalAttention(dimensions=16, heads=2, ctx_window=12,
                                rope_base=10000.0, learned_temperature=True)
    feed_forward = FeedForwardBlock(dimensions=16, ff_multiplier=2)
    with torch.no_grad():
        source.blocks[0].attn.log_temperature.copy_(torch.tensor([0.3, -0.2]))
    attention.attn.load_state_dict(source.blocks[0].attn.state_dict())
    feed_forward.ff.load_state_dict(source.blocks[0].ff.state_dict())
    hidden = torch.randn(2, 4, 16)
    positions = torch.tensor([[0, 1, 2, 3], [0, 1, 0, 0]])
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])
    expected, _ = source.blocks[0].attn(hidden, source.rope_cos[positions], source.rope_sin[positions], key_mask=mask)
    torch.testing.assert_close(attention(hidden, positions, mask)["hidden"], expected, rtol=0, atol=0)
    torch.testing.assert_close(feed_forward(hidden)["hidden"], source.blocks[0].ff(hidden), rtol=0, atol=0)


def test_real_custom_ffn_source_load_capture_and_gradients():
    """The supplied gated FFN captures, admits bounded training memory and differentiates with shared/tied ownership."""
    source = (EXAMPLES / "custom_ffn.py").read_text()
    reference = RevisionRef(kind="block", resource_id=uuid4(), revision_id=uuid4(),
                            content_hash=hashlib.sha256(source.encode()).hexdigest())
    with SourceLoader() as loader:
        definition = loader.load(SourceRevision(reference, source))
        module = definition.module_type(width=16, expansion=3)
        x = torch.randn(2, 4, 16, requires_grad=True)
        result = module(x)["hidden"]
        captured = torch.export.export(module, (x,), strict=True).module()
        torch.testing.assert_close(captured(x)["hidden"], result)
        result.square().mean().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in module.parameters())
        recipe = parse_recipe((EXAMPLES / "custom.yaml").read_text())
        pinned = recipe.modules["core"].reference
        definition = loader.load(SourceRevision(pinned, source))
        definitions = {value.reference.revision_id: value for value in builtin_definitions().values()}
        definitions[pinned.revision_id] = definition
        tokenizer = FinalTokenizer(kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=24)
        model = build_model(resolve_recipe(recipe, {"dimensions": 16, "context_window": 12}, definitions, tokenizer))
        tokens = torch.tensor([[1, 2, 3]])
        logits = model(tokens)
        assert logits.shape == (1, 3, 24)
        captured = capture_complete(CausalCapture(model),
            (tokens, torch.arange(3).reshape(1, 3), torch.ones_like(tokens, dtype=torch.bool)))
        memory = plan_model(model, captured.exported, allocator_bytes=2 << 30)
        assert memory.parameter_bytes == sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
        assert 0 < memory.activation_bytes < memory.target_bytes
        logits.square().mean().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())


def test_manifest_and_yaml_examples_pin_real_implementations():
    """Checked-in copyable YAML and manifests name actual source hashes instead of synthetic contract references."""
    manifest = compatibility_manifest()
    assert manifest["abi"] == "slate-transformer-v1"
    assert len(manifest["sources"]) == 3
    assert len(builtin_definitions()) == 9
    for name in ("standard", "expanded", "looped", "nested"):
        saved = parse_recipe((EXAMPLES / f"{name}.yaml").read_text())
        assert saved == recipe_template(name)


@pytest.mark.parametrize("name", ["standard_transformer", "token_embedding", "transformer_layer", "normalization",
                                 "language_model_head", "causal_attention", "feed_forward", "add", "concat"])
def test_every_builtin_has_an_eligible_executable_custom_copy(name):
    """Copy source uses only the public SDK/builtin seam and executes through real immutable source loading."""
    source = copyable_source(name)
    reference = RevisionRef(kind="block", resource_id=uuid4(), revision_id=uuid4(),
                            content_hash=hashlib.sha256(source.encode()).hexdigest())
    definition = builtin_definitions()[name]
    parameters = {key: declaration.default for key, declaration in definition.declaration.parameters.items()}
    inputs = {}
    for port_name, port in definition.declaration.inputs.items():
        sizes = {"batch": 2, "sequence": 3, "width": 32, "dimensions": 32,
                 "left_width": 16, "right_width": 24}
        shape = tuple(size if type(size) is int else sizes[size] for size in port.shape)
        if port.dtype == "int64":
            inputs[port_name] = torch.arange(3).expand(2, -1)
        elif port.dtype == "bool":
            inputs[port_name] = torch.ones(shape, dtype=torch.bool)
        else:
            inputs[port_name] = torch.randn(shape)
    with SourceLoader() as loader:
        copied = loader.load(SourceRevision(reference, source))
        assert copied.reference.content_hash == hashlib.sha256(source.encode()).hexdigest()
        assert copied.declaration.capabilities == ()
        with torch.random.fork_rng():
            torch.manual_seed(42)
            custom = copied.module_type(**parameters)
            torch.manual_seed(42)
            original = definition.module_type(**parameters)
        actual, expected = custom(**inputs), original(**inputs)
        assert actual.keys() == expected.keys()
        for key in actual:
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU hardware verification deferred by user")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_standard_cuda_placement_matches_legacy_full_prefix(dtype):
    """Keep actual CUDA precision/placement comparisons runnable on the user's NVIDIA machine."""
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("CUDA device does not support bfloat16")
    expected = legacy().to(device="cuda", dtype=dtype).eval()
    actual = build_template("standard").to(device="cuda", dtype=dtype).eval()
    actual.instances["model"].model.load_state_dict(expected.state_dict())
    tokens = torch.tensor([[1, 2, 3, 4]], device="cuda")
    reference, _ = expected(tokens)
    torch.testing.assert_close(actual(tokens), reference, atol=0.03 if dtype != torch.float32 else 2e-6,
                               rtol=0.03 if dtype != torch.float32 else 1e-5)
    assert all(parameter.device.type == "cuda" and parameter.dtype == dtype for parameter in actual.parameters())
