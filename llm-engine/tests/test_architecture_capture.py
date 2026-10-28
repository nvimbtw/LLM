"""ISS-48 capture feasibility evidence, not a production validation implementation."""

import copy
import io
import time

import pytest
import torch
from torch import nn

from tests.prototypes.capture import (
    ARCH, MAX_PROTOTYPE_CALLS, RepeatedBlock, ScalarBranch, WeightBranch,
    strict_capture, transformer,
)


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """Keep bounded CPU experiments reproducible and restore global RNG/threads."""
    threads = torch.get_num_threads()
    with torch.random.fork_rng():
        torch.set_num_threads(1)
        torch.manual_seed(48)
        yield
    torch.set_num_threads(threads)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("batch,sequence", [(1, 1), (1, 8), (3, 1), (3, 8)])
def test_strict_transformer_boundary_shapes(training, batch, sequence):
    """Capture each full-prefix boundary signature in both modes without graph breaks."""
    model = transformer().train(training)
    tokens = torch.randint(0, ARCH["vocab"], (batch, sequence))
    exported = strict_capture(model, (tokens,))
    torch.testing.assert_close(exported.module()(tokens)[0], model(tokens)[0])
    assert exported.range_constraints == {}
    with pytest.raises((RuntimeError, AssertionError)):
        exported.module()(tokens.repeat(2, 1))


@pytest.mark.parametrize("training", [False, True])
def test_ragged_positions_are_runtime_tensors(training):
    """Different positions/masks reuse one receipt; bad positions remain rejected."""
    model = transformer().train(training)
    tokens = torch.randint(0, ARCH["vocab"], (3, 8))
    positions = torch.arange(8).expand(3, -1).clone()
    mask = torch.ones_like(tokens, dtype=torch.bool)
    kwargs = {"positions": positions, "key_mask": mask}
    exported = strict_capture(model, (tokens,), kwargs)
    assert any("_assert_async" in str(n.target) for n in exported.graph.nodes)
    assert not any("_local_scalar_dense" in str(n.target) for n in exported.graph.nodes)
    for pad in (0, 3, 7):
        positions[1] = torch.arange(8).sub(pad).clamp_min(0)
        mask[1] = torch.arange(8) >= pad
        torch.testing.assert_close(exported.module()(tokens, **kwargs)[0], model(tokens, **kwargs)[0])
    for invalid in (-1, 8):
        bad_positions = positions.clone()
        bad_positions[0, 0] = invalid
        with pytest.raises(ValueError):
            model(tokens, positions=bad_positions, key_mask=mask)
        with pytest.raises(RuntimeError, match="positions must be within"):
            exported.module()(tokens, positions=bad_positions, key_mask=mask)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("prefix", [1, 7])
def test_prefill_decode_and_occurrence_owned_caches(shared, prefix):
    """Capture prefill/decode through the context boundary and isolate shared-layer state."""
    model = transformer(shared).eval()
    tokens = torch.randint(0, ARCH["vocab"], (1, prefix + 1))
    prompt = tokens[:, :prefix]
    prefill = strict_capture(model, (prompt,), {"caches": model.empty_caches()})
    _, caches = prefill.module()(prompt, caches=model.empty_caches())
    saved_caches = [(key.clone(), value.clone()) for key, value in caches]
    kwargs = {"caches": caches, "pos_offset": prefix}
    decode = strict_capture(model, (tokens[:, -1:],), kwargs)
    logits, updated = decode.module()(tokens[:, -1:], **kwargs)
    torch.testing.assert_close(logits, model(tokens)[0][:, -1:], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(logits, model(tokens[:, -1:], **kwargs)[0])
    assert len(updated) == 2
    assert updated[0][0].shape[2] == prefix + 1
    assert updated[0][0].data_ptr() != updated[1][0].data_ptr()
    for old, saved in zip(caches, saved_caches):
        torch.testing.assert_close(old, saved)
    if shared:
        assert model.blocks[0] is model.blocks[1]
        assert not torch.equal(updated[0][0], updated[1][0])


def test_ragged_cached_decode_matches_full_prefix():
    """Capture the explicit-position masked prefill and single-token decode signatures."""
    model = transformer().eval()
    tokens = torch.tensor([[0, 0, 3, 4], [2, 6, 7, 8]])
    positions = torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]])
    mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
    kwargs = {"positions": positions, "key_mask": mask, "caches": model.empty_caches()}
    prefill = strict_capture(model, (tokens,), kwargs)
    _, caches = prefill.module()(tokens, **kwargs)
    next_tokens = torch.tensor([[5], [9]])
    next_positions = positions[:, -1:] + 1
    next_mask = torch.cat((mask, torch.ones(2, 1, dtype=torch.bool)), 1)
    decode_kwargs = {"caches": caches, "positions": next_positions, "key_mask": next_mask}
    decode = strict_capture(model, (next_tokens,), decode_kwargs)
    actual, _ = decode.module()(next_tokens, **decode_kwargs)
    expected, _ = model(torch.cat((tokens, next_tokens), 1),
                        positions=torch.cat((positions, next_positions), 1), key_mask=next_mask)
    torch.testing.assert_close(actual, expected[:, -1:], atol=1e-6, rtol=1e-5)


def test_causality_prefix_and_batch_independence():
    """Suffix edits, prefix evaluation, padding and unrelated rows preserve live logits."""
    model = transformer().eval()
    tokens = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])
    whole, _ = model(tokens)
    changed = tokens.clone()
    changed[:, 4:] += 10
    torch.testing.assert_close(model(changed)[0][:, :4], whole[:, :4])
    for length in range(1, 9):
        torch.testing.assert_close(model(tokens[:, :length])[0], whole[:, :length], atol=1e-6, rtol=1e-5)
    batch = torch.cat((torch.cat((torch.zeros(1, 4, dtype=torch.long), tokens[:, :4]), 1), tokens), 0)
    positions = torch.tensor([[0, 0, 0, 0, 0, 1, 2, 3], list(range(8))])
    mask = torch.tensor([[False] * 4 + [True] * 4, [True] * 8])
    exported = strict_capture(model, (batch,), {"positions": positions, "key_mask": mask})
    logits, _ = exported.module()(batch, positions=positions, key_mask=mask)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[0, 4:], whole[0, :4], atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(logits[1], whole[0], atol=1e-6, rtol=1e-5)


def test_capture_preserves_source_and_unique_parameter_ownership():
    """Shared calls retain operator/source attribution and one learned parameter owner."""
    shared, independent = transformer(True), transformer(False)
    tokens = torch.ones(1, 8, dtype=torch.long)
    exported = strict_capture(shared, (tokens,))
    targets = [spec.target for spec in exported.graph_signature.input_specs if spec.kind.name == "PARAMETER"]
    assert len(targets) == len(set(targets)) == len(list(shared.parameters()))
    assert sum(p.numel() for p in shared.parameters()) < sum(p.numel() for p in independent.parameters())
    linears = [node for node in exported.graph.nodes if node.target == torch.ops.aten.linear.default]
    assert len(linears) == 8
    assert all("transformer.py" in node.meta["stack_trace"] for node in linears)
    assert all(node.meta.get("tensor_meta") for node in linears)
    assert any("blocks.0.attn.wqkv" in str(node.meta["nn_module_stack"]) for node in linears)
    projections = [node for node in linears if "attn.wqkv" in str(node.meta["nn_module_stack"])]
    assert len(projections) == 2
    assert projections[0].name != projections[1].name


@pytest.mark.parametrize("training", [False, True])
def test_custom_dropout_capture_rng_gradients_and_explicit_state(training):
    """Compare captured/eager repeated computation, gradients, RNG and immutable carry."""
    model = RepeatedBlock().train(training)
    hidden, carry = torch.randn(2, 3, 8), torch.zeros(2, 3, 8)
    before = carry.clone()
    state = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    exported = strict_capture(model, (hidden, carry))
    torch.testing.assert_close(torch.get_rng_state(), rng)
    captured = exported.module()
    torch.manual_seed(17)
    expected = model(hidden, carry)
    rng_after = torch.get_rng_state()
    expected[0].square().mean().backward()
    gradients = [p.grad.clone() for p in model.parameters()]
    model.zero_grad(set_to_none=True)
    torch.manual_seed(17)
    actual = captured(hidden, carry)
    torch.testing.assert_close(torch.get_rng_state(), rng_after)
    torch.testing.assert_close(actual, expected)
    actual[0].square().mean().backward()
    for parameter, gradient in zip(model.parameters(), gradients):
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        torch.testing.assert_close(parameter.grad, gradient)
    torch.testing.assert_close(carry, before)
    torch.testing.assert_close(model.state_dict(), state)
    assert any(node.target == torch.ops.aten.dropout.default for node in exported.graph.nodes)
    assert len(list(model.parameters())) == 2
    assert bool(gradients[0].abs().sum() > 0)


def test_nested_repeat_matches_explicit_unroll_and_recomputation():
    """All four shared calls participate in autograd; RNG replay survives state reload."""
    model = RepeatedBlock().train()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    buffer.seek(0)
    recomputed = RepeatedBlock(recompute=True).train()
    recomputed.load_state_dict(torch.load(buffer, weights_only=True))
    unrolled = copy.deepcopy(model)
    hidden, carry = torch.randn(2, 3, 8), torch.zeros(2, 3, 8)
    results, gradients = [], []
    for candidate in (model, recomputed, unrolled):
        torch.manual_seed(17)
        if candidate is unrolled:
            x, state = candidate.block(hidden, carry)
            x, state = candidate.block(x, state)
            x, state = candidate.block(x, state)
            result = candidate.block(x, state)
        else:
            result = candidate(hidden, carry)
        result[0].square().mean().backward()
        results.append(result)
        gradients.append([p.grad for p in candidate.parameters()])
    for index in (1, 2):
        torch.testing.assert_close(results[0], results[index])
        torch.testing.assert_close(gradients[0], gradients[index])


@pytest.mark.parametrize("model", [ScalarBranch(), WeightBranch()])
def test_input_and_weight_python_specialization_is_rejected(model):
    """Strict export refuses Python control flow derived from inputs or weights."""
    with pytest.raises((torch._dynamo.exc.UserError, torch._dynamo.exc.Unsupported)):
        strict_capture(model, (torch.ones(2, 3, 8),))


def test_work_and_graph_envelope(record_property):
    """Measure the bounded 128-call prototype without claiming production capacity."""
    model = RepeatedBlock(calls=MAX_PROTOTYPE_CALLS).eval()
    args = (torch.randn(1, 1, 8), torch.zeros(1, 1, 8))
    start = time.monotonic()
    exported = strict_capture(model, args)
    record_property("capture_seconds", round(time.monotonic() - start, 4))
    record_property("graph_nodes", len(list(exported.graph.nodes)))
    torch.testing.assert_close(exported.module()(*args), model(*args))
    with pytest.raises(ValueError, match="calls"):
        RepeatedBlock(calls=MAX_PROTOTYPE_CALLS + 2)


def test_bounded_dynamic_batch_profile():
    """A declared batch range supports singleton batches at one exact sequence length."""
    model = transformer().eval()
    tokens = torch.ones(2, 4, dtype=torch.long)
    batch = torch.export.Dim("batch", min=1, max=3)
    exported = strict_capture(model, (tokens,), dynamic_shapes=({0: batch},))
    assert len(exported.range_constraints) == 1
    for rows in (1, 3):
        args = (torch.randint(0, ARCH["vocab"], (rows, 4)),)
        torch.testing.assert_close(exported.module()(*args), model(*args))
    for shape in ((4, 4), (3, 8)):
        with pytest.raises((RuntimeError, AssertionError)):
            exported.module()(torch.ones(shape, dtype=torch.long))


def test_bounded_dynamic_sequence_profile():
    """Two range receipts cover every full-prefix length without per-token capture."""
    model = transformer().eval()
    sequence = torch.export.Dim("sequence", min=2, max=8)
    exported = strict_capture(model, (torch.ones(2, 4, dtype=torch.long),),
                              dynamic_shapes=({1: sequence},))
    assert len(exported.range_constraints) == 1
    for length in range(2, 9):
        tokens = torch.randint(0, ARCH["vocab"], (2, length))
        torch.testing.assert_close(exported.module()(tokens), model(tokens))
    with pytest.raises((RuntimeError, AssertionError)):
        exported.module()(torch.ones(2, 9, dtype=torch.long))


@pytest.mark.parametrize("shared", [False, True])
def test_transformer_captured_backward_preserves_all_gradients(shared):
    """Real next-token loss and every unique gradient match eager through both call styles."""
    model = transformer(shared).train()
    tokens = torch.randint(0, ARCH["vocab"], (2, 8))
    exported = strict_capture(model, (tokens,))
    captured = exported.module()
    losses, gradients = [], []
    for candidate in (model, captured):
        candidate.zero_grad(set_to_none=True)
        logits, _ = candidate(tokens)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 32), tokens[:, 1:].reshape(-1))
        assert torch.isfinite(loss)
        loss.backward()
        losses.append(loss.detach())
        grads = [parameter.grad.clone() for parameter in model.parameters()]
        assert all(torch.isfinite(gradient).all() for gradient in grads)
        gradients.append(grads)
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])


def test_extended_rope_signature_preserves_table_boundary():
    """Capture the existing context-extension override without changing saved buffers."""
    model = transformer().eval()
    rope = model.extended_rope(16)
    tokens = torch.tensor([[1]])
    kwargs = {"rope": rope, "pos_offset": 15}
    exported = strict_capture(model, (tokens,), kwargs)
    torch.testing.assert_close(exported.module()(tokens, **kwargs), model(tokens, **kwargs))
    assert model.rope_cos.shape[0] == 8
    with pytest.raises(ValueError, match="exceeds context"):
        model(tokens, rope=rope, pos_offset=16)


def test_eval_dropout_keeps_autograd_for_policy_recomputation():
    """Evaluation is deterministic while gradients remain available for an RL update."""
    model = RepeatedBlock().eval()
    args = (torch.randn(2, 3, 8), torch.zeros(2, 3, 8))
    first = model(*args)
    torch.manual_seed(912)
    second = model(*args)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    second[0].square().mean().backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in model.parameters())


@pytest.mark.parametrize("training", [False, True])
def test_dynamic_ragged_prefill_and_decode_domain(training):
    """One prefill and one decode graph cover all non-singleton lengths in the profile."""
    model = transformer().train(training)
    batch = torch.export.Dim("batch", min=1, max=3)
    sequence = torch.export.Dim("sequence", min=2, max=8)
    tokens = torch.ones(2, 4, dtype=torch.long)
    positions = torch.arange(4).expand(2, -1)
    mask = torch.ones_like(tokens, dtype=torch.bool)
    kwargs = {"positions": positions, "key_mask": mask, "caches": model.empty_caches()}
    shapes = {"idx": {0: batch, 1: sequence}, "positions": {0: batch, 1: sequence},
              "key_mask": {0: batch, 1: sequence}, "caches": [(None, None)] * 2}
    prefill = strict_capture(model, (tokens,), kwargs, dynamic_shapes=shapes)
    singleton_kwargs = {"positions": positions[:, :1], "key_mask": mask[:, :1],
                        "caches": model.empty_caches()}
    singleton_shapes = {"idx": {0: batch}, "positions": {0: batch},
                        "key_mask": {0: batch}, "caches": [(None, None)] * 2}
    singleton = strict_capture(model, (tokens[:, :1],), singleton_kwargs,
                               dynamic_shapes=singleton_shapes)
    _, caches = model(tokens, caches=model.empty_caches())
    past = torch.export.Dim("past", min=1, max=7)
    kwargs = {"caches": caches, "positions": torch.full((2, 1), 4),
              "key_mask": torch.ones(2, 5, dtype=torch.bool)}
    shapes = {"idx": {0: batch}, "positions": {0: batch}, "key_mask": {0: batch, 1: past + 1},
              "caches": [({0: batch, 2: past}, {0: batch, 2: past})] * 2}
    decode = strict_capture(model, (tokens[:, :1],), kwargs, dynamic_shapes=shapes)
    for rows in (1, 3):
        for length in range(1, 9):
            prompt = torch.randint(0, ARCH["vocab"], (rows, length))
            pos = torch.arange(length).expand(rows, -1)
            live = torch.ones_like(prompt, dtype=torch.bool)
            if length == 1:
                _, state = singleton.module()(prompt, positions=pos, key_mask=live,
                                              caches=model.empty_caches())
            else:
                _, state = prefill.module()(prompt, positions=pos, key_mask=live, caches=model.empty_caches())
            if length == 8:
                with pytest.raises((RuntimeError, AssertionError)):
                    decode.module()(prompt[:, :1], caches=state,
                                    positions=torch.full((rows, 1), length),
                                    key_mask=torch.ones(rows, length + 1, dtype=torch.bool))
                continue
            next_tokens = torch.randint(0, ARCH["vocab"], (rows, 1))
            actual, _ = decode.module()(next_tokens, caches=state,
                                       positions=torch.full((rows, 1), length),
                                       key_mask=torch.ones(rows, length + 1, dtype=torch.bool))
            expected, _ = model(torch.cat((prompt, next_tokens), 1))
            torch.testing.assert_close(actual, expected[:, -1:], atol=1e-6, rtol=1e-5)
    assert len(decode.range_constraints) == 3


def test_dynamic_512_token_profile_has_bounded_graph_size():
    """A practical context range captures once and agrees at 512 without unrolling length."""
    from llm.model.transformer import Transformer

    model = Transformer(**{**ARCH, "ctx_window": 512}).eval()
    shapes = ({0: torch.export.Dim("batch", min=1, max=3),
               1: torch.export.Dim("sequence", min=2, max=512)},)
    exported = strict_capture(model, (torch.ones(2, 4, dtype=torch.long),), dynamic_shapes=shapes)
    assert len(list(exported.graph.nodes)) < 256
    for length in (2, 16, 512):
        tokens = torch.randint(0, ARCH["vocab"], (1, length))
        torch.testing.assert_close(exported.module()(tokens), model(tokens))


@pytest.mark.parametrize("target", ["input", "buffer", "view"])
def test_mutating_input_or_persistent_buffer_is_rejected(target):
    """Captured writes are inspectable and rejected despite successful strict export."""
    class MutatingBlock(nn.Module):
        """An ineligible custom block with explicit unsupported storage writes."""

        def __init__(self):
            """Create the persistent counter and immutable mutation target."""
            super().__init__()
            self.register_buffer("counter", torch.zeros(1))
            self.target = target

        def forward(self, hidden):
            """Mutate a caller tensor and persistent state, violating the SDK contract."""
            if self.target == "input":
                hidden.add_(1)
            elif self.target == "view":
                hidden.view_as(hidden).add_(1)
            else:
                self.counter.add_(1)
            return hidden + self.counter

    with pytest.raises(AssertionError):
        strict_capture(MutatingBlock(), (torch.ones(2, 3, 8),))


def test_export_does_not_certify_ambient_python_state():
    """Expose frozen Python state so export success cannot replace source eligibility."""
    ambient = {"scale": 2.0}

    class AmbientBlock(nn.Module):
        """An explicitly ineligible fixture dependent on mutable ambient state."""

        def forward(self, hidden):
            """Read external Python state that tracing can silently specialize."""
            return hidden * ambient["scale"]

    model = AmbientBlock()
    tokens = torch.ones(2, 3, 8)
    exported = torch.export.export(model, (tokens,), strict=True)
    ambient["scale"] = 3.0
    assert not torch.equal(exported.module()(tokens), model(tokens))
