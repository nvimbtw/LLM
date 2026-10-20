"""Strict torch 2.13 capture fixtures for ISS-48, with explicit tensor state."""

import operator

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from llm.model.transformer import Transformer

ARCH = dict(vocab=32, dimensions=16, heads=2, layers=2, ff_multiplier=2, ctx_window=8)
MAX_PROTOTYPE_GRAPH_NODES = 4096
MAX_PROTOTYPE_CALLS = 128


def reject_persistent_mutation(exported: torch.export.ExportedProgram) -> None:
    """Reject writes through graph inputs and their views, allowing fresh local masks.

    This bounded experiment checks ATen alias schemas before functionalization;
    production source eligibility and state snapshots remain separate safeguards.
    """
    assert not any(spec.kind.name in {"USER_INPUT_MUTATION", "BUFFER_MUTATION"}
                   for spec in exported.graph_signature.output_specs)
    protected = set()
    for node in exported.graph.nodes:
        if node.op == "placeholder":
            protected.add(node)
        if node.op != "call_function":
            continue
        leaves = torch.utils._pytree.tree_leaves((node.args, node.kwargs))
        reads_protected = any(isinstance(value, torch.fx.Node) and value in protected
                              for value in leaves)
        if node.target is operator.getitem and reads_protected:
            protected.add(node)
        if not isinstance(node.target, torch._ops.OpOverload):
            continue
        for index, argument in enumerate(node.target._schema.arguments):
            if not argument.alias_info or not argument.alias_info.is_write:
                continue
            value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
            assert not any(isinstance(item, torch.fx.Node) and item in protected
                           for item in torch.utils._pytree.tree_leaves(value))
        if reads_protected and any(result.alias_info for result in node.target._schema.returns):
            protected.add(node)


def transformer(shared: bool = False) -> Transformer:
    """Construct deterministic CPU weights, optionally sharing both execution layers."""
    torch.manual_seed(48)
    model = Transformer(**ARCH)
    if shared:
        model.blocks = nn.ModuleList([model.blocks[0], model.blocks[0]])
    return model


def strict_capture(
    model: nn.Module, args: tuple, kwargs: dict | None = None,
    dynamic_shapes: tuple | dict | None = None,
) -> torch.export.ExportedProgram:
    """Capture one concrete signature completely and reject opaque Python nodes.

    Shapes are concrete unless an explicit bounded dynamic profile is supplied.
    Source policy and process budgets remain separate checks.
    """
    exported = torch.export.export(
        model, args, kwargs=kwargs, dynamic_shapes=dynamic_shapes, strict=True,
    )
    reject_persistent_mutation(exported)
    nodes = list(exported.graph.nodes)
    assert len(nodes) <= MAX_PROTOTYPE_GRAPH_NODES
    for node in nodes:
        assert node.op in {"placeholder", "call_function", "output"}
        if node.op == "call_function":
            assert isinstance(node.target, torch._ops.OpOverload) or node.target in {
                operator.getitem, operator.add, operator.sub, operator.mul,
                operator.floordiv,
            }
            assert node.meta.get("stack_trace")
            assert node.meta.get("nn_module_stack")
    return exported


class StatefulBlock(nn.Module):
    """A custom residual block whose carry is an explicit, caller-owned tensor."""

    def __init__(self, width: int = 8):
        """Create one parameter set and controlled PyTorch dropout for width channels."""
        super().__init__()
        self.projection = nn.Linear(width, width)
        self.dropout = nn.Dropout(0.25)

    def forward(self, hidden: torch.Tensor, carry: torch.Tensor):
        """Return transformed hidden values and fresh carry without input mutation."""
        update = self.dropout(torch.tanh(self.projection(hidden + carry)))
        return hidden + update, carry + update * 0.125


class RepeatedBlock(nn.Module):
    """A fixed nested repeat with one shared parameter owner and explicit carry."""

    def __init__(self, calls: int = 4, recompute: bool = False):
        """Bound expanded work before creating the shared block; reject excess calls."""
        super().__init__()
        if calls < 1 or calls > MAX_PROTOTYPE_CALLS or calls % 2:
            raise ValueError("calls must be even and within [2, 128]")
        self.block = StatefulBlock()
        self.outer = calls // 2
        self.recompute = recompute

    def forward(self, hidden: torch.Tensor, carry: torch.Tensor):
        """Run both fixed loop levels, preserving autograd and optional RNG replay."""
        for _ in range(self.outer):
            for _ in range(2):
                if self.recompute:
                    hidden, carry = checkpoint(
                        self.block, hidden, carry, use_reentrant=False,
                        preserve_rng_state=True,
                    )
                else:
                    hidden, carry = self.block(hidden, carry)
        return hidden, carry


class ScalarBranch(nn.Module):
    """An unsupported input-dependent Python branch that strict capture must reject."""

    def forward(self, hidden: torch.Tensor):
        """Read a tensor scalar as Python control flow to exercise strict rejection."""
        if int(hidden.sum()) > 0:
            return hidden * 2
        return hidden - 2


class WeightBranch(nn.Module):
    """An unsupported parameter-dependent Python branch."""

    def __init__(self):
        """Create the trainable value whose Python specialization must be rejected."""
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, hidden: torch.Tensor):
        """Branch on the current weight rather than captured tensor control flow."""
        if self.weight.item() > 0:
            return hidden * self.weight
        return -hidden * self.weight
