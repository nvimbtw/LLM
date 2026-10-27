"""Complete strict graph capture and bounded operator/mutation metadata extraction."""

import operator
import re
from dataclasses import dataclass
from collections.abc import Mapping

import torch
from torch import Tensor, nn

from llm.architecture.errors import ArchitectureError
from llm.architecture.execute import invoke
from llm.architecture.spec import MAX_GRAPH_EDGES, MAX_GRAPH_NODES

STRUCTURAL_OPERATIONS = frozenset({operator.getitem, operator.add, operator.sub, operator.mul,
                                   operator.floordiv, operator.truediv, operator.mod, operator.eq,
                                   operator.ne, operator.lt, operator.le, operator.gt, operator.ge})
MAX_CAPTURE_TENSOR_BYTES = 64 << 20


@dataclass(frozen=True)
class CapturedGraph:
    """Actual exported arithmetic and bounded inspection metadata for one mode/signature domain."""

    exported: torch.export.ExportedProgram
    nodes: tuple[dict, ...]
    edges: tuple[dict, ...]
    guards: tuple[str, ...]


def reject_mutation(exported: torch.export.ExportedProgram) -> None:
    """Reject graph writes through input/parameter/buffer aliases before functionalization."""
    if any(spec.kind.name in {"USER_INPUT_MUTATION", "BUFFER_MUTATION"}
           for spec in exported.graph_signature.output_specs):
        raise ArchitectureError("capture_mutation", "capture reports persistent/input mutation")
    protected = set()
    for node in exported.graph.nodes:
        if node.op == "placeholder":
            protected.add(node)
        if node.op != "call_function":
            continue
        leaves = torch.utils._pytree.tree_leaves((node.args, node.kwargs))
        reads = any(isinstance(value, torch.fx.Node) and value in protected for value in leaves)
        if node.target is operator.getitem and reads:
            protected.add(node)
        if not isinstance(node.target, torch._ops.OpOverload):
            continue
        for index, argument in enumerate(node.target._schema.arguments):
            if argument.alias_info and argument.alias_info.is_write:
                value = node.args[index] if index < len(node.args) else node.kwargs.get(argument.name)
                if any(isinstance(item, torch.fx.Node) and item in protected
                       for item in torch.utils._pytree.tree_leaves(value)):
                    raise ArchitectureError("capture_mutation", "operation writes through persistent/input storage")
        if reads and any(result.alias_info for result in node.target._schema.returns):
            protected.add(node)


def tensor_metadata(value: object) -> dict:
    """Summarize bounded tensor metadata without serializing values or private tensor objects."""
    if not isinstance(value, Tensor):
        return {}
    return {"value": {"shape": [size if type(size) is int else f"dim{index}" for index, size in enumerate(value.shape)],
                       "dtype": str(value.dtype)}}


def graph_metadata(exported: torch.export.ExportedProgram) -> tuple[tuple[dict, ...], tuple[dict, ...]]:
    """Require explicit supported operations and extract bounded dependency/source/module metadata."""
    nodes, edges = [], []
    ownership = {spec.arg.name: (spec.target,) for spec in exported.graph_signature.input_specs
                 if spec.kind.name == "PARAMETER"}
    for index, node in enumerate(exported.graph.nodes):
        if index >= MAX_GRAPH_NODES:
            raise ArchitectureError("graph_node_limit", "captured graph exceeds node budget")
        if node.op not in {"placeholder", "call_function", "output"}:
            raise ArchitectureError("opaque_capture", "capture includes an opaque Python/module operation")
        if node.op == "call_function" and not (isinstance(node.target, torch._ops.OpOverload) or node.target in STRUCTURAL_OPERATIONS):
            raise ArchitectureError("unknown_operation", "capture contains an unsupported operation estimate")
        stack = node.meta.get("nn_module_stack", {})
        module_path = list(stack.values())[-1][0] if stack else None
        owners = tuple(dict.fromkeys(owner for value in node.all_input_nodes for owner in ownership.get(value.name, ())))
        if node.op == "placeholder":
            owners = ownership.get(node.name, ())
        aliases = isinstance(node.target, torch._ops.OpOverload) and any(value.alias_info for value in node.target._schema.returns)
        if aliases or node.target is operator.getitem:
            ownership[node.name] = owners
        kind = "input" if node.op == "placeholder" else "output" if node.op == "output" else "operation"
        item = {"id": node.name, "kind": kind, "label": str(node.target)[:120],
                "occurrence_path": node.name, "parameter_owner_ids": owners,
                "outputs": tensor_metadata(node.meta.get("val"))}
        if module_path:
            item["instance_path"] = module_path[:512]
        if node.op == "call_function":
            item["operation_target"] = str(node.target)[:256]
        source = re.search(r'blocks/([0-9a-f-]{36})\.py["\']?, line (\d+)', node.meta.get("stack_trace", ""))
        if source:
            item["source"] = {"revision_id": source.group(1), "file": f"blocks/{source.group(1)}.py",
                              "line": int(source.group(2))}
        nodes.append(item)
        for number, source in enumerate(node.all_input_nodes):
            if len(edges) >= MAX_GRAPH_EDGES:
                raise ArchitectureError("graph_edge_limit", "captured graph exceeds edge budget")
            edges.append({"source": source.name, "source_port": "value", "target": node.name,
                          "target_port": f"input_{number}"})
    return tuple(nodes), tuple(edges)


def tensor_output_bytes(value: object, exported: torch.export.ExportedProgram) -> int:
    """Count tensor leaves at admitted symbolic upper bounds without materializing values or guessing shapes."""
    substitutions = {symbol: bounds.upper for symbol, bounds in exported.range_constraints.items()}
    total = 0
    for tensor in torch.utils._pytree.tree_leaves(value):
        if not isinstance(tensor, Tensor):
            continue
        elements = 1
        for size in tensor.shape:
            try:
                maximum = size if type(size) is int else int(size.node.expr.subs(substitutions))
            except (AttributeError, TypeError, ValueError, OverflowError) as error:
                raise ArchitectureError("unknown_allocation", "capture contains an unbounded tensor shape estimate") from error
            if maximum < 0:
                raise ArchitectureError("unknown_allocation", "capture contains an invalid tensor shape estimate")
            elements *= maximum
        total += elements * tensor.element_size()
    return total


def check_tensor_estimates(exported: torch.export.ExportedProgram) -> int:
    """Bound complete fake tensor allocations using admitted symbolic upper bounds; unknown sizes fail closed."""
    total = sum(tensor_output_bytes(node.meta.get("val"), exported) for node in exported.graph.nodes)
    if total > MAX_CAPTURE_TENSOR_BYTES:
        raise ArchitectureError("capture_allocation_limit", "conservative graph tensor allocation estimate exceeds budget")
    return total


def capture_complete(model: nn.Module, args: tuple, dynamic_shapes=None) -> CapturedGraph:
    """Capture the exact eager implementation strictly, rejecting mutation and opaque sections.

    The admitted worker bounds compiler time/memory. The returned graph is for
    inspection/equivalence checks; production arithmetic continues using eager code.
    """
    try:
        exported = torch.export.export(model, args, dynamic_shapes=dynamic_shapes, strict=True)
    except Exception as error:
        raise ArchitectureError("capture_failed", "strict export failed for the configured source/signature") from error
    reject_mutation(exported)
    check_tensor_estimates(exported)
    nodes, edges = graph_metadata(exported)
    guards = tuple(f"{symbol}: {bounds}"[:1024] for symbol, bounds in exported.range_constraints.items())
    if len(guards) > 128:
        raise ArchitectureError("capture_guard_limit", "capture produced too many shape guards")
    return CapturedGraph(exported, nodes, edges, guards)


class NamedCapture(nn.Module):
    """Explicit named-input wrapper for ordinary tensor blocks without LM token assumptions."""

    def __init__(self, module: nn.Module):
        """Own the already constructed block for one bounded validation lifetime."""
        super().__init__()
        self.module = module

    def forward(self, inputs: Mapping[str, Tensor]):
        """Return the declared tensor mapping using the exact validated keyword interface."""
        return invoke(self.module, inputs)


class CausalCapture(nn.Module):
    """One explicit full-prefix signature with token/position/mask shapes linked together."""

    def __init__(self, module: nn.Module):
        """Own a resolved causal adapter without changing its parameters or structural behavior."""
        super().__init__()
        self.module = module

    def forward(self, tokens: Tensor, positions: Tensor, key_mask: Tensor) -> Tensor:
        """Evaluate canonical causal logits from three explicit request-owned inputs."""
        return self.module(tokens, positions=positions, key_mask=key_mask)
