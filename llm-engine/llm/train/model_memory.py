"""Conservative custom-model planning from real bounded exports and unique tensor ownership."""

from dataclasses import asdict, dataclass

import torch
from torch import nn

from llm.architecture.capture import STRUCTURAL_OPERATIONS, check_tensor_estimates
from llm.architecture.errors import ArchitectureError
from llm.train.memory import planning_target

SUPPORTED_OPERATIONS = frozenset({
    "__and__", "__iand__", "_assert_async", "_assert_tensor_metadata", "add", "all", "arange",
    "clone", "contiguous", "embedding", "exp", "expand", "flatten", "ge", "gelu", "index",
    "index_put_", "layer_norm", "lift_fresh_copy", "linear", "lt", "matmul", "mul", "numpy_T",
    "ones", "permute", "scaled_dot_product_attention", "slice", "stack", "sub", "to", "transpose",
    "tril", "unbind", "unsqueeze", "view", "cat", "div", "sum", "mean", "square", "pow", "masked_fill", "bitwise_not",
    "relu", "sigmoid", "silu", "tanh", "softmax", "_softmax", "dropout", "native_dropout", "rand_like",
    "zeros", "zeros_like", "ones_like", "reshape", "detach", "alias", "select", "t", "addmm", "mm", "bmm", "scalar_tensor",
})


@dataclass(frozen=True)
class ModelMemoryPlan:
    """A supported conservative estimate under one parent-owned allocator admission ceiling."""

    parameter_bytes: int
    optimizer_bytes: int
    activation_bytes: int
    reference_bytes: int
    resident_bytes: int
    allocator_bytes: int
    target_bytes: int
    input_shapes: tuple[tuple[int,...],...] = ()

    def __post_init__(self) -> None:
        """Reject invalid byte units or ineffective allocator/target bounds before this evidence can reach a training loop."""
        values = (self.parameter_bytes,self.optimizer_bytes,self.activation_bytes,self.reference_bytes,
                  self.resident_bytes,self.allocator_bytes,self.target_bytes)
        if any(type(value) is not int or value < 0 for value in values) or not 0 <= self.target_bytes <= self.allocator_bytes or self.allocator_bytes < 1:
            raise ValueError("invalid captured model memory plan")
        if any(type(size) is not int or size < 0 for shape in self.input_shapes for size in shape):
            raise ValueError("invalid captured input shape")

    @property
    def estimated_bytes(self) -> int:
        """Return the full training/reference/resident estimate without discounting shared execution."""
        return self.parameter_bytes + self.optimizer_bytes + self.activation_bytes + self.reference_bytes + self.resident_bytes

    @property
    def budget_mib(self) -> float:
        """Report the actual allocator admission ceiling in binary mebibytes."""
        return self.allocator_bytes / 2**20

    @property
    def reason(self) -> str:
        """Describe the supported captured estimate and planning target without guessed layer formulas."""
        return f"Captured model estimate {self.estimated_bytes / 2**20:.1f} MiB within {self.target_bytes / 2**20:.1f} MiB target"

    def as_dict(self) -> dict:
        """Retain exact byte-based planning evidence separately from legacy Transformer estimates."""
        return {"kind":"captured_model",**asdict(self),"estimated_bytes":self.estimated_bytes}


def owned_tensors(model: nn.Module) -> dict[int, torch.Tensor]:
    """Enumerate unique Parameter/buffer identities without allocating real weights for meta models."""
    return {id(value): value for value in (*model.parameters(), *model.buffers())}


def upper_dimension(value, exported: torch.export.ExportedProgram) -> int:
    """Resolve a captured symbolic extent at admitted upper bounds; unknown shapes fail closed."""
    try:
        bounds = {symbol: interval.upper for symbol, interval in exported.range_constraints.items()}
        maximum = value if type(value) is int else int(value.node.expr.subs(bounds))
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise ArchitectureError("memory_estimate", "operation workspace has an unknown shape bound") from error
    if maximum < 0:
        raise ArchitectureError("memory_estimate", "operation workspace has an invalid shape bound")
    return maximum


def activation_bytes(exported: torch.export.ExportedProgram) -> int:
    """Bound supported forward/backward storage and unfused attention workspace from actual graph shapes.

    Four times the sum of all captured tensor outputs conservatively retains
    forward values, backward intermediates and copies, including SiLU's
    pointwise sigmoid/product intermediates. Attention separately
    reserves fp32 quadratic scores. Unknown operator workspaces are unsupported.
    """
    return check_tensor_estimates(exported) * 4 + sum(operation_workspace(node, exported) for node in exported.graph.nodes)


def operation_workspace(node: torch.fx.Node, exported: torch.export.ExportedProgram) -> int:
    """Bound one supported operation's additional workspace; unknown operations or attention shapes fail closed."""
    if node.op != "call_function" or node.target in STRUCTURAL_OPERATIONS:
        return 0
    if not isinstance(node.target, torch._ops.OpOverload) or node.target._schema.name.split("::")[-1] not in SUPPORTED_OPERATIONS:
        raise ArchitectureError("memory_estimate", f"training memory estimate is unavailable for captured operation {str(node.target)[:120]}")
    if node.target._schema.name != "aten::scaled_dot_product_attention":
        return 0
    query, key = (argument.meta.get("val") for argument in node.args[:2])
    if not isinstance(query, torch.Tensor) or not isinstance(key, torch.Tensor) or query.dim() != 4 or key.dim() != 4:
        raise ArchitectureError("memory_estimate", "attention workspace needs bounded rank-four query/key tensors")
    batch, heads, sequence, _ = [upper_dimension(value, exported) for value in query.shape]
    return batch * heads * sequence * upper_dimension(key.shape[-2], exported) * 4


def require_capture(model: nn.Module, exported: torch.export.ExportedProgram) -> None:
    """Require operation estimates from this exact constructed parameter snapshot, including meta weights."""
    captured = {id(value) for value in exported.state_dict.values() if isinstance(value, nn.Parameter)}
    if captured != {id(value) for value in model.parameters()}:
        raise ArchitectureError("memory_profile", "capture belongs to another model parameter snapshot")


def input_shapes(exported: torch.export.ExportedProgram) -> tuple[tuple[int,...],...]:
    """Record bounded upper shapes of actual user tensor inputs, excluding captured Parameters and buffers."""
    names = set(exported.graph_signature.user_inputs)
    return tuple(tuple(upper_dimension(size,exported) for size in value.shape)
        for node in exported.graph.nodes if node.name in names and isinstance(value := node.meta.get("val"),torch.Tensor))


def estimate_model(
    model: nn.Module, exported: torch.export.ExportedProgram, *, allocator_bytes: int,
    reference_model: nn.Module | None = None, reference_export: torch.export.ExportedProgram | None = None,
    resident_bytes: int = 0,
) -> ModelMemoryPlan:
    """Estimate custom storage/AdamW/work before selecting replay or enforcing the final admission target.

    No width/layer/head assumptions or real allocation probes occur. A frozen
    reference counts shared storage once and retains its separately captured
    execution workspace. The caller supplies the actual admitted ceiling and
    other resident bytes excluding explicitly modeled tensors. AdamW reserves
    gradients plus fp32 moments/master storage and an autocast copy per trainable
    element. No estimate is discounted for recomputation without separate evidence.
    """
    if type(allocator_bytes) is not int or allocator_bytes < 1 or type(resident_bytes) is not int or resident_bytes < 0:
        raise ValueError("memory planning requires positive admission and nonnegative resident bytes")
    require_capture(model,exported)
    if (reference_model is None) != (reference_export is None):
        raise ArchitectureError("memory_profile", "reference model requires its own captured workspace estimate")
    tensors = owned_tensors(model)
    parameter_bytes = sum(value.numel() * value.element_size() for value in tensors.values())
    optimizer_bytes = sum(value.numel() * (value.element_size() + 14) for value in model.parameters() if value.requires_grad)
    reference = owned_tensors(reference_model) if reference_model is not None else {}
    reference_bytes = sum(value.numel() * value.element_size() for identity, value in reference.items() if identity not in tensors)
    if reference_model is not None:
        require_capture(reference_model,reference_export)
        reference_bytes += activation_bytes(reference_export)
    plan = ModelMemoryPlan(parameter_bytes, optimizer_bytes, activation_bytes(exported), reference_bytes,
                           resident_bytes, allocator_bytes, planning_target(allocator_bytes),input_shapes(exported))
    return plan


def plan_model(
    model: nn.Module, exported: torch.export.ExportedProgram, *, allocator_bytes: int,
    reference_model: nn.Module | None = None, reference_export: torch.export.ExportedProgram | None = None,
    resident_bytes: int = 0,
) -> ModelMemoryPlan:
    """Require the ordinary captured model/reference estimate to fit before caller allocations or optimization."""
    plan = estimate_model(model, exported, allocator_bytes=allocator_bytes, reference_model=reference_model,
                          reference_export=reference_export, resident_bytes=resident_bytes)
    if plan.estimated_bytes > plan.target_bytes:
        raise ArchitectureError("memory_budget", "supported custom-model estimate exceeds admitted planning target")
    return plan
