"""Declared observation routing through actual fixed module/loop occurrences, without enabling unvalidated measurements."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import Tensor, nn

from llm.architecture.errors import ArchitectureError
from llm.architecture.execute import CompositionModule, execute_flow, invoke, validate_tensors
from llm.architecture.interfaces import BlockIntrospection, ExecutionOccurrence, InspectionPoint, ModelIntrospection
from llm.architecture.recompute import module_occurrences
from llm.architecture.spec import canonical_digest, MAX_ATTENTION_HEADS

if TYPE_CHECKING:
    from llm.architecture.build import RecipeModule

MAX_INSPECTION_POINTS = 128


@dataclass(frozen=True)
class InspectionBinding:
    """Bind one model-local opaque view ID to its actual owner call and declared local measurement method."""

    point: InspectionPoint
    owner: ExecutionOccurrence
    local: InspectionPoint
    named: bool


def validate_point(point: InspectionPoint) -> None:
    """Reject malformed/oversized interpreted observation metadata before any tensor measurement allocation."""
    if (not isinstance(point, InspectionPoint) or not isinstance(point.occurrence, ExecutionOccurrence)
            or type(point.id) is not str or not 1 <= len(point.id) <= 128
            or type(point.title) is not str or not 1 <= len(point.title) <= 120
            or type(point.port) is not str or not 1 <= len(point.port) <= 64
            or point.kind not in {"attention", "hidden"}
            or type(point.occurrence.instance_path) is not str or len(point.occurrence.instance_path) > 512
            or type(point.occurrence.call_path) is not tuple or len(point.occurrence.call_path) > 64
            or any(type(value) is not int or not 0 <= value < 2**53 for value in point.occurrence.call_path)
            or point.heads is not None and (type(point.heads) is not int or not 1 <= point.heads <= MAX_ATTENTION_HEADS)):
        raise ArchitectureError("inspection_metadata", "inspection point metadata is malformed or exceeds measurement limits")


def inspection_bindings(model: "RecipeModule") -> tuple[InspectionBinding, ...]:
    """Discover declared attention methods and actual call identities; declaration/protocol presence alone does not authorize numerical use.

    Custom metadata methods execute in the owning bounded worker. Its numerical
    owner must preserve/check model state and separately validate each view.
    """
    bindings, identities = [], set()
    for occurrence, module in module_occurrences(model):
        if not {"introspection", "attention"} <= set(module.__slate_capabilities__):
            continue
        named = isinstance(module, BlockIntrospection)
        if not named and not isinstance(module, ModelIntrospection):
            continue
        points = module.inspection_points()
        if type(points) is not tuple or len(points) > MAX_INSPECTION_POINTS:
            raise ArchitectureError("inspection_metadata", "inspection metadata requires a bounded tuple of points")
        for local in points:
            validate_point(local)
            if local.kind != "attention":
                continue
            path = occurrence.instance_path + ("/" + local.occurrence.instance_path if local.occurrence.instance_path else "")
            call = occurrence.call_path + local.occurrence.call_path
            if len(path) > 1024 or len(call) > 64 or len(bindings) == MAX_INSPECTION_POINTS:
                raise ArchitectureError("inspection_metadata", "assembled inspection metadata exceeds occurrence limits")
            identity = canonical_digest({"instance_path": path, "call_path": call, "id": local.id, "port": local.port})
            if identity in identities:
                raise ArchitectureError("inspection_metadata", "inspection point identities must be unique within the model")
            identities.add(identity)
            point = InspectionPoint(identity, ExecutionOccurrence(path, call), local.port, local.kind, local.title, local.heads)
            bindings.append(InspectionBinding(point, occurrence, local, named))
    return tuple(bindings)


def inspect_block(module: nn.Module, inputs: dict[str, Tensor], binding: InspectionBinding) -> Tensor:
    """Invoke only the selected declared observation and reject eager caller/parameter/buffer mutation before normal execution continues."""
    before = [] if torch.compiler.is_compiling() else [(value, value._version)
        for value in (*inputs.values(), *module.parameters(), *module.buffers())]
    result = (module.inspect_named(inputs, point_id=binding.local.id) if binding.named else
        module.inspect(inputs["tokens"], point_id=binding.local.id,
            positions=inputs.get("positions"), key_mask=inputs.get("key_mask")))
    if not isinstance(result, Tensor) or not result.is_floating_point() or result.ndim != 4:
        raise ArchitectureError("inspection_output", "attention inspection must return a floating [batch, heads, queries, keys] tensor")
    if any(value._version != version for value, version in before):
        raise ArchitectureError("inspection_mutation", "inspection mutated caller inputs or persistent model tensors")
    return result


class PreparedInspection(nn.Module):
    """Route one fixed view through the same complete recipe and actual shared-call identities for capture and measurement."""

    def __init__(self, model: "RecipeModule", binding: InspectionBinding):
        """Borrow one admitted model and immutable selected point without retaining request tensors or changing module state."""
        super().__init__()
        self.model, self.binding = model, binding

    def forward(self, tokens: Tensor, positions: Tensor, key_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Return the selected attention and ordinary logits so strict capture/memory planning retain every executed computation."""
        inputs = self.model.language_inputs(tokens, positions=positions, key_mask=key_mask)
        observations = []

        def execute(module, values, occurrence):
            """Observe only the exact selected leaf call and execute every normal call with its original parameters/ports."""
            if isinstance(module, CompositionModule):
                return composition(module, values, occurrence.call_path, occurrence.instance_path + "/")
            if occurrence == self.binding.owner:
                observations.append(inspect_block(module, values, self.binding))
            return invoke(module, values)

        def composition(model, values, call_path, instance_path):
            """Keep nested subarchitecture routing explicit without conflating independently repeated occurrences."""
            symbols = validate_tensors(values, model.plan.inputs)
            environment = dict(values)
            execute_flow(model.plan.operations, environment, model.instances, executor=execute,
                call_path=call_path, instance_path=instance_path)
            result = {port: environment[value] for port, value in model.plan.result.items()}
            validate_tensors(result, model.plan.outputs, symbols)
            return result

        outputs = composition(self.model, inputs, (), "")
        if len(observations) != 1:
            raise ArchitectureError("inspection_occurrence", "selected inspection occurrence was not executed exactly once")
        return observations[0], outputs["logits"]
