"""Captured leaf-call storage lifetimes, including all returned port backing storage."""

from dataclasses import dataclass
from collections.abc import Mapping

import torch

from llm.architecture.build import RecipeModule
from llm.architecture.capture import MAX_CAPTURE_TENSOR_BYTES, check_tensor_estimates, tensor_output_bytes
from llm.architecture.errors import ArchitectureError
from llm.architecture.interfaces import ExecutionOccurrence
from llm.architecture.recompute import module_occurrences
from llm.train.model_memory import operation_workspace, require_capture


@dataclass(frozen=True)
class ReplayRegion:
    """One real leaf-call occurrence's bounded outputs/workspace and returned backing storage, in bytes."""

    occurrence: ExecutionOccurrence
    tensor_bytes: int
    workspace_bytes: int
    output_bytes: int

    @property
    def training_bytes(self) -> int:
        """Keep the ordinary four-copy tensor bound and separate supported operator workspace."""
        return 4 * self.tensor_bytes + self.workspace_bytes


@dataclass(frozen=True)
class ReplayStorage:
    """An exact strict graph partition plus observed port storage, without tensors or source execution handles."""

    outside_bytes: int
    regions: tuple[ReplayRegion, ...]

    @property
    def captured_training_bytes(self) -> int:
        """Return the unmodified graph bound so stale or missing partitions cannot authorize a discount."""
        return self.outside_bytes + sum(region.training_bytes for region in self.regions)

    def activation_bytes(self, selected: tuple[ExecutionOccurrence, ...]) -> int:
        """Retain unselected work, every selected output's full backing store and one largest replay region.

        Selected calls are nonnested leaf occurrences on one admitted device.
        All leaf outputs remain charged even when unused by the exported result;
        views charge the complete backing store. Rebuild uses the same four-copy
        and workspace bounds as ordinary capture, without layer-based discounts.
        Port storage is charged separately even for unselected calls, so export
        dead-code elimination cannot hide a retained unused eager result.
        """
        chosen = set(selected)
        if len(chosen) != len(selected) or chosen - {region.occurrence for region in self.regions}:
            raise ArchitectureError("recompute_occurrence", "storage estimate has unknown or duplicate replay occurrences")
        retained = self.outside_bytes + sum(4 * region.output_bytes +
            (0 if region.occurrence in chosen else region.training_bytes) for region in self.regions)
        transient = max((region.training_bytes for region in self.regions if region.occurrence in chosen), default=0)
        return retained + transient


def output_storage(model: RecipeModule, inputs: Mapping[str, torch.Tensor]) -> dict[ExecutionOccurrence, int]:
    """Observe all actual leaf outputs in a grad-enabled forward; caller preserves probe RNG/state.

    Hooks are always removed. The storage byte count includes full aliased bases
    and unused ports rather than inferring retained tensors from graph users.
    Meta models execute named ports directly, avoiding data-dependent root asserts.
    Grad mode matches training so gradient-dependent views cannot hide a larger
    backing allocation behind a cheaper no-grad output. No graph escapes the probe.
    """
    calls = module_occurrences(model)
    pending = iter(calls)
    outputs = {}

    def record(module, _inputs, result):
        """Match each actual hook to the resolved execution order and retain only bounded byte counts."""
        expected = next(pending, None)
        if expected is None or expected[1] is not module or not isinstance(result, dict):
            raise ArchitectureError("recompute_capture", "observed leaf calls differ from resolved execution")
        if any(not isinstance(value, torch.Tensor) for value in result.values()):
            raise ArchitectureError("recompute_capture", "observed port is not a tensor")
        outputs[expected[0]] = sum(value.untyped_storage().nbytes() for value in result.values())
        if sum(outputs.values()) > MAX_CAPTURE_TENSOR_BYTES:
            raise ArchitectureError("recompute_capture", "retained port storage exceeds capture budget")

    handles = []
    try:
        for module in {id(module): module for _, module in calls}.values():
            handles.append(module.register_forward_hook(record))
        with torch.enable_grad():
            model.forward_named(inputs)
        if next(pending, None) is not None:
            raise ArchitectureError("recompute_capture", "captured execution omitted a resolved leaf call")
        return outputs
    finally:
        for handle in handles:
            handle.remove()


def capture_replay_storage(model: RecipeModule, exported: torch.export.ExportedProgram, args: tuple) -> ReplayStorage:
    """Partition this exact ordinary strict graph by engine annotations and observe complete port backing storage.

    No higher-order checkpoint graph or opaque operation is admitted. Unknown
    annotations/workspaces fail. The caller preserves model state/RNG around
    the finite grad-enabled observation; meta construction needs no real weights.
    """
    require_capture(model, exported)
    check_tensor_estimates(exported)
    external = dict(zip(("tokens", "positions", "key_mask"), args, strict=True))
    outputs = output_storage(model, {name: external[name] for name in model.plan.inputs})
    totals = {occurrence: [0, 0] for occurrence in outputs}
    outside = 0
    for node in exported.graph.nodes:
        size = tensor_output_bytes(node.meta.get("val"), exported)
        workspace = operation_workspace(node, exported)
        tag = node.meta.get("custom", {}).get("slate_occurrence") if node.op == "call_function" else None
        if tag is None:
            outside += 4 * size + workspace
            continue
        if (not isinstance(tag, tuple) or len(tag) != 2 or not isinstance(tag[0], str)
                or not isinstance(tag[1], tuple) or any(type(value) is not int for value in tag[1])):
            raise ArchitectureError("recompute_capture", "invalid captured occurrence annotation")
        occurrence = ExecutionOccurrence(tag[0], tag[1])
        if occurrence not in totals:
            raise ArchitectureError("recompute_capture", "capture contains an unknown leaf occurrence")
        totals[occurrence][0] += size
        totals[occurrence][1] += workspace
    regions = tuple(ReplayRegion(occurrence, *totals[occurrence], outputs[occurrence]) for occurrence in outputs)
    return ReplayStorage(outside, regions)
