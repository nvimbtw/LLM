"""Bounded parameter/occurrence diagnostics without fixed Transformer layout assumptions."""

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn

from llm.architecture.build import RecipeModule
from llm.architecture.errors import ArchitectureError
from llm.architecture.interfaces import ExecutionOccurrence
from llm.architecture.numerical import output_tensors, preserved_model
from llm.architecture.recompute import module_occurrences
from llm.train.model_memory import owned_tensors
from llm.train.model_policy import parameter_paths


@dataclass(frozen=True)
class DiagnosticBudget:
    """Separate diagnostic allocation/work limits, applied before persistent snapshots or tensor reductions."""

    bytes: int = 64 << 20
    occurrences: int = 128

    def __post_init__(self) -> None:
        """Require effective finite integer byte/work bounds."""
        if type(self.bytes) is not int or self.bytes < 1 or type(self.occurrences) is not int or not 1 <= self.occurrences <= 128:
            raise ValueError("invalid diagnostic resource budget")


class ModelDiagnostics:
    """Own optional CPU snapshots and temporary hooks for one admitted model's diagnostic lifetime."""

    def __init__(self, model: nn.Module, budget: DiagnosticBudget = DiagnosticBudget()):
        """Retain the existing model reference and explicit limits without allocating snapshots."""
        self.model = model
        self.budget = budget
        self.disabled_reason: str | None = None
        self.snapshot: dict[str, Tensor] = {}

    def owners(self) -> dict[str, nn.Parameter]:
        """Return one stable semantic owner per actual Parameter, retaining no alias duplicates."""
        owners = {}
        seen = set()
        for path, parameter in parameter_paths(self.model).items():
            if id(parameter) not in seen:
                owners[path] = parameter
                seen.add(id(parameter))
        return owners

    def admit(self, requested_bytes: int) -> None:
        """Refuse diagnostic allocations before they exceed the caller's separate sampling budget."""
        if requested_bytes > self.budget.bytes:
            raise ArchitectureError("diagnostic_budget", "diagnostic sample exceeds its byte budget")

    def disable(self, error: Exception) -> None:
        """Release optional snapshots after diagnostic OOM/budget exhaustion, without swallowing training errors."""
        self.snapshot.clear()
        self.disabled_reason = f"{type(error).__name__}: {error}"

    def before_step(self) -> dict[str, float] | None:
        """Snapshot unique weights on CPU and measure available gradients before the optimizer update."""
        if self.disabled_reason is not None:
            return None
        try:
            owners = self.owners()
            self.admit(sum(parameter.numel() * (parameter.element_size() + 16) for parameter in owners.values()))
            self.snapshot = {path: parameter.detach().to("cpu", copy=True) for path, parameter in owners.items()}
            return {path: float(parameter.grad.detach().float().norm()) for path, parameter in owners.items()
                    if parameter.grad is not None}
        except (torch.cuda.OutOfMemoryError, ArchitectureError) as error:
            if isinstance(error, ArchitectureError) and error.code != "diagnostic_budget":
                raise
            self.disable(error)
            return None

    def after_step(self) -> dict[str, float] | None:
        """Measure update ratios against CPU snapshots, releasing them on every outcome."""
        if not self.snapshot or self.disabled_reason is not None:
            return None
        try:
            ratios = {}
            for path, parameter in self.owners().items():
                value = parameter.detach().to("cpu", copy=True).float()
                before = self.snapshot[path].float()
                norm = float(value.norm())
                if norm > 0:
                    ratios[path] = float((value - before).norm()) / norm
            return ratios
        except torch.cuda.OutOfMemoryError as error:
            self.disable(error)
            return None
        finally:
            self.snapshot.clear()

    def sample(self, tokens: Tensor, *, positions: Tensor | None = None, key_mask: Tensor | None = None) -> tuple[dict, ...] | None:
        """Measure actual call occurrences in deterministic eval while preserving mode/RNG/weights/gradients.

        No attention/embedding-specific values are invented. Hook and snapshot
        lifetimes end on success, budget exhaustion, OOM or a propagated model error.
        """
        if self.disabled_reason is not None:
            return None
        handles = []
        try:
            calls = (module_occurrences(self.model) if isinstance(self.model, RecipeModule) else
                     ((ExecutionOccurrence("model", ()), self.model),))
            if len(calls) > self.budget.occurrences:
                raise ArchitectureError("diagnostic_budget", "diagnostic occurrence work exceeds budget")
            retained = sum(value.numel() * value.element_size() for value in owned_tensors(self.model).values())
            retained += sum(value.numel() * value.element_size() for value in self.snapshot.values())
            retained += sum(value.numel() * value.element_size() for value in (tokens, positions, key_mask) if value is not None)
            self.admit(retained)
            rows = []
            pending = list(calls)
            hook = self.capture_hook(pending, rows, retained)
            modules = {id(module): module for _, module in calls}
            for module in modules.values():
                handles.append(module.register_forward_hook(hook))
            with preserved_model(self.model), torch.no_grad():
                self.model.eval()
                kwargs = {name:value for name,value in (("positions", positions), ("key_mask", key_mask)) if value is not None}
                self.model(tokens, **kwargs)
            if pending:
                raise ArchitectureError("diagnostic_occurrence", "model execution omitted declared diagnostic occurrences")
            return tuple(rows)
        except (torch.cuda.OutOfMemoryError, ArchitectureError) as error:
            if isinstance(error, ArchitectureError) and error.code != "diagnostic_budget":
                raise
            self.disable(error)
            return None
        finally:
            for handle in handles:
                handle.remove()

    def capture_hook(self, pending: list, rows: list, retained_bytes: int):
        """Build one request-local reducer that records scalar/shape data and releases all output tensors."""
        used = retained_bytes

        def capture(module: nn.Module, _args: tuple, output: object) -> None:
            """Reduce one actual module call under its expected execution identity and remaining work budget."""
            nonlocal used
            if not pending or pending[0][1] is not module:
                raise ArchitectureError("diagnostic_occurrence", "model execution differs from declared diagnostic occurrences")
            occurrence, _ = pending.pop(0)
            values = (output,) if isinstance(output, Tensor) else tuple(output.values()) if isinstance(output, dict) else ()
            if not values or any(not isinstance(value, Tensor) for value in values):
                raise ArchitectureError("diagnostic_output", "diagnostics require named or single tensor outputs")
            used += sum(value.numel() * 8 for value in values)
            self.admit(used)
            output_tensors(output)
            rms = [float(value.detach().float().square().mean().sqrt()) for value in values if value.is_floating_point()]
            if any(not math.isfinite(value) for value in rms):
                raise ArchitectureError("nonfinite_output", "diagnostic output is nonfinite")
            rows.append({"instance_path":occurrence.instance_path, "call_path":occurrence.call_path,
                         "shapes":tuple(tuple(value.shape) for value in values),
                         "dtypes":tuple(str(value.dtype) for value in values), "output_rms":tuple(rms)})

        return capture
