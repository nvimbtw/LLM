"""Parameter-identity optimizer ownership for arbitrary constructed modules and shared recipes."""

from collections.abc import Mapping
from dataclasses import dataclass
import math

from torch import nn

from llm.architecture.build import RecipeModule, parameter_groups
from llm.architecture.errors import ArchitectureError
from llm.model.transformer import Attention


@dataclass(frozen=True)
class OptimizerPolicy:
    """An explicit per-parameter decay policy, independent of display names and execution occurrences."""

    weight_decay: float

    def __post_init__(self) -> None:
        """Reject nonfinite/negative decay settings before constructing optimizer groups."""
        if isinstance(self.weight_decay, bool) or not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight decay must be finite and nonnegative")


@dataclass(frozen=True)
class ParameterPolicy:
    """A model capability binds an actual owned Parameter to its required optimizer policy."""

    parameter: nn.Parameter
    policy: OptimizerPolicy


def parameter_paths(model: nn.Module) -> dict[str, nn.Parameter]:
    """List stable semantic recipe paths or ordinary PyTorch paths, retaining every alias."""
    if isinstance(model, RecipeModule):
        return model.parameter_paths()
    return dict(model.named_parameters(remove_duplicate=False))


def builtin_policies(model: nn.Module) -> tuple[ParameterPolicy, ...]:
    """Expose legacy attention-temperature exclusions through their owning component capability.

    Arbitrary custom names ending in log_temperature do not imply this policy.
    A caller may supply additional explicit capabilities for other components.
    """
    return tuple(ParameterPolicy(module.log_temperature, OptimizerPolicy(0.0))
                 for module in model.modules() if isinstance(module, Attention)
                 and module.log_temperature is not None)


def optimizer_groups(
    model: nn.Module, *, weight_decay: float,
    capabilities: tuple[ParameterPolicy, ...] = (),
    overrides: Mapping[str, OptimizerPolicy] | None = None,
) -> tuple[dict, ...]:
    """Assign every trainable Parameter once, rejecting conflicting shared aliases/capability policies.

    Groups and parameter order follow first semantic ownership, so recovery can
    verify the same layout. Frozen Parameters never enter the optimizer.
    """
    default = OptimizerPolicy(weight_decay)
    paths = parameter_paths(model)
    overrides = overrides or {}
    if overrides.keys() - paths.keys():
        raise ArchitectureError("optimizer_policy", "optimizer policy names an unknown Parameter")
    owners = {}
    for path, parameter in paths.items():
        owners.setdefault(id(parameter),path)
    assignments = dict(overrides)
    for requirement in (*builtin_policies(model), *capabilities):
        identity = id(requirement.parameter)
        if identity not in owners:
            raise ArchitectureError("optimizer_policy", "optimizer capability names a foreign Parameter")
        path = owners[identity]
        if path in assignments and assignments[path] != requirement.policy:
            raise ArchitectureError("optimizer_policy", "foreign Parameter or conflicting shared optimizer policies")
        assignments[path] = requirement.policy
    policies = {path:{"weight_decay":policy.weight_decay} for path,policy in assignments.items()}
    owned_groups = parameter_groups(model,policies)
    groups = {}
    for group in owned_groups:
        policy = OptimizerPolicy(group.get("weight_decay",default.weight_decay))
        groups.setdefault(policy, []).extend(group["params"])
    return tuple({"params": parameters, "weight_decay": policy.weight_decay} for policy, parameters in groups.items())
