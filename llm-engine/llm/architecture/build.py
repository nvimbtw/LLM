"""Construct resolved model instances and exact Parameter aliases inside an admitted worker."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from llm.architecture.errors import ArchitectureError
from llm.architecture.cache import execute_cached_flow
from llm.architecture.execute import CompositionModule, execute_flow, validate_tensors
from llm.architecture.interfaces import CacheIdentity, CachedBlock, LanguageModelFacts, RequestCache
from llm.architecture.plan import BlockDefinition, ResolvedPlan
from llm.architecture.resolve import bound_port
from llm.architecture.spec import Scalar


def tensor_assert(condition: Tensor, message: str) -> None:
    """Preserve tensor-dependent boundary checks as runtime assertions under strict capture."""
    if torch.compiler.is_compiling():
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ArchitectureError("tensor_bounds", message)


class RecipeModule(CompositionModule):
    """One resolved composition owning child modules, tensor execution and shared Parameters."""

    def __init__(self, plan: ResolvedPlan, modules: Mapping[str, nn.Module]):
        """Own one already constructed module per resolved path without copying Parameters."""
        super().__init__()
        self.plan = plan
        self.instances = nn.ModuleDict(modules)

    @property
    def facts(self) -> LanguageModelFacts:
        """Return final causal model facts, refusing tensor-only composition as a language model."""
        if self.plan.interface != "causal_lm":
            raise ArchitectureError("model_interface", "tensor subarchitecture is not a causal language model")
        return LanguageModelFacts(self.plan.vocabulary_size, self.plan.context_window)

    def forward_named(self, inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Execute validated named inputs and return declared named tensors without persistent state."""
        symbols = validate_tensors(inputs, self.plan.inputs)
        environment = dict(inputs)
        execute_flow(self.plan.operations, environment, self.instances)
        result = {port: environment[value] for port, value in self.plan.result.items()}
        validate_tensors(result, self.plan.outputs, symbols)
        return result

    def forward(
        self, tokens: Tensor, *, positions: Tensor | None = None, key_mask: Tensor | None = None,
    ) -> Tensor:
        """Return causal LM logits using explicit/default logical positions and padding visibility.

        Bounds, caller tensor ranks/dtypes and final vocabulary shape are checked;
        numerical causality/capture acceptance remains a separate validation gate.
        """
        facts = self.facts
        if tokens.dtype != torch.int64 or tokens.dim() != 2:
            raise ArchitectureError("causal_input_dtype", "tokens must be rank-two int64")
        batch, sequence = tokens.shape
        if batch < 1 or sequence < 1 or sequence > facts.context_window:
            raise ArchitectureError("context_binding", "token shape exceeds model context or is empty")
        tensor_assert(((tokens >= 0) & (tokens < facts.vocabulary_size)).all(), "token ID is outside vocabulary")
        if positions is None:
            positions = torch.arange(sequence, device=tokens.device).expand(batch, -1)
        if positions.dtype != torch.int64 or positions.shape != tokens.shape or positions.device != tokens.device:
            raise ArchitectureError("position_shape", "positions must be matching int64 tensors on the token device")
        tensor_assert(((positions >= 0) & (positions < facts.context_window)).all(), "position exceeds model context")
        if key_mask is None:
            key_mask = torch.ones_like(tokens, dtype=torch.bool)
        if key_mask.dtype != torch.bool or key_mask.shape != tokens.shape or key_mask.device != tokens.device:
            raise ArchitectureError("mask_shape", "key mask must be matching bool tensors on the token device")
        external = {"tokens": tokens, "positions": positions, "key_mask": key_mask}
        result = self.forward_named({name: external[name] for name in self.plan.inputs})["logits"]
        if result.shape != (batch, sequence, facts.vocabulary_size) or not result.is_floating_point():
            raise ArchitectureError("causal_output", "logits do not match the tokenizer-bound causal interface")
        return result

    def parameter_paths(self) -> dict[str, nn.Parameter]:
        """Enumerate stable public paths including aliases without multiplying unique owners."""
        result = {}
        for path, module in self.instances.items():
            if isinstance(module, RecipeModule):
                result.update({f"{path}/{name}": value for name, value in module.parameter_paths().items()})
            else:
                result.update({f"{path}.{name}": value for name, value in module.named_parameters(remove_duplicate=False)})
        return result

    def _execute_cached(self, inputs: Mapping[str, Tensor], previous: Mapping, updated: dict,
                        call_path: tuple[int, ...], instance_path: str) -> dict[str, Tensor]:
        """Route an already admitted cached signature through child compositions and occurrences."""
        if set(inputs) != set(self.plan.inputs):
            raise ArchitectureError("port_names", "cached input keys differ from the recipe")
        environment = dict(inputs)
        execute_cached_flow(self.plan.operations, environment, self.instances, previous, updated,
                            call_path, instance_path)
        return {port: environment[name] for port, name in self.plan.result.items()}

    def forward_cached_named(
        self, inputs: Mapping[str, Tensor], cache: RequestCache, *, identity: CacheIdentity,
    ) -> tuple[dict[str, Tensor], RequestCache]:
        """Execute explicit cached block tensors after a capability/profile adapter validates them.

        The owning worker supplies the current request/definition identity. This
        internal method does not itself authorize or expose whole-LM cached serving.
        """
        if cache.identity != identity:
            raise ArchitectureError("cache_identity", "cache belongs to another request or definition")
        updated = {}
        result = self._execute_cached(inputs, cache.entries, updated, (), "")
        if cache.entries.keys() - updated.keys():
            raise ArchitectureError("cache_occurrence", "cache contains state from another execution structure")
        return result, RequestCache(identity, updated)

    def parameter_location(self, path: str) -> tuple[nn.Module, str]:
        """Resolve an exact registered Parameter path without eval or trainable tensor views."""
        for name in sorted(self.instances.keys(), key=len, reverse=True):
            module = self.instances[name]
            if isinstance(module, RecipeModule) and path.startswith(name + "/"):
                return module.parameter_location(path[len(name) + 1:])
            if path.startswith(name + "."):
                local = path[len(name) + 1:]
                parent, _, parameter = local.rpartition(".")
                try:
                    owner = module.get_submodule(parent) if parent else module
                except (AttributeError, KeyError) as error:
                    raise ArchitectureError("parameter_alias", "unknown Parameter alias path") from error
                if parameter not in owner._parameters or owner._parameters[parameter] is None:
                    raise ArchitectureError("parameter_alias", "alias must address an exact registered Parameter")
                return owner, parameter
        raise ArchitectureError("parameter_alias", "unknown Parameter alias instance path")


def apply_aliases(model: RecipeModule) -> None:
    """Assign exact compatible Parameter objects, rejecting cycles and conflicting target aliases."""
    aliases = {alias.target: alias.owner for alias in model.plan.aliases}
    if len(aliases) != len(model.plan.aliases):
        raise ArchitectureError("parameter_alias", "Parameter alias target is declared more than once")
    for target, initial_owner in aliases.items():
        owner, seen = initial_owner, {target}
        while owner in aliases:
            if owner in seen:
                raise ArchitectureError("parameter_alias", "cyclic Parameter aliases")
            seen.add(owner)
            owner = aliases[owner]
        owner_module, owner_name = model.parameter_location(owner)
        target_module, target_name = model.parameter_location(target)
        parameter = owner_module._parameters[owner_name]
        replaced = target_module._parameters[target_name]
        if parameter.shape != replaced.shape or parameter.dtype != replaced.dtype or parameter.device != replaced.device:
            raise ArchitectureError("incompatible_parameter_alias", "aliased Parameter shape/dtype/device differ")
        target_module._parameters[target_name] = parameter


def reject_overlapping_parameters(model: nn.Module) -> None:
    """Reject separately optimized Parameter objects whose real storage byte ranges overlap."""
    intervals = {}
    for parameter in model.parameters():
        if not parameter.numel() or parameter.device.type == "meta":
            continue
        storage = (parameter.device, parameter.untyped_storage().data_ptr())
        start = parameter.storage_offset() * parameter.element_size()
        span = sum((size - 1) * stride for size, stride in zip(parameter.shape, parameter.stride())) + 1
        end = start + span * parameter.element_size()
        if any(start < other_end and other_start < end for other_start, other_end in intervals.get(storage, ())):
            raise ArchitectureError("overlapping_parameters", "distinct Parameters overlap the same storage")
        intervals.setdefault(storage, []).append((start, end))


def parameter_groups(
    model: RecipeModule, policies: Mapping[str, Mapping[str, Scalar]] | None = None,
) -> tuple[dict, ...]:
    """Deduplicate stable Parameter owners and reject conflicting alias-specific optimizer policies."""
    paths = model.parameter_paths()
    policies = policies or {}
    if policies.keys() - paths.keys():
        raise ArchitectureError("optimizer_policy", "optimizer policy refers to an unknown Parameter")
    owners, settings = {}, {}
    for path, parameter in paths.items():
        key = id(parameter)
        owners.setdefault(key, parameter)
        if path in policies:
            policy = dict(policies[path])
            if "params" in policy or key in settings and settings[key] != policy:
                raise ArchitectureError("optimizer_policy", "one shared Parameter has conflicting optimizer policies")
            settings[key] = policy
    return tuple({"params": [parameter], **settings.get(key, {})} for key, parameter in owners.items()
                 if parameter.requires_grad)


def place_module(module: nn.Module, device: str | torch.device, dtype: torch.dtype | None) -> None:
    """Place every parameter/buffer while restoring exact aliases across PyTorch conversion.

    Explicit constructor device choices cannot bypass worker placement. Aliased
    Parameter/buffer objects remain one owner even if _apply allocates replacements.
    """
    aliases = []
    for kind, values in (("_parameters", module.named_parameters(remove_duplicate=False)),
                         ("_buffers", module.named_buffers(remove_duplicate=False))):
        owners = {}
        for path, value in values:
            if id(value) in owners:
                aliases.append((kind, owners[id(value)], path))
            else:
                owners[id(value)] = path
    module.to(device=device, dtype=dtype)
    for kind, owner, target in aliases:
        owner_path, _, owner_name = owner.rpartition(".")
        target_path, _, target_name = target.rpartition(".")
        source = module.get_submodule(owner_path) if owner_path else module
        destination = module.get_submodule(target_path) if target_path else module
        getattr(destination, kind)[target_name] = getattr(source, kind)[owner_name]


def build_model(
    plan: ResolvedPlan, *, device: str | torch.device = "cpu", dtype: torch.dtype | None = None,
    optimizer_policies: Mapping[str, Mapping[str, Scalar]] | None = None,
) -> RecipeModule:
    """Construct one instance per stable path under worker admission, then establish aliases.

    Placement precedes alias assignment so a device/dtype transform cannot split
    shared Parameter identities. No optimizer, source imports or storage effects occur.
    """
    modules = {}
    with torch.device(device):
        for instance in plan.instances:
            if isinstance(instance.definition, BlockDefinition):
                module = instance.definition.module_type(**instance.parameters)
                if not isinstance(module, nn.Module):
                    raise ArchitectureError("invalid_block", "block constructor did not return a module")
                place_module(module, device, dtype)
                modules[instance.path] = module
            else:
                modules[instance.path] = build_model(instance.definition, device=device, dtype=dtype)
            module = modules[instance.path]
            declaration = instance.definition.declaration if isinstance(instance.definition, BlockDefinition) else instance.definition
            dimensions = {**instance.parameters, **({"vocabulary": plan.vocabulary_size} if plan.vocabulary_size else {})}
            module.__slate_resolved_inputs__ = {name: bound_port(port, dimensions) for name, port in declaration.inputs.items()}
            module.__slate_resolved_outputs__ = {name: bound_port(port, dimensions) for name, port in declaration.outputs.items()}
            module.__slate_capabilities__ = tuple(declaration.capabilities) if isinstance(instance.definition, BlockDefinition) else ()
            module.__slate_cache_supported__ = "cache" in module.__slate_capabilities__ and isinstance(module, CachedBlock)
    model = RecipeModule(plan, modules)
    apply_aliases(model)
    reject_overlapping_parameters(model)
    parameter_groups(model, optimizer_policies)
    return model
