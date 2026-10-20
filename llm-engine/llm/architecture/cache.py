"""Explicit occurrence-owned block cache routing, without enabling unvalidated LM caching."""

from collections.abc import Mapping

import torch
from torch import Tensor, nn

from llm.architecture.errors import ArchitectureError
from llm.architecture.execute import CompositionModule, compatible_carry
from llm.architecture.interfaces import BlockCachedResult, ExecutionOccurrence
from llm.architecture.plan import PlanCall, PlanRepeat, PlanStack

CacheEntries = Mapping[ExecutionOccurrence, Mapping[str, Tensor]]


def cached_call(
    module: nn.Module, inputs: Mapping[str, Tensor], state: Mapping[str, Tensor],
) -> BlockCachedResult:
    """Invoke an explicitly supported cached block while preserving caller state tensors.

    Cache-specific input shape contracts differ from full-prefix ports and are
    validated by the capability/profile adapter. This function enforces named
    tensors and mutation ownership, never inferring cache support from weights.
    """
    if not module.__slate_cache_supported__:
        raise ArchitectureError("cache_unsupported", "every cached block needs an explicit validated capability")
    before, persistent = [], []
    if not torch.compiler.is_compiling():
        before = [(value, value._version) for value in (*inputs.values(), *state.values())]
        persistent = [(value, value._version) for value in (*module.parameters(), *module.buffers())]
    result = module.cached_forward(inputs, state)
    if not isinstance(result, BlockCachedResult) or set(result.outputs) != set(module.__slate_resolved_outputs__):
        raise ArchitectureError("cache_result", "cached block returned an incompatible named output")
    if any(not isinstance(name, str) or not isinstance(value, Tensor)
           for name, value in (*result.outputs.items(), *result.state.items())):
        raise ArchitectureError("cache_result", "cache output and state must be named tensors")
    if any(value._version != version for value, version in before):
        raise ArchitectureError("input_mutation", "cached block mutated caller-owned inputs/state")
    if any(value._version != version for value, version in persistent):
        raise ArchitectureError("persistent_mutation", "cached block mutated persistent model tensors")
    return result


def execute_cached_flow(
    operations: tuple, environment: dict[str, Tensor], modules: nn.ModuleDict,
    previous: CacheEntries, updated: dict, call_path: tuple[int, ...] = (),
    instance_path: str = "",
) -> None:
    """Route fixed calls/loops into fresh occurrence-scoped state without sharing caches."""
    for index, operation in enumerate(operations):
        occurrence = call_path + (index,)
        if isinstance(operation, PlanCall):
            inputs = {port: environment[name] for port, name in operation.inputs.items()}
            module = modules[operation.instance]
            path = instance_path + operation.instance
            if isinstance(module, CompositionModule):
                outputs = module._execute_cached(inputs, previous, updated, occurrence, path + "/")
            else:
                key = ExecutionOccurrence(path, occurrence)
                result = cached_call(module, inputs, previous.get(key, {}))
                updated[key] = dict(result.state)
                outputs = result.outputs
            environment.update({name: outputs[port] for port, name in operation.outputs.items()})
        elif isinstance(operation, PlanRepeat):
            carried = {carry.input: environment[carry.initial] for carry in operation.carry}
            for iteration in range(operation.count):
                inner = {**environment, **carried}
                execute_cached_flow(operation.body, inner, modules, previous, updated,
                                    occurrence + (iteration,), instance_path)
                values = {carry.input: inner[carry.output] for carry in operation.carry}
                for name, value in values.items():
                    compatible_carry(carried[name], value)
                carried = values
            environment.update({carry.result: carried[carry.input] for carry in operation.carry})
        else:
            execute_cached_stack(operation, environment, modules, previous, updated, occurrence, instance_path)


def execute_cached_stack(
    operation: PlanStack, environment: dict[str, Tensor], modules: nn.ModuleDict,
    previous: CacheEntries, updated: dict, call_path: tuple[int, ...], instance_path: str,
) -> None:
    """Keep independent stack instances and repeated execution occurrences independently cached."""
    carried = {carry.input: environment[carry.initial] for carry in operation.carry}
    static = {port: environment[name] for port, name in operation.inputs.items()}
    for index, path in enumerate(operation.instances):
        module = modules[path]
        occurrence = call_path + (index,)
        if isinstance(module, CompositionModule):
            outputs = module._execute_cached({**static, **carried}, previous, updated,
                                             occurrence, instance_path + path + "/")
        else:
            key = ExecutionOccurrence(instance_path + path, occurrence)
            result = cached_call(module, {**static, **carried}, previous.get(key, {}))
            updated[key] = dict(result.state)
            outputs = result.outputs
        values = {carry.input: outputs[carry.output] for carry in operation.carry}
        for name, value in values.items():
            compatible_carry(carried[name], value)
        carried = values
    environment.update({carry.result: carried[carry.input] for carry in operation.carry})
