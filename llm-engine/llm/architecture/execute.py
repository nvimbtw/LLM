"""Named tensor routing and bounded fixed-loop execution without storage or transport."""

from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn

from llm.architecture.errors import ArchitectureError
from llm.architecture.interfaces import ExecutionOccurrence
from llm.architecture.plan import PlanCall, PlanRepeat, PlanStack
from llm.architecture.spec import TensorPort

CallExecutor = Callable[[nn.Module, Mapping[str, Tensor], ExecutionOccurrence], dict[str, Tensor]]


class CompositionModule(nn.Module):
    """Engine-owned composition marker, separating adapters from ordinary declared blocks."""


def validate_tensors(
    values: Mapping[str, Tensor], ports: Mapping[str, TensorPort], symbols: dict | None = None,
) -> dict:
    """Check exact named tensor ranks/dtypes/dimensions without converting tensor values to Python."""
    if set(values) != set(ports):
        raise ArchitectureError("port_names", "computed tensor keys do not match declared ports")
    symbols = {} if symbols is None else symbols
    for name, port in ports.items():
        value = values[name]
        if not isinstance(value, Tensor) or value.dim() != len(port.shape):
            raise ArchitectureError("port_shape", f"port {name} requires a tensor of declared rank")
        if port.dtype == "floating":
            valid_dtype = value.is_floating_point()
        else:
            valid_dtype = value.dtype == getattr(torch, port.dtype)
        if not valid_dtype:
            raise ArchitectureError("port_dtype", f"port {name} has an incompatible dtype")
        for size, dimension in zip(value.shape, port.shape):
            expected = dimension if type(dimension) is int else symbols.setdefault(dimension, size)
            if size != expected:
                raise ArchitectureError("port_shape", f"port {name} has incompatible dimensions")
    return symbols


def invoke(module: nn.Module, inputs: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Invoke a named module, detecting eager caller/persistent tensor mutation.

    Capture retains tensor arithmetic directly; the separate complete-graph stage
    must inspect write/alias schemas before functionalization, as established by48.
    """
    symbols = validate_tensors(inputs, module.__slate_resolved_inputs__)
    if torch.compiler.is_compiling():
        result = module.forward_named(inputs) if isinstance(module, CompositionModule) else module(**inputs)
        validate_tensors(result, module.__slate_resolved_outputs__, symbols)
        return result
    inputs_before = [(value, value._version) for value in inputs.values()]
    persistent = [(value, value._version) for value in (*module.parameters(), *module.buffers())]
    result = module.forward_named(inputs) if isinstance(module, CompositionModule) else module(**inputs)
    if any(value._version != version for value, version in inputs_before):
        raise ArchitectureError("input_mutation", "block mutated a caller-owned input tensor")
    if any(value._version != version for value, version in persistent):
        raise ArchitectureError("persistent_mutation", "block mutated a parameter or registered buffer")
    if not isinstance(result, dict) or any(not isinstance(value, Tensor) for value in result.values()):
        raise ArchitectureError("port_result", "block must return a named dict of tensor outputs")
    validate_tensors(result, module.__slate_resolved_outputs__, symbols)
    return result


def compatible_carry(before: Tensor, after: Tensor) -> None:
    """Keep carried state rank, dtype, device and dimensions fixed through every iteration."""
    if before.shape != after.shape or before.dtype != after.dtype or before.device != after.device:
        raise ArchitectureError("carry_shape", "loop carry changed shape, dtype or device")


def execute_flow(operations: tuple, environment: dict[str, Tensor], modules: nn.ModuleDict, *,
                 executor: CallExecutor | None = None, call_path: tuple[int, ...] = (), instance_path: str = "") -> None:
    """Evaluate already resolved operations, retaining autograd through all repeated calls."""
    for index, operation in enumerate(operations):
        occurrence = call_path + (index,)
        if isinstance(operation, PlanCall):
            inputs = {port: environment[name] for port, name in operation.inputs.items()}
            module = modules[operation.instance]
            values = (invoke(module, inputs) if executor is None else
                      executor(module, inputs, ExecutionOccurrence(instance_path + operation.instance, occurrence)))
            if set(values) != set(operation.outputs):
                raise ArchitectureError("port_names", "block output keys differ from the resolved call")
            environment.update({name: values[port] for port, name in operation.outputs.items()})
        elif isinstance(operation, PlanRepeat):
            carried = {carry.input: environment[carry.initial] for carry in operation.carry}
            for iteration in range(operation.count):
                inner = {**environment, **carried}
                execute_flow(operation.body, inner, modules, executor=executor,
                             call_path=occurrence + (iteration,), instance_path=instance_path)
                updated = {carry.input: inner[carry.output] for carry in operation.carry}
                for name, tensor in updated.items():
                    compatible_carry(carried[name], tensor)
                carried = updated
            environment.update({carry.result: carried[carry.input] for carry in operation.carry})
        else:
            execute_stack(operation, environment, modules, executor=executor,
                          call_path=occurrence, instance_path=instance_path)


def execute_stack(operation: PlanStack, environment: dict[str, Tensor], modules: nn.ModuleDict, *,
                  executor: CallExecutor | None = None, call_path: tuple[int, ...] = (), instance_path: str = "") -> None:
    """Run distinct child instances with shared static inputs and explicit port-carried state."""
    carried = {carry.input: environment[carry.initial] for carry in operation.carry}
    static = {port: environment[value] for port, value in operation.inputs.items()}
    for index, path in enumerate(operation.instances):
        inputs = {**static, **carried}
        values = (invoke(modules[path], inputs) if executor is None else
                  executor(modules[path], inputs, ExecutionOccurrence(instance_path + path, call_path + (index,))))
        updated = {carry.input: values[carry.output] for carry in operation.carry}
        for name, tensor in updated.items():
            compatible_carry(carried[name], tensor)
        carried = updated
    environment.update({carry.result: carried[carry.input] for carry in operation.carry})
