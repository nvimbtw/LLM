"""Explicit occurrence recomputation for supported immutable named-port execution."""

from collections.abc import Iterator, Mapping
from functools import partial

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from llm.architecture.execute import CallExecutor, CompositionModule, invoke
from llm.architecture.interfaces import ExecutionOccurrence
from llm.architecture.plan import PlanCall, PlanRepeat


def local_calls(operations: tuple, call_path: tuple[int, ...] = ()) -> Iterator[tuple[str, tuple[int, ...]]]:
    """Enumerate each bounded call in execution order without conflating weight sharing with occurrences."""
    for index, operation in enumerate(operations):
        occurrence = call_path + (index,)
        if isinstance(operation, PlanCall):
            yield operation.instance, occurrence
        elif isinstance(operation, PlanRepeat):
            for iteration in range(operation.count):
                yield from local_calls(operation.body, occurrence + (iteration,))
        else:
            for number, path in enumerate(operation.instances):
                yield path, occurrence + (number,)


def module_occurrences(model: CompositionModule, *, prefix: str = "", call_path: tuple[int, ...] = ()) -> tuple[tuple[ExecutionOccurrence, nn.Module], ...]:
    """Resolve stable module/occurrence pairs across nested recipes and shared fixed loops."""
    result = []
    for path, occurrence in local_calls(model.plan.operations, call_path):
        module = model.instances[path]
        identity = prefix + path
        if isinstance(module, CompositionModule):
            result.extend(module_occurrences(module, prefix=identity + "/", call_path=occurrence))
        else:
            result.append((ExecutionOccurrence(identity, occurrence), module))
    return tuple(result)


def recomputed_call(module: nn.Module, inputs: Mapping[str, Tensor], occurrence: ExecutionOccurrence, *,
                    selected: frozenset[ExecutionOccurrence]) -> dict[str, Tensor]:
    """Replay only selected explicit-state calls with nonreentrant autograd and preserved CPU/device RNG.

    Selection changes storage policy, never loop count or parameter ownership.
    Evaluation and no-grad passes use eager arithmetic. Named-port mutation and
    interface guards remain active on both original and replay computation.
    """
    if isinstance(module, CompositionModule):
        return module.forward_recomputed(inputs, selected=selected, call_path=occurrence.call_path,
                                          instance_path=occurrence.instance_path + "/")
    if torch.compiler.is_compiling() and not selected:
        with torch.fx.traceback.annotate({"slate_occurrence": (occurrence.instance_path, occurrence.call_path)}):
            return invoke(module, inputs)
    if occurrence in selected and module.training and torch.is_grad_enabled():
        return checkpoint(invoke, module, dict(inputs), use_reentrant=False, preserve_rng_state=True)
    return invoke(module, inputs)


def recomputation_executor(selected: frozenset[ExecutionOccurrence]) -> CallExecutor:
    """Bind one immutable between-step policy without retaining per-request tensors."""
    return partial(recomputed_call, selected=selected)
