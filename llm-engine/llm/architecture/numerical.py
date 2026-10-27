"""Bounded numerical probes preserving caller RNG, modes, tensors and gradients."""

from contextlib import contextmanager

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm.architecture.errors import ArchitectureError


@contextmanager
def preserved_model(model: nn.Module):
    """Restore RNG, module modes and original gradient references after probes; reject state mutation."""
    modes = [(module, module.training) for module in model.modules()]
    parameters = list(model.parameters())
    gradients = [parameter.grad for parameter in parameters]
    tensors = [(tensor, tensor.detach().clone(), tensor._version) for tensor in (*parameters, *model.buffers())]
    devices = sorted({tensor.device.index for tensor in parameters if tensor.device.type == "cuda"})
    with torch.random.fork_rng(devices=devices):
        try:
            yield
        finally:
            changed = any(tensor._version != version or not torch.equal(tensor.detach(), before)
                          for tensor, before, version in tensors)
            for module, training in modes:
                module.training = training
            for parameter, gradient in zip(parameters, gradients):
                parameter.grad = gradient
            if changed:
                raise ArchitectureError("persistent_mutation", "numerical probes observed persistent tensor mutation")


def close(actual: Tensor, expected: Tensor, precision: str) -> None:
    """Compare bounded outputs/gradients using explicit precision-specific numerical tolerances."""
    tolerance = 0.03 if precision in {"fp16", "bf16"} else 2e-5
    try:
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    except AssertionError as error:
        raise ArchitectureError("numerical_mismatch", "eager/captured or causal reference values differ") from error


def output_tensors(result: object) -> tuple[Tensor, ...]:
    """Require only finite declared tensor results, never opaque Python output state."""
    values = (result,) if isinstance(result, Tensor) else tuple(result.values()) if isinstance(result, dict) else ()
    if not values or any(not isinstance(value, Tensor) or not bool(torch.isfinite(value).all()) for value in values):
        raise ArchitectureError("nonfinite_output", "forward must return finite declared tensor outputs")
    return values


def probe_loss(result: object, tokens: Tensor | None) -> Tensor | None:
    """Compute finite next-token loss, or a singleton/tensor-only differentiability probe."""
    values = output_tensors(result)
    if not any(value.is_floating_point() for value in values):
        return None
    if tokens is not None and tokens.shape[1] > 1:
        loss = F.cross_entropy(values[0][:, :-1].float().reshape(-1, values[0].shape[-1]), tokens[:, 1:].reshape(-1))
    else:
        loss = sum(value.float().square().mean() for value in values if value.is_floating_point())
    if not isinstance(loss, Tensor) or not bool(torch.isfinite(loss)):
        raise ArchitectureError("nonfinite_loss", "numerical loss probe is not finite/differentiable")
    return loss


def equivalence(model: nn.Module, exported: nn.Module, args: tuple, precision: str,
                *, tokens: Tensor | None = None, check_rng: bool = False) -> None:
    """Compare eager/captured forward and gradients under identical controlled RNG draws.

    Fixed repeat capture contains explicit operation occurrences, so its backward
    comparison also checks the executor against complete captured unrolling.
    Optional RNG checking requires identical post-backward CPU/device draws.
    No optimizer step occurs, and zero gradients alone never imply disconnection.
    """
    devices = sorted({parameter.device.index for parameter in model.parameters() if parameter.device.type == "cuda"})
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    inputs = [(tensor, tensor.detach().clone(), tensor._version)
              for tensor in torch.utils._pytree.tree_leaves(args) if isinstance(tensor, Tensor)]
    with torch.random.fork_rng(devices=devices):
        cpu_state = torch.get_rng_state()
        cuda_states = {device: torch.cuda.get_rng_state(device) for device in devices}
        eager = model(*args)
        loss = probe_loss(eager, tokens)
        if parameters and (loss is None or not loss.requires_grad):
            raise ArchitectureError("disconnected_parameter", "trainable parameters have no path to the probed loss")
        eager_gradients = torch.autograd.grad(loss, parameters, allow_unused=True) if parameters else ()
        eager_cpu_rng = torch.get_rng_state() if check_rng else None
        eager_cuda_rng = {device: torch.cuda.get_rng_state(device) for device in devices} if check_rng else {}
        torch.set_rng_state(cpu_state)
        for device, state in cuda_states.items():
            torch.cuda.set_rng_state(state, device)
        captured = exported(*args)
        for actual, expected in zip(output_tensors(captured), output_tensors(eager)):
            close(actual, expected, precision)
        captured_loss = probe_loss(captured, tokens)
        captured_gradients = torch.autograd.grad(captured_loss, parameters, allow_unused=True) if parameters else ()
        if check_rng and (not torch.equal(eager_cpu_rng, torch.get_rng_state()) or any(
                not torch.equal(state, torch.cuda.get_rng_state(device)) for device, state in eager_cuda_rng.items())):
            raise ArchitectureError("recompute_rng", "replay and captured execution consumed different RNG draws")
    for eager_gradient, captured_gradient in zip(eager_gradients, captured_gradients):
        if eager_gradient is None or captured_gradient is None:
            raise ArchitectureError("disconnected_parameter", "trainable parameter has no path to the probed loss")
        if not bool(torch.isfinite(eager_gradient).all()) or not bool(torch.isfinite(captured_gradient).all()):
            raise ArchitectureError("nonfinite_gradient", "backward produced a nonfinite gradient")
        close(eager_gradient, captured_gradient, precision)
    if any(tensor._version != version or not torch.equal(tensor.detach(), before) for tensor, before, version in inputs):
        raise ArchitectureError("input_mutation", "probe changed caller-owned input tensors")


def causal_probes(model: nn.Module, tokens: Tensor, precision: str) -> None:
    """Check deterministic eval, suffix causality, prefix, batch and padding independence."""
    model.eval()
    with torch.no_grad():
        reference = model(tokens)
        close(model(tokens), reference, precision)
        if tokens.shape[1] > 1:
            changed = tokens.clone()
            changed[:, -1] = (changed[:, -1] + 1) % model.facts.vocabulary_size
            close(model(changed)[:, :-1], reference[:, :-1], precision)
            close(model(tokens[:, :-1]), reference[:, :-1], precision)
        for row in range(tokens.shape[0]):
            close(model(tokens[row:row + 1]), reference[row:row + 1], precision)
        if tokens.shape[1] > 1:
            live = max(1, tokens.shape[1] - 1)
            positions = torch.arange(tokens.shape[1], device=tokens.device).expand(tokens.shape[0], -1).clone()
            mask = torch.ones_like(tokens, dtype=torch.bool)
            mask[:, live:] = False
            positions[:, live:] = 0
            padded = model(tokens, positions=positions, key_mask=mask)
            close(padded[:, :live], model(tokens[:, :live]), precision)
            altered = tokens.clone()
            altered[:, live:] = (altered[:, live:] + 1) % model.facts.vocabulary_size
            close(model(altered, positions=positions, key_mask=mask)[:, :live], padded[:, :live], precision)
