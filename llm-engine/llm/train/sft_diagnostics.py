"""Bounded model-independent masked and held-out SFT measurements."""

from contextlib import nullcontext

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from llm.architecture.numerical import preserved_model
from llm.model.language import logits, model_facts
from llm.train.diagnostics import Probes, DIAGNOSTIC_ROWS, _bucket_means, _conversation_losses, _weighted_mean
from llm.train.model_diagnostics import ModelDiagnostics
from llm.train.model_memory import owned_tensors


def split_losses(values: Tensor, targets: Tensor, mask: Tensor, documents: np.ndarray | None) -> dict:
    """Use the existing SFT reductions for actual target/prompt, position and conversation losses."""
    losses = F.cross_entropy(values.float().reshape(-1, values.shape[-1]), targets.reshape(-1), reduction="none").reshape_as(targets)
    trained = mask.float()
    result = {"loss_masked": _weighted_mean(losses, trained),
              "loss_unmasked": _weighted_mean(losses, 1.0 - trained),
              "loss_by_position": _bucket_means(losses, trained)}
    if documents is not None:
        result.update(_conversation_losses(losses, trained, documents))
    return {name: value for name, value in result.items() if value is not None}


def probe_batch(array: np.ndarray, *, columns: int, vocabulary: int, device: torch.device) -> Tensor:
    """Validate a bounded actual probe before allocating its device tensor; use a private copy of readonly mappings."""
    if (array.dtype != np.int64 or array.ndim != 2 or not 1 <= array.shape[0] <= DIAGNOSTIC_ROWS
            or array.shape[1] != columns or np.any(array < 0) or np.any(array >= vocabulary)):
        raise ValueError("SFT diagnostic probe differs from the inherited vocabulary/context")
    return torch.tensor(array, dtype=torch.int64, device=device)


def reference_kl(model: nn.Module, reference: nn.Module, batch: Tensor) -> float:
    """Measure mean KL(base || tuned) in nats per token one held-out row at a time."""
    total = 0.0
    for row in batch:
        tokens = row.unsqueeze(0)
        tuned = logits(model, tokens).float().log_softmax(-1)
        base = logits(reference, tokens).float().log_softmax(-1)
        total += float((base.exp() * (base - tuned)).sum(-1).mean())
    return total / len(batch)


def sample_sft(sampler: ModelDiagnostics, inputs: Tensor, targets: Tensor | None,
               mask: Tensor | None, documents: np.ndarray | None, probes: Probes | None) -> dict:
    """Measure only available SFT values with separate allocation bounds and preserved model/reference RNG/modes/state."""
    model = sampler.model
    probes = probes or Probes()
    reference = probes.reference if probes.chat is not None else None
    facts = model_facts(model)
    snapshots = sum(value.numel() * value.element_size() for value in owned_tensors(model).values())
    if reference is not None:
        if model_facts(reference) != facts or any(value.requires_grad for value in reference.parameters()):
            raise ValueError("SFT diagnostic reference must be a frozen compatible language model")
        snapshots += sum(value.numel() * value.element_size() for value in owned_tensors(reference).values())
    tokens = max(inputs.numel(), DIAGNOSTIC_ROWS * facts.context_window if probes.pretrain is not None else 0)
    sampler.admit(snapshots + tokens * facts.vocabulary_size * 32)
    measured = {}
    with preserved_model(model), (preserved_model(reference) if reference is not None else nullcontext()), torch.no_grad():
        model.eval()
        if reference is not None:
            reference.eval()
        if mask is not None and targets is not None:
            measured.update(split_losses(logits(model, inputs), targets, mask, documents))
        if probes.pretrain is not None:
            batch = probe_batch(probes.pretrain, columns=facts.context_window + 1,
                                vocabulary=facts.vocabulary_size, device=inputs.device)
            values = logits(model, batch[:, :-1])
            measured["pretrain_loss"] = float(F.cross_entropy(values.float().reshape(-1, facts.vocabulary_size),
                                                              batch[:, 1:].reshape(-1)))
        if probes.chat is not None and reference is not None:
            sampler.admit(snapshots + facts.context_window * facts.vocabulary_size * 48)
            batch = probe_batch(probes.chat, columns=facts.context_window,
                                vocabulary=facts.vocabulary_size, device=inputs.device)
            measured["base_kl"] = reference_kl(model, reference, batch)
    return measured
