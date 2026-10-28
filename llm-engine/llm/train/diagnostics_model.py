"""Shared-loop adapter for bounded semantic model diagnostics, without Transformer-specific metrics."""

from dataclasses import dataclass

from torch import nn
import torch

from llm.architecture.errors import ArchitectureError
from llm.train.diagnostics import Probes, diagnostic_rows
from llm.train.model_diagnostics import ModelDiagnostics
from llm.train.sft_diagnostics import sample_sft


@dataclass(frozen=True)
class ModelReport:
    """Only measured Parameter-owner gradients/updates and actual forward occurrences for one reported step."""

    gradients: dict[str,float]
    updates: dict[str,float] | None
    occurrences: tuple[dict,...] | None
    step: int
    supervised: dict | None = None

    def as_dict(self) -> dict:
        """Expose measured semantic values without inventing attention, layer or embedding diagnostics."""
        result = {"step":self.step,"parameter_grad_norm":self.gradients}
        if self.updates is not None:
            result["parameter_update_ratio"] = self.updates
        if self.occurrences is not None:
            result["occurrences"] = self.occurrences
        if self.supervised is not None:
            result.update(self.supervised)
        return result


class ModelCollector:
    """Own optional generic diagnostics at the existing loop's observe/before/after update boundaries."""

    def __init__(self, model: nn.Module, probes: Probes | None = None):
        """Retain the admitted model and bounded sampler without allocating diagnostic snapshots yet."""
        self.samples = ModelDiagnostics(model)
        self.occurrences = None
        self.probes = probes
        self.supervised = None

    def observe(self, inputs, targets, weights, documents) -> None:
        """Sample actual occurrences and available SFT measurements with one optional diagnostic failure owner."""
        rows = diagnostic_rows(inputs.shape[0])
        self.occurrences = self.samples.sample(inputs[:rows])
        self.supervised = None
        if self.samples.disabled_reason is not None or weights is None and self.probes is None:
            return
        try:
            self.supervised = sample_sft(self.samples, inputs[:rows],
                targets[:rows] if targets is not None else None, weights[:rows] if weights is not None else None,
                documents, self.probes)
        except (torch.cuda.OutOfMemoryError, ArchitectureError) as error:
            if isinstance(error, ArchitectureError) and error.code != "diagnostic_budget":
                raise
            self.samples.disable(error)

    def before_step(self) -> dict[str,float] | None:
        """Capture bounded owner gradients/CPU weights immediately before the real optimizer update."""
        return self.samples.before_step()

    def after_step(self, gradients: dict[str,float], step: int) -> ModelReport:
        """Measure owner update ratios after the update and release the diagnostic weight snapshot."""
        return ModelReport(gradients,self.samples.after_step(),self.occurrences,step,self.supervised)
