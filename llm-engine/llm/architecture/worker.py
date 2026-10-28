"""Production child-only validation backend consuming verified session artifact grants."""

import gc
from contextlib import contextmanager
from collections.abc import Iterator

import torch

from llm.architecture.validation import validate_snapshot
from llm.architecture.validation_records import VALIDATION_PHASES, ValidationSnapshot
from llm.architecture.spec import RuntimeProfile
from llm.runtime.protocol import Completed, Inspect, Progress, Validate
from llm.runtime.worker import WorkerContext


@contextmanager
def validation_math_settings(profile: RuntimeProfile | None, device: str) -> Iterator[None]:
    """Apply a CUDA request's explicit TF32 policy inside its admitted worker, restoring both backend flags on success, failure or cancellation; inspection/CPU requests leave them unchanged."""
    if profile is None or device != "cuda":
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = profile.allow_tf32
        torch.backends.cudnn.allow_tf32 = profile.allow_tf32
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


class ValidationBackend:
    """Own bounded source inspection/full preflight without catalog/network/device-gate clients."""

    def __init__(self, context: WorkerContext):
        """Bind the admitted child context; importing this factory executes no custom source."""
        self.context = context

    def progress(self, phase: str) -> None:
        """Observe cancellation and publish the actual numerical phase to the parent watchdog."""
        runtime_phase = {"inspection": "inspection", "resolution": "construction", "numerical": "validation"}.get(phase, phase)
        self.context.progress(Progress(type="progress", phase=runtime_phase, completed=VALIDATION_PHASES.index(phase),
                                       total=len(VALIDATION_PHASES),unit="phases"))

    def handle(self, payload, context: WorkerContext) -> Completed:
        """Run a typed immutable snapshot and return bounded complete JSON without publishing evidence."""
        if not isinstance(payload, (Inspect, Validate)):
            raise ValueError("validation backend accepts inspection/validation only")
        snapshot = ValidationSnapshot.model_validate_json(context.artifacts.read_bytes(payload.snapshot))
        if isinstance(payload, Inspect) != (snapshot.request.mode == "inspect"):
            raise ValueError("validation snapshot mode differs from runtime command")
        with validation_math_settings(snapshot.request.profile, context.device):
            output = validate_snapshot(snapshot, device=context.device, progress=self.progress)
        encoded = output.model_dump_json().encode("utf-8")
        if len(encoded) > 8 << 20:
            raise ValueError("validation output byte budget exceeded")
        result = context.artifacts.write_bytes(encoded, "validation_result_v1")
        return Completed(type="completed", outcome="ok", result=result)

    def finish_request(self) -> None:
        """Retain no source loader, model, optimizer, gradients or request tensors between validations."""
        torch._dynamo.reset()
        gc.collect()
        if self.context.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def close(self) -> None:
        """Release the child context when its supervised process lifetime ends."""
        self.context = None
