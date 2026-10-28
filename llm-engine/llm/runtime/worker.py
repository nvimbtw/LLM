"""Child-only backend execution and resident model lifetime with bounded parent effects."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import resource
import select
import socket
import time
from typing import Protocol
from uuid import UUID, uuid4

from llm.architecture.errors import ArchitectureError
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.bootstrap import WorkerBootstrap, read_bootstrap
from llm.runtime.protocol import (
    ArtifactAdopted, ArtifactReady, Cancel, Completed, EffectOperation, EffectReply,
    EffectRequest, Failed, Generate, Inspect, Load, Measure, Phase, Progress,
    RuntimeArtifact, RuntimeEnvelope, RuntimePayload, RuntimePins, Shutdown, Train,
)
from llm.runtime.transport import receive, send


class WorkerCancelled(RuntimeError):
    """A cooperative boundary observed parent cancellation of the current request."""


class WorkerBackend(Protocol):
    """Child-local numerical backend; model/optimizer state never leaves its owning instance."""

    def handle(self, payload: RuntimePayload, context: WorkerContext) -> Completed:
        """Execute one complete typed operation and return only bounded artifact metadata."""
        ...

    def finish_request(self) -> None:
        """Drop request-local caches and partial tensor state while retaining the resident model."""
        ...

    def close(self) -> None:
        """Release all resident model/optimizer references before process-tree teardown."""
        ...


class WorkerContext:
    """Own the child's current IPC identity, grants, finite waits and progress cadence."""

    def __init__(self, bootstrap: WorkerBootstrap, directory: Path, channel: socket.socket) -> None:
        """Initialize without service/storage clients, actors, credentials or parent gate objects."""
        self.bootstrap = bootstrap
        self.directory = directory
        self.channel = channel
        self.artifacts = ArtifactStore(directory / "artifacts", bootstrap.policy.resources)
        self.request_id = bootstrap.startup_request_id
        self.phase: Phase = "import"
        self._sent = 0
        self._received = 0
        self._last_progress = 0.0
        self._last_phase: Phase | None = None
        self.pins: RuntimePins | None = None

    @property
    def device(self) -> str:
        """Expose the explicitly admitted worker device, without a worker-owned admission gate."""
        return self.bootstrap.policy.resources.device

    def emit(self, payload: RuntimePayload) -> None:
        """Send one bounded worker event under a finite socket deadline."""
        self._sent += 1
        send(self.channel, RuntimeEnvelope(
            worker_id=self.bootstrap.worker_id, request_id=self.request_id,
            sequence=self._sent, payload=payload,
        ))

    def receive(self, timeout: float | None = None) -> RuntimeEnvelope:
        """Validate the parent's globally ordered command stream, including guardian-injected cancel."""
        self.channel.settimeout(timeout)
        try:
            message = receive(self.channel)
        finally:
            self.channel.settimeout(1.0)
        message.require_sender("parent")
        message.require_context(
            worker_id=self.bootstrap.worker_id, request_id=message.request_id,
            next_sequence=self._received + 1,
        )
        self._received = message.sequence
        if isinstance(message.payload, (Cancel, Shutdown)):
            raise WorkerCancelled("request cancelled")
        return message

    def check_cancelled(self) -> None:
        """Observe cooperative cancellation without waiting or accepting an unrelated request."""
        if select.select([self.channel], [], [], 0)[0]:
            self.receive(0.1)
            raise ValueError("runtime_unsolicited_parent_command")

    def progress(self, event: Progress, *, force: bool = False) -> None:
        """Report bounded real work; phase transitions are immediate and repeated reports coalesce."""
        self.check_cancelled()
        self.phase = event.phase
        now = time.monotonic()
        interval = self.bootstrap.policy.progress_interval_seconds
        if force or event.phase != self._last_phase or now - self._last_progress >= interval:
            self.emit(event)
            self._last_phase = event.phase
            self._last_progress = now

    def effect(self, operation: EffectOperation) -> EffectReply:
        """Ask the parent for one actor-bound Rust effect; never retry an uncertain write."""
        request = EffectRequest(type="effect_request", operation_id=uuid4(), effect=operation)
        self.emit(request)
        reply = self.receive(self.bootstrap.policy.deadlines.effect_seconds).payload
        if not isinstance(reply, EffectReply) or reply.operation_id != request.operation_id:
            raise ValueError("runtime_effect_reply_identity")
        if reply.outcome != "completed":
            raise RuntimeError("parent-mediated effect failed or is uncertain")
        return reply

    def publish(self, artifact: RuntimeArtifact, *, completed_step: int, final: bool = False) -> None:
        """Wait for durable parent adoption before another checkpoint may be produced."""
        self.artifacts.verify(artifact)
        notice = ArtifactReady(
            type="artifact_ready", publication_id=uuid4(), artifact=artifact,
            completed_step=completed_step, final=final,
        )
        self.emit(notice)
        reply = self.receive(self.bootstrap.policy.deadlines.artifact_adoption_seconds).payload
        if not isinstance(reply, ArtifactAdopted) or reply.publication_id != notice.publication_id:
            raise ValueError("runtime_artifact_acknowledgement")
        if reply.artifact_id != artifact.artifact_id or reply.outcome != "adopted":
            raise RuntimeError("parent rejected artifact publication")
        self.artifacts.remove(artifact)

    def require_pins(self, payload: RuntimePayload) -> None:
        """Reject stale resident pins; empty-worker training establishes exact final authorization inside its backend."""
        if isinstance(payload, (Train, Generate, Measure)):
            if isinstance(payload, Train) and payload.pins is None:
                if self.pins is not None:
                    raise ValueError("runtime_training_bootstrap_requires_empty_worker")
                return
            if self.pins is None:
                raise ValueError("runtime_model_not_loaded")
            payload.pins.require_current(self.pins)


def configure_limits(bootstrap: WorkerBootstrap, directory: Path) -> None:
    """Apply per-process bounds before source import; guardian owns aggregate tree supervision."""
    resources = bootstrap.policy.resources
    resource.setrlimit(resource.RLIMIT_AS, (resources.process_address_bytes,) * 2)
    resource.setrlimit(resource.RLIMIT_FSIZE, (resources.artifact_bytes,) * 2)
    resource.setrlimit(resource.RLIMIT_NOFILE, (resources.open_files_per_process,) * 2)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    cache = directory / "cache"
    cache.mkdir(mode=0o700)
    os.environ["TMPDIR"] = str(cache)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(cache / "inductor")
    os.environ["TRITON_CACHE_DIR"] = str(cache / "triton")
    if resources.device == "cuda":
        configure_cuda(bootstrap)


def configure_cuda(bootstrap: WorkerBootstrap) -> None:
    """Set a worker-side allocator cap with the existing driver margin before custom imports."""
    import torch
    from llm.train import memory

    if not torch.cuda.is_available():
        raise RuntimeError("requested CUDA device is unavailable")
    cap = min(bootstrap.policy.resources.device_allocator_bytes, memory.budget_bytes("cuda"))
    if cap <= 0:
        raise ValueError("runtime_device_memory_unavailable")
    memory.guard("cuda", cap)


def execute_request(backend: WorkerBackend, context: WorkerContext, message: RuntimeEnvelope) -> None:
    """Run one request with owned cleanup and bounded failures, exposing domain reasons only for administrative training."""
    context.request_id = message.request_id
    context._last_phase = None
    try:
        if context.bootstrap.metadata_only and not isinstance(message.payload, Inspect):
            raise ValueError("runtime_inspection_worker_cannot_execute")
        context.require_pins(message.payload)
        result = backend.handle(message.payload, context)
        if not isinstance(result, Completed):
            raise ValueError("runtime_backend_result")
        backend.finish_request()
        if isinstance(message.payload, Load) and result.outcome == "ok":
            context.pins = message.payload.pins
        context.emit(result)
    except WorkerCancelled:
        backend.finish_request()
        context.emit(Completed(type="completed", outcome="cancelled"))
    except Exception as error:
        backend.finish_request()
        if isinstance(message.payload, Train) and isinstance(error, ArchitectureError):
            context.emit(Failed(type="failed", phase=context.phase, code=error.code, message=str(error)[:512]))
            return
        if isinstance(message.payload, Train) and type(error) is ValueError:
            context.emit(Failed(type="failed", phase=context.phase, code="runtime_backend_failed",
                                message=f"ValueError: {error}"[:512]))
            return
        context.emit(Failed(
            type="failed", phase=context.phase, code="runtime_backend_failed", message=type(error).__name__,
        ))


def run_worker(directory: Path, channel_fd: int) -> None:
    """Import only the approved backend in this fresh child and retain it across complete requests."""
    bootstrap = read_bootstrap(directory)
    channel = socket.socket(fileno=channel_fd)
    channel.settimeout(1.0)
    context = WorkerContext(bootstrap, directory, channel)
    backend: WorkerBackend | None = None
    try:
        context.progress(Progress(type="progress", phase="import", completed=0, unit="phases"))
        configure_limits(bootstrap, directory)
        module, factory = bootstrap.backend.split(":", 1)
        constructor = getattr(importlib.import_module(module), factory)
        context.progress(Progress(type="progress", phase="construction", completed=0, unit="phases"))
        backend = constructor(context)
        context.emit(Completed(type="completed", outcome="ok"))
        while True:
            execute_request(backend, context, context.receive())
    except (WorkerCancelled, EOFError, BrokenPipeError):
        pass
    except Exception as error:
        context.emit(Failed(
            type="failed", phase=context.phase, code="runtime_bootstrap_failed", message=type(error).__name__,
        ))
    finally:
        if backend is not None:
            backend.close()
        channel.close()
