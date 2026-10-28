"""Pure request/deadline reduction consumed by the independent trusted guardian."""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from uuid import UUID

from llm.runtime.protocol import (
    ArtifactAdopted, ArtifactReady, Completed, EffectReply, EffectRequest,
    Progress, RuntimeDeadlines, RuntimeEnvelope, Tokens,
)


@dataclass
class Watchdog:
    """Track one active request; phase labels/heartbeats cannot extend absolute bounds."""

    limits: RuntimeDeadlines
    request_id: UUID
    kind: str = "bootstrap"
    phase: str = "import"
    operation_deadline: float = field(default_factory=time.monotonic)
    phase_deadline: float = field(default_factory=time.monotonic)
    last_work: float = field(default_factory=time.monotonic)
    idle_since: float = field(default_factory=time.monotonic)
    waiting_until: float | None = None
    waiting_id: UUID | None = None
    waiting_kind: str | None = None
    work: dict[tuple[str, str], int] = field(default_factory=dict)

    def bootstrap(self, now: float) -> None:
        """Start bounded trusted backend import/construction before accepting commands."""
        self.operation_deadline = now + self.limits.import_seconds + self.limits.construction_seconds
        self.phase_deadline = now + self.limits.import_seconds

    def command(self, message: RuntimeEnvelope, now: float) -> None:
        """Admit one operation or its matched effect/publication acknowledgement."""
        payload = message.payload
        if isinstance(payload, (EffectReply, ArtifactAdopted)):
            if message.request_id != self.request_id:
                raise ValueError("runtime_acknowledgement_request_identity")
            identity = payload.operation_id if isinstance(payload, EffectReply) else payload.publication_id
            expected_kind = "effect" if isinstance(payload, EffectReply) else "artifact"
            if identity != self.waiting_id or expected_kind != self.waiting_kind:
                raise ValueError("runtime_unsolicited_acknowledgement")
            self.waiting_until = self.waiting_id = self.waiting_kind = None
            return
        if self.kind != "idle":
            raise ValueError("runtime_concurrent_request")
        self.kind = payload.type
        self.request_id = message.request_id
        self.phase = "load" if payload.type == "load" else "validation"
        self.last_work = now
        self.work.clear()
        duration = {
            "inspect": self.limits.validation_seconds, "validate": self.limits.validation_seconds,
            "load": self.limits.construction_seconds, "generate": self.limits.inference_seconds,
            "measure": self.limits.measurement_seconds, "train": self.limits.training_progress_seconds,
        }[payload.type]
        self.operation_deadline = now + duration
        self.phase_deadline = self.operation_deadline

    def event(self, message: RuntimeEnvelope, now: float) -> None:
        """Observe finite real work and matched bounded waits without trusting wall-time heartbeats."""
        if message.request_id != self.request_id or self.kind == "idle":
            raise ValueError("runtime_stale_worker_event")
        payload = message.payload
        if isinstance(payload, Progress):
            if payload.phase != self.phase:
                self.phase = payload.phase
                if payload.phase in {"import", "construction"}:
                    duration = getattr(self.limits, f"{payload.phase}_seconds")
                    self.phase_deadline = now + duration
                else:
                    self.phase_deadline = self.operation_deadline
            key = (payload.phase, payload.unit)
            if payload.unit in {"batches", "steps", "iterations", "tokens"}:
                if payload.completed > self.work.get(key, 0):
                    self.last_work = now
                    self.work[key] = payload.completed
        if isinstance(payload, Tokens):
            self.last_work = now
        if isinstance(payload, (EffectRequest, ArtifactReady)):
            if self.waiting_id is not None:
                raise ValueError("runtime_multiple_outstanding_effects")
            effect = isinstance(payload, EffectRequest)
            self.waiting_id = payload.operation_id if effect else payload.publication_id
            self.waiting_kind = "effect" if effect else "artifact"
            self.waiting_until = now + (self.limits.effect_seconds if effect else self.limits.artifact_adoption_seconds)
        if isinstance(payload, Completed):
            if self.waiting_id is not None:
                raise ValueError("runtime_completion_before_adoption")
            self.kind = "idle"
            self.idle_since = now

    def expired(self, now: float) -> bool:
        """Return true for idle, absolute phase/request, effect or stalled real training work."""
        if self.kind == "idle":
            return now > self.idle_since + self.limits.serving_idle_seconds
        if self.waiting_until is not None and now > self.waiting_until:
            return True
        if self.kind == "train":
            if self.phase in {"import", "construction"} and now > self.phase_deadline:
                return True
            return now > self.last_work + self.limits.training_progress_seconds
        return now > min(self.operation_deadline, self.phase_deadline)
