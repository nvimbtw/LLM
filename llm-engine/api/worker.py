"""Explicit parent runner seam for handlers whose numerical work lives in a child."""

from collections.abc import Callable
from dataclasses import dataclass
from threading import Event
from typing import Any, Protocol, runtime_checkable
from uuid import UUID

from llm.model.device import DeviceLease
from llm.runtime.supervisor import WorkerManager
from llm.train.progress import Checkpoint


@dataclass(frozen=True)
class RegisteredCheckpoint:
    """Announce an already committed parent publication without asking the legacy progress sink to register it again."""

    frame: Checkpoint


@dataclass(frozen=True)
class SupervisedRun:
    """The actual parent job/admission context; none of it is serialized to the child."""

    job_id: UUID
    config: dict
    actor: str | None
    progress: Callable[[Any], None]
    cancel: Event
    lease: DeviceLease
    workers: WorkerManager


@runtime_checkable
class SupervisedHandler(Protocol):
    """Existing queue variation for child-based validation/training, separate from legacy callables."""

    def run_supervised(self, context: SupervisedRun) -> dict:
        """Run through context.workers and return only after artifact adoption and session cleanup."""
        ...
