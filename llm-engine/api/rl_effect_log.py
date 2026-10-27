"""Durable parent ownership of gym resources and uncertain writes, without retaining gym trajectories."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, TypeAdapter

from llm.architecture.package_integrity import decode_json, encode_json
from llm.architecture.spec import Contract, NonnegativeInt, Sha256
from llm.gym.effect_records import CreatedRun, CreateRun, Identifier
from llm.runtime.artifacts import sync_directory
from llm.runtime.protocol import EffectRequest


class Authority(Contract):
    """Bind an engine-owned ledger to one queued job, actor and immutable configuration digest."""

    type: Literal["authority"] = "authority"
    job_id: UUID
    actor_id: UUID
    configuration_sha256: Sha256
    first_iteration: Annotated[NonnegativeInt, Field(lt=2**53)] = 0
    training_group_start: Annotated[NonnegativeInt, Field(lt=2**64)] = 0


class Issued(Contract):
    """Persist operation identity and bounded grant metadata before Rust issuance, excluding commands and replies."""

    type: Literal["issued"] = "issued"
    request: EffectRequest
    status: Literal["running", "done", "failed", "abandoned"] | None = None
    creation: CreateRun | None = None


class EpisodeGrant(Contract):
    """Retain only a resource ID and authorized tool names for subsequent episode writes."""

    id: UUID
    tools: Annotated[tuple[Identifier, ...], Field(max_length=64)]


class Acquired(Contract):
    """Fsync newly returned resource identities before the worker can observe their creation acknowledgement."""

    type: Literal["acquired"] = "acquired"
    operation_id: UUID
    run_id: UUID
    episodes: Annotated[tuple[EpisodeGrant, ...], Field(min_length=1, max_length=256)]


class Completed(Contract):
    """Record a validated Rust response without retaining the observation, grade or sampled reply."""

    type: Literal["completed"] = "completed"
    operation_id: UUID


class RetirementIssued(Contract):
    """Parent-only cleanup intent for a known resource, distinct from replaying a worker's uncertain tool/finish write."""

    type: Literal["retirement_issued"] = "retirement_issued"
    operation_id: UUID
    resource_id: UUID
    action: Literal["close_episode", "abandon_run"]


class Retired(Contract):
    """Validated response to an already-issued cleanup operation; an absent response never permits retry."""

    type: Literal["retired"] = "retired"
    operation_id: UUID


Event = Annotated[Issued | Acquired | Completed | RetirementIssued | Retired, Field(discriminator="type")]
EVENT = TypeAdapter(Event)


@dataclass
class OwnedEpisode:
    """Parent-owned authorization and terminal fence for one episode, reconstructed from issued writes."""

    run_id: UUID
    tools: tuple[str, ...]
    steps: int = 0
    terminal: bool = False


class GymEffectLog:
    """Own one bounded append-only job ledger under the engine's exclusive queue/session lifecycle.

    The dispatcher serializes access. An incomplete issued operation fences the
    entire job across restart; a recovery job must create a new ledger and gym
    runs. This records ownership and issuance, while gyms retains all task,
    command, observation and reward history. Exhaustion fails closed.
    """

    def __init__(self, path: Path, authority: Authority, *, max_bytes: int) -> None:
        """Create or validate bounded durable authority and reconstruct resources without importing executable data."""
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("gym effect log requires a positive admitted byte budget")
        self.path = path
        self.authority = authority
        self.max_bytes = max_bytes
        self.pending: Issued | None = None
        self.issued: set[UUID] = set()
        self.runs: dict[UUID, bool] = {}
        self.episodes: dict[UUID, OwnedEpisode] = {}
        self.creations: dict[str, int] = {"train": 0, "validation": 0, "test": 0}
        self.retirements: dict[UUID, RetirementIssued] = {}
        self.retired: set[UUID] = set()
        self._acquired = False
        self._size = 0
        if path.exists():
            self._restore()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._append(authority, create=True)

    @classmethod
    def open_existing(cls, path: Path, *, max_bytes: int) -> GymEffectLog:
        """Read a bounded data-only authority header for restart cleanup even after the original job history was pruned."""
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            raw = source.readline(4097)
        if len(raw) > 4096 or not raw.endswith(b"\n"):
            raise ValueError("gym effect authority header is oversized or incomplete")
        authority = Authority.model_validate_json(encode_json(decode_json(raw)))
        return cls(path, authority, max_bytes=max_bytes)

    def _restore(self) -> None:
        """Reject oversized, partial, duplicate-field or foreign-job evidence before reconstructing write fences."""
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW)
        with os.fdopen(descriptor, "rb") as source:
            if os.fstat(source.fileno()).st_size > self.max_bytes:
                raise ValueError("gym effect log exceeds its admitted budget")
            raw = source.read(self.max_bytes + 1)
        if len(raw) > self.max_bytes or not raw.endswith(b"\n"):
            raise ValueError("gym effect log is oversized or incomplete")
        lines = raw.splitlines()
        restored = Authority.model_validate_json(encode_json(decode_json(lines[0])))
        if restored != self.authority:
            raise ValueError("gym effect log belongs to another queued job or actor")
        for line in lines[1:]:
            event = EVENT.validate_json(encode_json(decode_json(line)))
            self._validate(event)
            self._apply(event)
        self._size = len(raw)

    def _append(self, record: Contract, *, create: bool = False) -> None:
        """Append and fsync one bounded record before mutating in-memory authorization or acknowledging a remote result."""
        raw = encode_json(record) + b"\n"
        if self._size + len(raw) > self.max_bytes:
            raise ValueError("gym effect log exceeds its admitted budget")
        flags = os.O_WRONLY | os.O_NOFOLLOW | (os.O_CREAT | os.O_EXCL if create else os.O_APPEND)
        descriptor = os.open(self.path, flags, 0o600)
        with os.fdopen(descriptor, "ab") as output:
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        sync_directory(self.path.parent)
        self._size += len(raw)

    def _record(self, event: Event) -> None:
        """Validate a transition, persist it, then expose its new resource state to the sole dispatcher owner."""
        self._validate(event)
        self._append(event)
        self._apply(event)

    def require_settled(self) -> None:
        """Refuse every new operation while an earlier write lacks a durable validated response."""
        if self.pending is not None or self.retirements:
            raise ValueError("gym job is fenced; recovery requires a new job and fresh gym runs")

    def issue(self, request: EffectRequest, *, status: str | None = None, creation: CreateRun | None = None) -> None:
        """Persist an authorized write identity and terminal fence before the first external attempt."""
        self._record(Issued(request=request, status=status, creation=creation))

    def acquire(self, operation_id: UUID, result: CreatedRun) -> None:
        """Retain validated new IDs and tools before the creation result can be granted to a worker."""
        self._record(Acquired(operation_id=operation_id, run_id=result.id,
            episodes=tuple(EpisodeGrant(id=episode.id, tools=episode.tools) for episode in result.episodes)))

    def complete(self, operation_id: UUID) -> None:
        """Retire one pending write only after its response has passed the operation-specific boundary checks."""
        self._record(Completed(operation_id=operation_id))

    def retire(self, intent: RetirementIssued) -> None:
        """Fsync one new parent cleanup intent without clearing or replaying any uncertain worker write."""
        self._record(intent)

    def retired_response(self, operation_id: UUID) -> None:
        """Record a validated cleanup response while keeping all original worker uncertainty intact."""
        self._record(Retired(operation_id=operation_id))

    def episode(self, episode_id: UUID) -> OwnedEpisode:
        """Authorize a live episode owned by this job and a run whose terminal transition has not been issued."""
        owned = self.episodes.get(episode_id)
        if owned is None or owned.terminal or self.runs[owned.run_id]:
            raise ValueError("gym episode is foreign or terminal")
        return owned

    def _validate(self, event: Event) -> None:
        """Check durable transition order, ownership, uniqueness and terminal boundaries without changing state."""
        if isinstance(event, (RetirementIssued, Retired)):
            self._validate_retirement(event)
            return
        if isinstance(event, Issued):
            self.require_settled()
            request = event.request
            effect = request.effect
            if request.operation_id in self.issued:
                raise ValueError("gym operation was already issued")
            if (effect.operation == "gym_status") != (event.status is not None):
                raise ValueError("gym status intent is inconsistent")
            if (effect.operation == "gym_create") != (event.creation is not None):
                raise ValueError("gym creation intent is inconsistent")
            if effect.operation in {"gym_step", "gym_finish", "gym_abandon"}:
                self.episode(effect.episode_id)
            elif effect.operation == "gym_status":
                if effect.run_id not in self.runs or self.runs[effect.run_id]:
                    raise ValueError("gym run is foreign or terminal")
            elif effect.operation != "gym_create":
                raise ValueError("gym ownership ledger accepts only writes")
            return
        if self.pending is None or self.pending.request.operation_id != event.operation_id:
            raise ValueError("gym response has no matching issued operation")
        if isinstance(event, Acquired):
            identities = [episode.id for episode in event.episodes]
            if (self.pending.request.effect.operation != "gym_create" or self._acquired
                    or event.run_id in self.runs or len(set(identities)) != len(identities)
                    or any(identity in self.episodes for identity in identities)):
                raise ValueError("gym creation response aliases existing ownership")
        elif self.pending.request.effect.operation == "gym_create" and not self._acquired:
            raise ValueError("gym creation acknowledgement requires durable resource ownership")

    def _validate_retirement(self, event: RetirementIssued | Retired) -> None:
        """Allow fresh parent cleanup of known resources, fencing each cleanup attempt before issuance and never retrying it."""
        if isinstance(event, Retired):
            if event.operation_id not in self.retirements or event.operation_id in self.retired:
                raise ValueError("gym cleanup has no matching outstanding intent")
            return
        if event.operation_id in self.issued:
            raise ValueError("gym cleanup operation was already issued")
        if event.action == "close_episode":
            episode = self.episodes.get(event.resource_id)
            if episode is None or episode.terminal:
                raise ValueError("gym cleanup episode is foreign or terminal")
        elif event.resource_id not in self.runs or self.runs[event.resource_id]:
            raise ValueError("gym cleanup run is foreign or terminal")

    def _apply(self, event: Event) -> None:
        """Apply a previously validated and persisted event to the parent's bounded authorization state."""
        if isinstance(event, RetirementIssued):
            self.issued.add(event.operation_id)
            self.retirements[event.operation_id] = event
            if event.action == "close_episode":
                self.episodes[event.resource_id].terminal = True
            else:
                self.runs[event.resource_id] = True
        elif isinstance(event, Retired):
            self.retired.add(event.operation_id)
        elif isinstance(event, Issued):
            self.pending = event
            self.issued.add(event.request.operation_id)
            self._acquired = False
            effect = event.request.effect
            if event.creation is not None:
                self.creations[event.creation.split] += 1
            elif effect.operation == "gym_step":
                self.episodes[effect.episode_id].steps += 1
            elif effect.operation in {"gym_finish", "gym_abandon"}:
                self.episodes[effect.episode_id].terminal = True
            elif effect.operation == "gym_status" and event.status != "running":
                self.runs[effect.run_id] = True
        elif isinstance(event, Acquired):
            self.runs[event.run_id] = False
            for episode in event.episodes:
                self.episodes[episode.id] = OwnedEpisode(event.run_id, episode.tools)
            self._acquired = True
        else:
            self.pending = None
            self._acquired = False
