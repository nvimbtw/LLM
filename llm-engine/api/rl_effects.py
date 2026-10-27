"""Parent-only gym authorization and bounded Rust dispatch for one queued numerical RL job."""

import hashlib
import logging
import math
from threading import Lock
import time
from typing import TypeVar
from uuid import UUID, uuid4

from pydantic import BaseModel

from api.rl_effect_log import GymEffectLog, RetirementIssued
from llm.architecture.package_integrity import decode_json, encode_json
from llm.backend import Hop, REQUEST_TIMEOUT_SECS
from llm.config import RlConfig
from llm.gym.effect_records import (
    Catalogue, ClosedEpisode, CreatedRun, CreateRun, EmptyResult, Finish, FinishResult, Status, Step, StepResult, UpdatedRun,
)
from llm.runtime.artifacts import ArtifactStore
from llm.runtime.protocol import EffectData, EffectRequest, RuntimeArtifact
from llm.train.rl_records import validate_configuration

Record = TypeVar("Record", bound=BaseModel)
MAX_PAYLOAD_BYTES = 4 << 20
LOG = logging.getLogger(__name__)


class GymResourceOwner:
    """Own actor-bound gym transport and durable resource retirement independently of prunable queued configuration."""

    def __init__(self, log: GymEffectLog, *, response_bytes: int) -> None:
        """Retain durable actor/resource authority and finite response limits under one serialized parent lifecycle."""
        if type(response_bytes) is not int or response_bytes < 1:
            raise ValueError("gym dispatcher needs a positive admitted response allowance")
        self.log = log
        self.response_bytes = response_bytes
        self._lock = Lock()
        self._stopped = False

    def _request(self, method: str, path: str, body: dict | None = None, *, timeout: float = REQUEST_TIMEOUT_SECS) -> object:
        """Make exactly one bounded request through Rust using credentials retained only by this parent callback."""
        with Hop(str(self.log.authority.actor_id)) as hop:
            kwargs = {} if body is None else {"json": body}
            return decode_json(hop.request_bytes(method, path, limit_bytes=self.response_bytes, timeout_seconds=timeout, **kwargs))

    def _close(self, episode_id: UUID, *, timeout: float = REQUEST_TIMEOUT_SECS) -> None:
        """Close one previously authorized episode once and require the matching finite service acknowledgement."""
        raw = self._request("POST", f"/internal/gym/episodes/{episode_id}/close", timeout=timeout)
        closed = ClosedEpisode.model_validate_json(encode_json(raw))
        if closed.episode_id != episode_id:
            raise ValueError("gym close response belongs to another episode")

    def _status(self, run_id: UUID, status: str, *, timeout: float = REQUEST_TIMEOUT_SECS) -> None:
        """Issue one previously authorized status transition and require its exact returned identity/value."""
        raw = self._request("POST", f"/internal/gym/runs/{run_id}/status", {"status": status}, timeout=timeout)
        updated = UpdatedRun.model_validate_json(encode_json(raw))
        if updated.id != run_id or updated.status != status:
            raise ValueError("gym status response differs from the issued transition")

    def retire_resources(self, *, seconds: float = REQUEST_TIMEOUT_SECS) -> None:
        """After worker/callback draining, fence this job and close only known untouched resources with new parent cleanup identities.

        Original uncertain writes remain fenced. Cleanup attempts are themselves
        fsynced and never retried, including after restart. The elapsed allowance
        is checked between finite-timeout requests; an in-flight I/O is not
        represented as forcibly stopped. Unissued resources remain for startup
        reconciliation and every failure retains its ownership evidence.
        """
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("gym cleanup requires a positive finite elapsed allowance")
        with self._lock:
            self._stopped = True
            deadline = time.monotonic() + seconds
            pending = [(identity, "close_episode") for identity, episode in self.log.episodes.items() if not episode.terminal]
            pending.extend((identity, "abandon_run") for identity, terminal in self.log.runs.items() if not terminal)
            for resource_id, action in pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return
                intent = RetirementIssued(operation_id=uuid4(), resource_id=resource_id, action=action)
                try:
                    self.log.retire(intent)
                    timeout = min(REQUEST_TIMEOUT_SECS, remaining)
                    if action == "close_episode":
                        self._close(resource_id, timeout=timeout)
                    else:
                        self._status(resource_id, "abandoned", timeout=timeout)
                    self.log.retired_response(intent.operation_id)
                except Exception:
                    LOG.exception("RL job %s cleanup attempt %s failed; retaining ownership evidence",
                        self.log.authority.job_id, intent.operation_id)


class GymDispatcher(GymResourceOwner):
    """Authorize resource IDs and issue one actor-bound Rust request at a time.

    The supervisor owns callback draining. Any exception latches this dispatcher;
    issued writes remain pending in durable ownership evidence, including when
    a response or a worker acknowledgement is lost. A subsequent recovery job
    receives a different ledger and creates fresh runs.
    """

    def __init__(self, cfg: RlConfig, log: GymEffectLog, *, response_bytes: int) -> None:
        """Bind immutable queued configuration, durable authority and a finite admitted upstream-response allowance."""
        super().__init__(log, response_bytes=response_bytes)
        validate_configuration(cfg)
        digest = hashlib.sha256(encode_json(cfg.model_dump(mode="json"))).hexdigest()
        if log.authority.configuration_sha256 != digest:
            raise ValueError("gym dispatcher configuration differs from durable job authority")
        if log.authority.first_iteration > cfg.iterations:
            raise ValueError("gym authority starts beyond the queued iteration schedule")
        self.cfg = cfg.model_copy(deep=True)
        self.base_model = UUID(cfg.base_model)

    def __call__(self, actor: str, request: EffectRequest, artifacts: ArtifactStore) -> RuntimeArtifact:
        """Authorize the broker's initiating actor and typed request, then grant only projected public service fields."""
        with self._lock:
            if self._stopped:
                raise ValueError("gym dispatcher is stopped")
            try:
                if actor != str(self.log.authority.actor_id):
                    raise ValueError("gym dispatcher actor differs from the queued job owner")
                self.log.require_settled()
                result = self._dispatch(request, artifacts)
                return artifacts.write_bytes(encode_json(result.model_dump(mode="json")), "effect_data_v1")
            except BaseException:
                self._stopped = True
                raise

    def _payload(self, payload: EffectData, artifacts: ArtifactStore, record: type[Record]) -> Record:
        """Read one verified finite grant and reject duplicate fields, nonfinite JSON or malformed typed operation data."""
        if payload.size_bytes > MAX_PAYLOAD_BYTES:
            raise ValueError("gym effect payload exceeds its admitted limit")
        return record.model_validate_json(encode_json(decode_json(artifacts.read_bytes(payload))))


    def _dispatch(self, request: EffectRequest, artifacts: ArtifactStore) -> BaseModel:
        """Select fixed Rust routes and typed domain checks; worker input cannot supply an actor, URL or HTTP method."""
        effect = request.effect
        if effect.operation == "gym_catalogue":
            return Catalogue.model_validate_json(encode_json({"gyms": self._request("GET", "/internal/gyms")}))
        if effect.operation == "gym_create":
            body = self._payload(effect.payload, artifacts, CreateRun)
            self._authorize_create(effect.gym_id, body)
            self.log.issue(request, creation=body)
            raw = self._request("POST", "/internal/gym/runs", {
                **body.model_dump(mode="json", exclude_none=True), "gym_id": effect.gym_id, "kind": "rl"})
            result = CreatedRun.model_validate_json(encode_json(raw))
            self._validate_created(effect.gym_id, body, result)
            self.log.acquire(request.operation_id, result)
        elif effect.operation in {"gym_step", "gym_finish", "gym_abandon"}:
            result = self._episode_write(request, artifacts)
        elif effect.operation == "gym_status":
            body = self._payload(effect.payload, artifacts, Status)
            self.log.issue(request, status=body.status)
            self._status(effect.run_id, body.status)
            result = EmptyResult()
        else:
            raise ValueError("operation is not supported by this RL gym dispatcher")
        self.log.complete(request.operation_id)
        return result

    def _episode_write(self, request: EffectRequest, artifacts: ArtifactStore) -> BaseModel:
        """Require job-owned live episode IDs, authorized tools and the queued round allowance before issuance."""
        effect = request.effect
        owned = self.log.episode(effect.episode_id)
        path = f"/internal/gym/episodes/{effect.episode_id}"
        if effect.operation == "gym_step":
            body = self._payload(effect.payload, artifacts, Step)
            if body.tool not in owned.tools or owned.steps >= self.cfg.max_rounds:
                raise ValueError("gym step tool or round allowance is unauthorized")
            self.log.issue(request)
            raw = self._request("POST", f"{path}/step", body.model_dump(mode="json"))
            return StepResult.model_validate_json(encode_json(raw))
        if effect.operation == "gym_finish":
            body = self._payload(effect.payload, artifacts, Finish)
            self.log.issue(request)
            raw = self._request("POST", f"{path}/finish", body.model_dump(mode="json"))
            return FinishResult.model_validate_json(encode_json(raw))
        self.log.issue(request)
        self._close(effect.episode_id)
        return EmptyResult()


    def _authorize_create(self, gym_id: str, body: CreateRun) -> None:
        """Restrict creation to the queued base model, lane, task pool, batch shape and inherited curriculum exam sizes."""
        cfg = self.cfg
        selected = {entry.id: entry.weight for entry in cfg.task_types}
        if gym_id != cfg.gym_id or body.model_name != self.base_model or not set(body.task_types) <= selected.keys():
            raise ValueError("gym creation differs from the queued model, gym or task selection")
        if body.split == "train":
            if self.log.creations["train"] >= cfg.iterations - self.log.authority.first_iteration:
                raise ValueError("gym creation exceeds the queued iteration allowance")
            expected = cfg.seed_base + self.log.authority.training_group_start + self.log.creations["train"] * cfg.tasks_per_batch
            if body.seed_base != expected:
                raise ValueError("gym creation differs from the queued completed-iteration seed position")
            last_seed = body.seed_base + cfg.tasks_per_batch - 1
            if (body.seed_base < cfg.seed_base or last_seed >= 2**64
                    or body.seed_base <= cfg.holdout_base <= last_seed):
                raise ValueError("gym training seeds exceed the uint64 boundary or overlap the queued holdout")
            if body.count != cfg.tasks_per_batch * cfg.group_size or body.group_size != cfg.group_size:
                raise ValueError("gym creation differs from the queued training batch")
            if gym_id == "general":
                if (tuple(body.task_types) != tuple(entry.id for entry in cfg.task_types)
                        or body.gym_weights != cfg.gym_weights or body.task_weights != selected):
                    raise ValueError("general gym creation differs from the queued sampling weights")
            elif len(body.task_types) != cfg.tasks_per_batch or body.gym_weights is not None or body.task_weights is not None:
                raise ValueError("gym training groups must be drawn from the queued task pool")
        else:
            schedule = cfg.curriculum
            samples = None if schedule is None else (schedule.validation_samples if body.split == "validation" else schedule.test_samples)
            if (samples is None or body.count != samples or body.group_size != 1 or len(body.task_types) != 1
                    or body.gym_weights is not None or body.task_weights is not None):
                raise ValueError("gym exam differs from the queued curriculum")
            if body.split == "test" and body.seed_base != cfg.holdout_base:
                raise ValueError("gym test seed differs from the queued holdout")
            limit = len(selected) * (cfg.iterations // schedule.check_every + 2) if body.split == "validation" else 2 * len(selected)
            if self.log.creations[body.split] >= limit:
                raise ValueError("gym creation exceeds the queued curriculum exam allowance")

    def _validate_created(self, gym_id: str, body: CreateRun, result: CreatedRun) -> None:
        """Check returned episode count and selected source/task boundaries before retaining IDs or exposing public prompts."""
        if len(result.episodes) != body.count:
            raise ValueError("gym returned an incomplete requested batch")
        groups = {}
        for episode in result.episodes:
            if episode.task_type not in body.task_types or (gym_id != "general" and episode.gym_id != gym_id):
                raise ValueError("gym returned an unrequested task")
            if body.split == "train":
                group = episode.index // body.group_size
                if episode.seed != body.seed_base + group:
                    raise ValueError("gym returned a task outside its requested group seed")
                if gym_id != "general" and episode.task_type != body.task_types[group]:
                    raise ValueError("gym returned a task outside its pre-drawn group assignment")
                identity = (episode.gym_id, episode.task_type, episode.seed, episode.prompt, episode.system, episode.tools)
                if groups.setdefault(group, identity) != identity:
                    raise ValueError("gym returned different questions within one policy-baseline group")
