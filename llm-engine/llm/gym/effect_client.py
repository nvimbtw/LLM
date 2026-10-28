"""Worker-only gym adapter over serialized parent effects, with no service URL, actor, credentials or retry owner."""

from collections.abc import Callable
from threading import Lock
from typing import TypeVar
from uuid import UUID

from pydantic import BaseModel

from llm.architecture.package_integrity import decode_json, encode_json
from llm.architecture.spec import Contract
from llm.gym.client import Episode, Graded, Observed
from llm.gym.effect_records import Catalogue, CreatedRun, CreateRun, EmptyResult, Finish, FinishResult, Status, Step, StepResult
from llm.runtime.protocol import (
    EffectData, EffectOperation, GymAbandonEffect, GymCatalogueEffect, GymCreateEffect,
    GymFinishEffect, GymStatusEffect, GymStepEffect, require_format,
)
from llm.runtime.worker import WorkerCancelled, WorkerContext

Response = TypeVar("Response", bound=BaseModel)


class EffectClient:
    """Serialize one worker's gym operations and latch failure so a partially completed iteration cannot issue further effects."""

    def __init__(self, context: WorkerContext):
        """Retain only the admitted worker channel/grant owner and a finite outstanding-effect lock."""
        self.context = context
        self._lock = Lock()
        self._stopped = False

    def exchange(self, build: Callable[[EffectData | None], EffectOperation], body: Contract | None,
                 response: type[Response]) -> Response:
        """Own payload/result grants through one exchange, removing them after consumption and refusing any post-failure operation.

        Legacy rollout threads may call this adapter concurrently, while the
        runtime permits one outstanding effect. Failure or cancellation latches
        the whole client; no write is retried or converted to an invented grade.
        """
        with self._lock:
            if self._stopped:
                raise WorkerCancelled("gym effect client is stopped")
            payload = result = None
            try:
                self.context.check_cancelled()
                if body is not None:
                    grant = self.context.artifacts.write_bytes(encode_json(body), "effect_data_v1")
                    payload = EffectData.model_validate(grant.model_dump())
                reply = self.context.effect(build(payload))
                if reply.outcome != "completed" or reply.result is None:
                    raise RuntimeError("gym effect did not return a completed typed result")
                result = reply.result
                require_format(result, "effect_data_v1")
                raw = self.context.artifacts.read_bytes(result)
                return response.model_validate_json(encode_json(decode_json(raw)))
            except BaseException:
                self._stopped = True
                raise
            finally:
                for grant in (payload, result):
                    if grant is not None:
                        self.context.artifacts.remove(grant)

    def catalogue(self) -> list[dict]:
        """Read bounded catalogue records through the parent, retaining no service handle in the worker."""
        result = self.exchange(lambda _: GymCatalogueEffect(operation="gym_catalogue"), None, Catalogue)
        return list(result.gyms)

    def create_run(self, *, gym_id: str, model_name: str, task_types: list[str], count: int,
                   seed_base: int, group_size: int, split: str = "train",
                   gym_weights: dict[str, float] | None = None,
                   task_weights: dict[str, float] | None = None) -> tuple[str, list[Episode]]:
        """Create one new iteration/exam run through typed inputs and require every bounded public episode before rollout."""
        body = CreateRun(model_name=model_name, task_types=tuple(task_types), count=count, seed_base=seed_base,
            group_size=group_size, split=split, gym_weights=gym_weights, task_weights=task_weights)
        result = self.exchange(lambda data: GymCreateEffect(operation="gym_create", gym_id=gym_id, payload=data), body, CreatedRun)
        if len(result.episodes) != body.count:
            with self._lock:
                self._stopped = True
            raise ValueError("gym effect returned an incomplete iteration/exam")
        return str(result.id), [episode.episode() for episode in result.episodes]

    def step(self, episode_id: str, tool: str, argument: str, reply: str) -> Observed:
        """Issue one explicit tool action; failure/uncertainty stops the iteration without replay."""
        body = Step(tool=tool, argument=argument, reply=reply)
        result = self.exchange(lambda data: GymStepEffect(operation="gym_step", episode_id=UUID(episode_id), payload=data), body, StepResult)
        return result.observed()

    def finish(self, episode_id: str, reply: str) -> Graded:
        """Grade one final reply through the parent and expose its validated finite outcome only afterward."""
        result = self.exchange(lambda data: GymFinishEffect(operation="gym_finish", episode_id=UUID(episode_id), payload=data),
            Finish(reply=reply), FinishResult)
        return result.graded()

    def abandon(self, episode_id: str) -> None:
        """Close one authorized episode without inventing a grade or suppressing an uncertain write."""
        self.exchange(lambda _: GymAbandonEffect(operation="gym_abandon", episode_id=UUID(episode_id)), None, EmptyResult)

    def set_status(self, run_id: str, status: str) -> None:
        """Request one typed run-status transition; an uncertain completion stops further effects."""
        self.exchange(lambda data: GymStatusEffect(operation="gym_status", run_id=UUID(run_id), payload=data),
            Status(status=status), EmptyResult)

    def __enter__(self):
        """Expose this worker's effect owner for one numerical request."""
        if self._stopped:
            raise WorkerCancelled("gym effect client is stopped")
        return self

    def __exit__(self, *args) -> None:
        """Stop new exchanges without closing the worker-owned channel or issuing cleanup writes from the child."""
        with self._lock:
            self._stopped = True
