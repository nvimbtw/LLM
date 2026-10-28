"""Bounded interpreted gym effect payloads; service-owned sibling response fields remain outside the worker's view."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from llm.architecture.spec import Contract, NonnegativeInt, PositiveInt
from llm.gym.client import Episode, Graded, Observed

Identifier = Annotated[str, Field(min_length=1, max_length=128)]
ReplyText = Annotated[str, Field(max_length=1 << 20)]
EpisodeCount = Annotated[PositiveInt, Field(le=256)]
Reward = Annotated[float, Field(strict=True, allow_inf_nan=False)]
Weights = Annotated[dict[Identifier, Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]], Field(max_length=4096)]


class CreateRun(Contract):
    """Exact existing Rust gym-create inputs, without caller credentials, URL or arbitrary method/path."""

    model_name: UUID
    task_types: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=4096)]
    count: EpisodeCount
    seed_base: Annotated[NonnegativeInt, Field(le=2**64 - 1)]
    group_size: EpisodeCount
    split: Literal["train", "validation", "test"] = "train"
    gym_weights: Weights | None = None
    task_weights: Weights | None = None

    @model_validator(mode="after")
    def finite_groups(self):
        """Require complete bounded groups before external issuance; typed fields separately bound finite weight maps."""
        if self.count % self.group_size:
            raise ValueError("gym create requires complete groups")
        return self


class Step(Contract):
    """One model-issued tool command and its sampled reply, preserving the existing backend layout."""

    tool: Identifier
    argument: ReplyText
    reply: ReplyText


class Finish(Contract):
    """One final sampled reply to grade; answers are not part of any input effect."""

    reply: ReplyText


class Status(Contract):
    """The existing finite run-status transition requested by the training driver."""

    status: Literal["running", "done", "failed", "abandoned"]


class ServiceRecord(BaseModel):
    """Strict interpreted fields with deliberate omission of unconsumed service-owned response siblings."""

    model_config = ConfigDict(extra="ignore", frozen=True, strict=True, allow_inf_nan=False)


class PublicEpisode(ServiceRecord):
    """An authorized public task and its resource identity; no grading answer enters the worker before finish."""

    id: UUID
    index: NonnegativeInt
    gym_id: Identifier
    task_type: Identifier
    seed: NonnegativeInt
    prompt: ReplyText = ""
    system: ReplyText = ""
    tools: Annotated[tuple[Identifier, ...], Field(max_length=64)] = ("bash",)

    def episode(self) -> Episode:
        """Translate validated public task values to the existing rollout domain without coercing malformed remote values."""
        return Episode(str(self.id), self.index, self.gym_id, self.task_type, self.seed, self.prompt, self.system, self.tools)


class CreatedRun(ServiceRecord):
    """Created resource identities and bounded public tasks returned to one owning iteration."""

    id: UUID
    episodes: Annotated[tuple[PublicEpisode, ...], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def unique_episodes(self):
        """Refuse duplicate identities or missing/duplicated ordering indices before they can alias action provenance."""
        if (len({episode.id for episode in self.episodes}) != len(self.episodes)
                or {episode.index for episode in self.episodes} != set(range(len(self.episodes)))):
            raise ValueError("gym create returned inconsistent episode identity/order")
        return self


class StepResult(ServiceRecord):
    """Rendered backend observation and finite shaping reward after one issued tool step."""

    observation: ReplyText = ""
    reward: Reward = 0.0
    step_num: NonnegativeInt = 0

    def observed(self) -> Observed:
        """Translate a validated step result without re-rendering backend-owned observations."""
        return Observed(self.observation, self.reward, self.step_num)


class FinishResult(ServiceRecord):
    """Finite graded outcome, available only after finishing the episode."""

    score: Reward = 0.0
    answer: ReplyText = ""

    def graded(self) -> Graded:
        """Translate the actual grade to the existing policy return domain."""
        return Graded(self.score, self.answer)


class Catalogue(Contract):
    """Bounded catalogue records passed through to the existing curriculum parser, never executable commands."""

    gyms: Annotated[tuple[dict[str, JsonValue], ...], Field(max_length=128)]


class EmptyResult(Contract):
    """Explicit acknowledgement for a completed status/abandon effect with no domain result."""


class ClosedEpisode(ServiceRecord):
    """Parent-only interpreted close acknowledgement; a previously graded episode remains graded."""

    episode_id: UUID
    status: Literal["closed", "graded"]


class UpdatedRun(ServiceRecord):
    """Parent-only run identity and status acknowledgement, excluding unrelated service history."""

    id: UUID
    status: Literal["running", "done", "failed", "abandoned"]
