"""The existing rollout/curriculum gym seam, implemented by the legacy parent hop or worker effect adapter."""

from typing import Protocol

from llm.gym.client import Episode, Graded, Observed


class GymClient(Protocol):
    """Typed task/action operations without exposing transport, actor credentials or resource-lifecycle ownership to numerical algorithms."""

    def catalogue(self) -> list[dict]:
        """Return authorized public catalogue metadata for curriculum selection."""
        ...

    def create_run(self, *, gym_id: str, model_name: str, task_types: list[str], count: int,
                   seed_base: int, group_size: int, split: str = "train",
                   gym_weights: dict[str, float] | None = None,
                   task_weights: dict[str, float] | None = None) -> tuple[str, list[Episode]]:
        """Create fresh bounded iteration/exam resources and return only their public task fields."""
        ...

    def step(self, episode_id: str, tool: str, argument: str, reply: str) -> Observed:
        """Issue one tool action and return the backend-rendered observation without retrying uncertain writes."""
        ...

    def finish(self, episode_id: str, reply: str) -> Graded:
        """Grade a final reply once; the answer becomes visible only after the episode is finished."""
        ...

    def abandon(self, episode_id: str) -> None:
        """Close an unavailable episode without fabricating a grade or policy target."""
        ...

    def set_status(self, run_id: str, status: str) -> None:
        """Record the iteration/exam's actual lifecycle through its owning transport."""
        ...
