"""Provider boundaries. Public task fields are an allowlist, never private state."""

from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class Generation:
    seed: int
    split: str = "train"
    version: int = 1


@dataclass(frozen=True)
class Task:
    gym_id: str
    task_type: str
    seed: int
    generator_version: int
    prompt: str
    system: str
    tools: tuple[str, ...]
    split: str | None
    problem_key: str | None
    private: dict

    def public(self) -> dict:
        return {
            "gym_id": self.gym_id, "task_type": self.task_type,
            "seed": self.seed, "generator_version": self.generator_version,
            "prompt": self.prompt, "system": self.system,
            "tools": list(self.tools), "split": self.split,
        }

    def stored(self) -> dict:
        # Keep the historical flat task_config readable by existing records.
        return {**self.private, **self.public(), "problem_key": self.problem_key}

    @classmethod
    def restore(cls, gym_id: str, config: dict, tools: tuple[str, ...]) -> "Task":
        public_keys = {"gym_id", "task_type", "seed", "generator_version",
                       "prompt", "system", "tools", "split", "problem_key"}
        return cls(
            gym_id, config["task_type"], config["seed"],
            config.get("generator_version", 1), config["prompt"], config["system"],
            tuple(config.get("tools", tools)), config.get("split"),
            config.get("problem_key"),
            {key: value for key, value in config.items() if key not in public_keys},
        )


@dataclass(frozen=True)
class Observation:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool
    reward: float


@dataclass(frozen=True)
class TaskType:
    id: str
    label: str
    description: str
    generate: Callable[[Generation], Task]
    level: str = "hard"
    family: str = ""
    prerequisites: tuple[str, ...] = ()


class Provider(Protocol):
    tools: tuple[str, ...]
    splits: tuple[str, ...]
    version: int
    demonstrations: bool

    def prepare(self, session: str, task: Task) -> None: ...

    def step(self, session: str, task: Task, tool: str, argument: str,
             number: int, history: list[tuple[str, str]]) -> Observation: ...

    def finish(self, session: str, task: Task, reply: str) -> tuple[float, str]: ...

    def demonstrate(self, session: str, task: Task) -> dict: ...


class UnsupportedAction(ValueError):
    pass
