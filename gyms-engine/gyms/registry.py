"""Which gyms exist, and what kinds of task each one generates.

One table, read by three callers that must not disagree: the route that creates
a run looks a task type up here, the route that lists gyms renders it, and the
frontend's catalogue is now this listing rather than a constant of its own.

Source gyms own task types and a runtime provider. Composite gyms reference
those types without taking ownership. No registry path implements grading or
environment behavior; those belong to each source provider.
"""

from dataclasses import dataclass
from functools import partial

from gyms.bash import composed, tasks
from gyms.bash import runtime as bash_runtime
from gyms.math import runtime as math_runtime
from gyms.contracts import Generation, Provider, Task, TaskType
from gyms.math import tasks as arithmetic
from gyms.state_tracking import tasks as state_tasks
from gyms.constraint_logic import tasks as logic_tasks
from gyms.text_runtime import Provider as TextProvider


#: The rungs of a family's ladder, in the order they should be attempted.
#:
#: A level is a **shaping** claim, not a taxonomy: each rung requires exactly one
#: shell skill more than the one below it, so a run across a whole family says
#: where the model stops rather than only that it stopped. That matters because
#: the first ten runs of this gym scored 0.00 over 105 episodes with no way to
#: tell a model that cannot search from one that can search and cannot read.
LEVELS = ("easy", "medium", "hard")


@dataclass(frozen=True)
class Gym:
    id: str
    name: str
    description: str
    task_types: tuple[TaskType, ...]
    provider: Provider | None

    @property
    def tools(self) -> tuple[str, ...]:
        return self.provider.tools if self.provider else ()


BASH = Gym(
    id="bash",
    provider=bash_runtime.Provider(),
    name="just-bash",
    description=(
        "A shell over a virtual filesystem. The model is given a workspace and a "
        "question it can only answer by looking around."
    ),
    task_types=(
        # ---- ls: the answer is in the listing. `ls` -> recurse -> filter. ----
        TaskType(
            id="bash_ls_easy",
            label="Name the odd file",
            description="A flat handful of files, one with a unique extension. `ls` solves it.",
            generate=bash_runtime.generator(tasks.generate_ls_easy),
            level="easy",
            family="ls",
        ),
        TaskType(
            id="bash_ls_medium",
            label="Name the odd file, nested",
            description="The same question with the files in subdirectories: the listing has to recurse.",
            generate=bash_runtime.generator(tasks.generate_ls_medium),
            level="medium",
            family="ls",
        ),
        TaskType(
            id="bash_ls_hard",
            label="Find the marked filename",
            description="A marker in a filename among forty files; the path is the answer.",
            generate=bash_runtime.generator(tasks.generate_ls_hard),
            level="hard",
            family="ls",
        ),
        # ---- read: the answer is inside a file the prompt already names. ----
        TaskType(
            id="bash_read_easy",
            label="Read the token",
            description="One named file holding one token. A single `cat` is the whole task.",
            generate=bash_runtime.generator(tasks.generate_read_easy),
            level="easy",
            family="read",
        ),
        TaskType(
            id="bash_read_medium",
            label="Read the setting",
            description="A named file of `key = value` lines; the value of the stated key is the answer.",
            generate=bash_runtime.generator(tasks.generate_read_medium),
            level="medium",
            family="read",
        ),
        TaskType(
            id="bash_csv",
            label="Read the cell",
            description="A stated row of a CSV; the cell's value is the answer.",
            generate=bash_runtime.generator(tasks.generate_csv),
            level="hard",
            family="read",
        ),
        TaskType(
            id="bash_code",
            label="Name the function",
            description="One marked function among decoys; its name is the answer.",
            generate=bash_runtime.generator(tasks.generate_code),
            level="hard",
            family="read",
        ),
        # ---- find: which file holds the needle. Contents, not names. ----
        TaskType(
            id="bash_find_easy",
            label="Search three files",
            description="One needle across three or four flat files; the filename is the answer.",
            generate=bash_runtime.generator(tasks.generate_find_easy),
            level="easy",
            family="find",
        ),
        TaskType(
            id="bash_find_medium",
            label="Search a dozen files",
            description="Enough files that reading them all costs more than searching them.",
            generate=bash_runtime.generator(tasks.generate_find_medium),
            level="medium",
            family="find",
        ),
        TaskType(
            id="bash_grep",
            label="Find the file",
            description="One needle across many files and subdirectories; the path is the answer.",
            generate=bash_runtime.generator(tasks.generate_grep),
            level="hard",
            family="find",
        ),
        # ---- composition: commands are chained inside one shell call. ----
        TaskType(
            id="bash_write_easy",
            label="Write a file",
            description="Create a named file with one redirection primitive.",
            generate=bash_runtime.generator(composed.generate_write_easy),
            level="easy",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_find_medium",
            label="Pipe a file search",
            description="Find and filter paths with exactly one pipe.",
            generate=bash_runtime.generator(composed.generate_pipe_find_medium),
            level="medium",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_extract_medium",
            label="Pipe an extraction",
            description="Read, select and extract a value with exactly one pipe.",
            generate=bash_runtime.generator(composed.generate_pipe_extract_medium),
            level="medium",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_write_medium",
            label="Pipe into a file",
            description="Filter a record and write it with exactly one pipe.",
            generate=bash_runtime.generator(composed.generate_pipe_write_medium),
            level="medium",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_search_hard",
            label="Reduce search results",
            description="Search, extract paths and deduplicate with three commands.",
            generate=bash_runtime.generator(composed.generate_pipe_search_hard),
            level="hard",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_transform_hard",
            label="Transform into a file",
            description="Read, filter, extract and write with three pipeline stages.",
            generate=bash_runtime.generator(composed.generate_pipe_transform_hard),
            level="hard",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_discover_edit_hard",
            label="Discover and edit",
            description="Find a file, extract a line and write it through a pipeline.",
            generate=bash_runtime.generator(composed.generate_pipe_discover_edit_hard),
            level="hard",
            family="compose",
        ),
    ),
)

MATH = Gym(
    id="math",
    name="Math",
    description="Arithmetic, expressions and word problems. Exact grading, no tools, automatic skill progression.",
    provider=math_runtime.Provider(),
    task_types=tuple(
        TaskType(
            id=f"math_{family}_{level}",
            label=f"{family.title()} · {level}",
            description=f"{family.title()} at the {level} rung; one exact numeric answer.",
            generate=partial(math_runtime.generate, family, level),
            level=level,
            family=family,
            prerequisites=arithmetic.prerequisites(family, level),
        )
        for family in arithmetic.FAMILIES for level in LEVELS
    ),
)

STATE_TRACKING = Gym(
    id="state_tracking",
    name="State tracking",
    description="Follow transfers of named objects through short event histories; exact final holder, no tools.",
    provider=TextProvider(),
    task_types=tuple(
        TaskType(
            id=f"state_tracking_{level}",
            label=f"Object transfers · {level}",
            description=f"Track an object's final holder through a {level} event sequence.",
            generate=partial(state_tasks.generate, level),
            level=level,
            family="objects",
            prerequisites=(f"state_tracking_{LEVELS[index - 1]}",) if index else (),
        )
        for index, level in enumerate(LEVELS)
    ),
)

CONSTRAINT_LOGIC = Gym(
    id="constraint_logic",
    name="Constraint logic",
    description="Solve uniquely determined line-order puzzles from generated clues; exact answer, no tools.",
    provider=TextProvider(),
    task_types=tuple(
        TaskType(
            id=f"constraint_logic_{level}",
            label=f"Line order · {level}",
            description=f"Find a person in a uniquely determined order of {logic_tasks.COUNT[level]}.",
            generate=partial(logic_tasks.generate, level),
            level=level,
            family="order",
            prerequisites=(f"constraint_logic_{LEVELS[index - 1]}",) if index else (),
        )
        for index, level in enumerate(LEVELS)
    ),
)

GENERAL = Gym(
    id="general",
    provider=None,
    name="General",
    description="Questions from every source gym, mixed across episodes with each gym's tools and grading.",
    task_types=(),
)

GYMS: tuple[Gym, ...] = (BASH, MATH, STATE_TRACKING, CONSTRAINT_LOGIC, GENERAL)


class GymRegistry:
    """The lookup the routes use. Built once, at import."""

    def __init__(self, gyms: tuple[Gym, ...]) -> None:
        if len({gym.id for gym in gyms}) != len(gyms):
            raise ValueError("duplicate gym id")
        if any(gym.provider is None and gym.task_types for gym in gyms):
            raise ValueError("a composite gym references source tasks; it cannot own task types")
        task_ids = [task.id for gym in gyms for task in gym.task_types]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("duplicate task type id")
        self._gyms = {gym.id: gym for gym in gyms}
        self._task_types = {
            task_type.id: (gym.id, task_type)
            for gym in gyms
            for task_type in gym.task_types
        }

    def gym(self, gym_id: str) -> Gym | None:
        return self._gyms.get(gym_id)

    def task_type(self, task_type_id: str) -> TaskType | None:
        found = self._task_types.get(task_type_id)
        return found[1] if found else None

    def gym_of(self, task_type_id: str) -> str | None:
        found = self._task_types.get(task_type_id)
        return found[0] if found else None

    def provider(self, gym_id: str) -> Provider:
        gym = self.gym(gym_id)
        if gym is None or gym.provider is None:
            raise ValueError(f"{gym_id!r} is not a source gym")
        return gym.provider

    def available(self, gym_id: str) -> tuple[TaskType, ...]:
        gym = self.gym(gym_id)
        if gym is None:
            raise ValueError(f"unknown gym {gym_id!r}")
        if gym.provider is not None:
            return gym.task_types
        return tuple(t for source in self._gyms.values() if source.provider
                     for t in source.task_types)

    def generate(self, task_type_id: str, seed: int, split: str = "train",
                 version: int | None = None) -> Task:
        task_type = self.task_type(task_type_id)
        if task_type is None:
            raise ValueError(f"unknown task type {task_type_id!r}")
        provider = self.provider(self.gym_of(task_type_id))
        task = task_type.generate(Generation(seed, split, provider.version if version is None else version))
        if task.gym_id != self.gym_of(task_type_id) or task.task_type != task_type_id:
            raise ValueError("generated task does not match its registered owner and type")
        return task

    def listing(self) -> list[dict]:
        return [
            {
                "id": gym.id,
                "name": gym.name,
                "description": gym.description,
                "tools": list(gym.tools),
                "generator_version": gym.provider.version if gym.provider else None,
                "composite": gym.provider is None,
                "splits": list(gym.provider.splits) if gym.provider else [],
                "demonstrations": gym.provider.demonstrations if gym.provider else all(
                    self.provider(self.gym_of(t.id)).demonstrations for t in self.available(gym.id)
                ),
                "task_types": [
                    {
                        "id": t.id,
                        "gym_id": self.gym_of(t.id),
                        "label": t.label,
                        "description": t.description,
                        "level": t.level,
                        "family": t.family,
                        "prerequisites": list(t.prerequisites),
                    }
                    for t in self.available(gym.id)
                ],
            }
            for gym in self._gyms.values()
        ]


REGISTRY = GymRegistry(GYMS)
