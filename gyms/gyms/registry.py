"""Which gyms exist, and what kinds of task each one generates.

One table, read by three callers that must not disagree: the route that creates
a run looks a task type up here, the route that lists gyms renders it, and the
frontend's catalogue is now this listing rather than a constant of its own.

A gym is a name and a set of task types. That is all it is — the shell it runs
in, the reward shaping and the grading belong to the task's own module, because
a second gym will share none of them.
"""

from dataclasses import dataclass
from functools import partial
from typing import Callable

from gyms.bash import composed, tasks
from gyms.bash.grade import grade
from gyms.bash.tasks import Task
from gyms.math import tasks as arithmetic
from gyms.math.grade import grade as grade_math


#: The rungs of a family's ladder, in the order they should be attempted.
#:
#: A level is a **shaping** claim, not a taxonomy: each rung requires exactly one
#: shell skill more than the one below it, so a run across a whole family says
#: where the model stops rather than only that it stopped. That matters because
#: the first ten runs of this gym scored 0.00 over 105 episodes with no way to
#: tell a model that cannot search from one that can search and cannot read.
LEVELS = ("easy", "medium", "hard")


@dataclass(frozen=True)
class TaskType:
    """One kind of task: how to make one, and how to judge an answer to it.

    `grade` returns `(score, extracted_answer)` — the answer is carried out
    because the trajectory view shows what the model said, not only whether it
    was right.
    """

    id: str
    label: str
    description: str
    generate: Callable[[int], Task | arithmetic.Problem]
    grade: Callable[[str, str], tuple[float, str]]
    #: Which rung of `LEVELS` this is, and which family it is a rung of. Served
    #: in the listing so the run form can offer "every easy task" without
    #: knowing any task type by name — the property that keeps this table the
    #: only place a task type is spelled out.
    level: str = "hard"
    family: str = ""
    prerequisites: tuple[str, ...] = ()


@dataclass(frozen=True)
class Gym:
    id: str
    name: str
    description: str
    task_types: tuple[TaskType, ...]
    tools: tuple[str, ...] = ("bash",)


BASH = Gym(
    id="bash",
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
            generate=tasks.generate_ls_easy,
            grade=grade,
            level="easy",
            family="ls",
        ),
        TaskType(
            id="bash_ls_medium",
            label="Name the odd file, nested",
            description="The same question with the files in subdirectories: the listing has to recurse.",
            generate=tasks.generate_ls_medium,
            grade=grade,
            level="medium",
            family="ls",
        ),
        TaskType(
            id="bash_ls_hard",
            label="Find the marked filename",
            description="A marker in a filename among forty files; the path is the answer.",
            generate=tasks.generate_ls_hard,
            grade=grade,
            level="hard",
            family="ls",
        ),
        # ---- read: the answer is inside a file the prompt already names. ----
        TaskType(
            id="bash_read_easy",
            label="Read the token",
            description="One named file holding one token. A single `cat` is the whole task.",
            generate=tasks.generate_read_easy,
            grade=grade,
            level="easy",
            family="read",
        ),
        TaskType(
            id="bash_read_medium",
            label="Read the setting",
            description="A named file of `key = value` lines; the value of the stated key is the answer.",
            generate=tasks.generate_read_medium,
            grade=grade,
            level="medium",
            family="read",
        ),
        TaskType(
            id="bash_csv",
            label="Read the cell",
            description="A stated row of a CSV; the cell's value is the answer.",
            generate=tasks.generate_csv,
            grade=grade,
            level="hard",
            family="read",
        ),
        TaskType(
            id="bash_code",
            label="Name the function",
            description="One marked function among decoys; its name is the answer.",
            generate=tasks.generate_code,
            grade=grade,
            level="hard",
            family="read",
        ),
        # ---- find: which file holds the needle. Contents, not names. ----
        TaskType(
            id="bash_find_easy",
            label="Search three files",
            description="One needle across three or four flat files; the filename is the answer.",
            generate=tasks.generate_find_easy,
            grade=grade,
            level="easy",
            family="find",
        ),
        TaskType(
            id="bash_find_medium",
            label="Search a dozen files",
            description="Enough files that reading them all costs more than searching them.",
            generate=tasks.generate_find_medium,
            grade=grade,
            level="medium",
            family="find",
        ),
        TaskType(
            id="bash_grep",
            label="Find the file",
            description="One needle across many files and subdirectories; the path is the answer.",
            generate=tasks.generate_grep,
            grade=grade,
            level="hard",
            family="find",
        ),
        # ---- composition: commands are chained inside one shell call. ----
        TaskType(
            id="bash_write_easy",
            label="Write a file",
            description="Create a named file with one redirection primitive.",
            generate=composed.generate_write_easy,
            grade=grade,
            level="easy",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_find_medium",
            label="Pipe a file search",
            description="Find and filter paths with exactly one pipe.",
            generate=composed.generate_pipe_find_medium,
            grade=grade,
            level="medium",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_extract_medium",
            label="Pipe an extraction",
            description="Read, select and extract a value with exactly one pipe.",
            generate=composed.generate_pipe_extract_medium,
            grade=grade,
            level="medium",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_write_medium",
            label="Pipe into a file",
            description="Filter a record and write it with exactly one pipe.",
            generate=composed.generate_pipe_write_medium,
            grade=grade,
            level="medium",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_search_hard",
            label="Reduce search results",
            description="Search, extract paths and deduplicate with three commands.",
            generate=composed.generate_pipe_search_hard,
            grade=grade,
            level="hard",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_transform_hard",
            label="Transform into a file",
            description="Read, filter, extract and write with three pipeline stages.",
            generate=composed.generate_pipe_transform_hard,
            grade=grade,
            level="hard",
            family="compose",
        ),
        TaskType(
            id="bash_pipe_discover_edit_hard",
            label="Discover and edit",
            description="Find a file, extract a line and write it through a pipeline.",
            generate=composed.generate_pipe_discover_edit_hard,
            grade=grade,
            level="hard",
            family="compose",
        ),
    ),
)

MATH = Gym(
    id="math",
    name="Math",
    description="Arithmetic, expressions and word problems. Exact grading, no tools, automatic skill progression.",
    tools=(),
    task_types=tuple(
        TaskType(
            id=f"math_{family}_{level}",
            label=f"{family.title()} · {level}",
            description=f"{family.title()} at the {level} rung; one exact numeric answer.",
            generate=partial(arithmetic.generate, family, level),
            grade=grade_math,
            level=level,
            family=family,
            prerequisites=arithmetic.prerequisites(family, level),
        )
        for family in arithmetic.FAMILIES for level in LEVELS
    ),
)

GYMS: tuple[Gym, ...] = (BASH, MATH)


class GymRegistry:
    """The lookup the routes use. Built once, at import."""

    def __init__(self, gyms: tuple[Gym, ...]) -> None:
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

    def generate(self, task_type_id: str, seed: int, split: str = "train") -> Task | arithmetic.Problem:
        task_type = self.task_type(task_type_id)
        if task_type is None:
            raise ValueError(f"unknown task type {task_type_id!r}")
        if self.gym_of(task_type_id) == "math":
            return arithmetic.generate(task_type.family, task_type.level, seed, split)
        return task_type.generate(seed)

    def grade(self, task_type_id: str, reply: str, ground_truth: str) -> tuple[float, str]:
        """An unknown task type is an episode nothing can score, and saying so
        is the only honest answer — a default of 0.0 would report a model as
        wrong for a bug on this side."""
        task_type = self.task_type(task_type_id)
        if task_type is None:
            raise ValueError(f"unknown task type {task_type_id!r}")
        return task_type.grade(reply, ground_truth)

    def listing(self) -> list[dict]:
        return [
            {
                "id": gym.id,
                "name": gym.name,
                "description": gym.description,
                "tools": list(gym.tools),
                "generator_version": arithmetic.VERSION if gym.id == "math" else None,
                "task_types": [
                    {
                        "id": t.id,
                        "label": t.label,
                        "description": t.description,
                        "level": t.level,
                        "family": t.family,
                        "prerequisites": list(t.prerequisites),
                    }
                    for t in gym.task_types
                ],
            }
            for gym in self._gyms.values()
        ]


REGISTRY = GymRegistry(GYMS)
