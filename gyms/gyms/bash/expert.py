"""The trajectory a model *should* have produced, written from the answer.

Behaviour cloning's whole premise, and the reason it is cheap here: the task
generator already holds the ground truth, so a correct trajectory is a template
rather than a dataset anybody has to collect. The gym can demonstrate itself.

**What this teaches is not how to use a shell.** The first measured runs showed
a model that drove the shell competently enough to put the answer on screen in 4
of 20 episodes and then said "let me run the script again" — every one of its
twenty final replies was search narration rather than an answer. Nothing it had
ever been trained on ended with a bare answer. So the shape of what follows
matters more than its content: **the last turn is the answer alone**.

SFT trains *every* assistant turn, not only that one — so each `Move` below is a
target too, and the command it carries is trained because the backend stores it
on the assistant turn that makes it rather than on the tool turn that answers
it (`crates/api/src/demonstrations.rs`). Which turn a call is stored on is the
difference between these demonstrations teaching a model to use a shell and
teaching it to talk about using one.

Two rules keep these honest as training data:

- **The commands must be ones the model could have written.** `grep -rl` over
  the directory the shell opens in, not a `cat` of the path the generator
  happens to know. A demonstration that uses knowledge the model cannot have at
  that point teaches a step it can never reproduce.
- **Nothing here renders an observation.** What the shell said comes back
  through the sandbox and is rendered by `tools::observation` in Rust, the same
  function serving uses — see
  `docs/decisions/gyms-are-driven-by-the-backend.md`. A Python copy of that
  layout would train the model for a shell it is never shown, which is the one
  failure this module exists to avoid repeating.
"""

from dataclasses import dataclass

from gyms.bash.tasks import Task


@dataclass(frozen=True)
class Move:
    """One demonstrated round: what the model says, and what it runs.

    `reply` is the assistant text the command is wrapped in. It is deliberately
    terse and deliberately *not* narration of the kind the model already
    over-produces — a demonstration that reads "let me check the directory"
    teaches the habit being trained out.
    """

    reply: str
    command: str


@dataclass(frozen=True)
class Trajectory:
    """A whole demonstrated episode, ending on the answer alone."""

    task: Task
    moves: tuple[Move, ...]
    answer: str

    @property
    def commands(self) -> tuple[str, ...]:
        return tuple(move.command for move in self.moves)


def solve(task: Task) -> Trajectory:
    """The expert trajectory for `task`.

    Dispatches on the task type rather than living in the registry beside
    `generate` and `grade`, because a demonstration is not part of what a gym
    *measures* — a task type without one is still a task type, it simply cannot
    be cloned from.
    """
    build = _MOVES.get(task.task_type)
    if build is None:
        raise ValueError(f"no expert trajectory for {task.task_type!r}")
    return Trajectory(task=task, moves=build(task), answer=task.answer)


def _grep_moves(task: Task) -> tuple[Move, ...]:
    """Look around, then search recursively — the needle may be in a
    subdirectory.

    `grep -rn` and deliberately not `grep -rl`, which was the first version of
    this. `-l` prints the filename alone, so the *needle never reaches the
    transcript*: the demonstration would answer with a path the model was shown
    but on evidence it was not, and `SAW_TARGET` — which reads the output —
    would not fire on the one trajectory that is meant to be exemplary. The
    shaping and the demonstration have to agree about what progress looks like.
    `tests/test_expert.py` is what caught it.
    """
    return (
        Move("I need to find which file holds it.", "ls -la"),
        Move("Searching every file for the marker.", f"grep -rn {task.target} ."),
    )


def _csv_moves(task: Task) -> tuple[Move, ...]:
    """Read the file the question named. No search: the prompt already says
    where to look, and demonstrating a search would teach a step the task does
    not contain."""
    name = task.extra.get("file", "data.csv")
    row = task.extra.get("row_id", "")
    return (
        Move("Reading the file the question names.", f"cat {name}"),
        Move("Taking the row it asked for.", f"grep '^{row},' {name}"),
    )


def _code_moves(task: Task) -> tuple[Move, ...]:
    """Grep for the marker rather than reading the whole file: it is the step
    that scales past a file that does not fit the observation budget."""
    name = task.extra.get("file", "code.py")
    mark = task.extra.get("mark", "TARGET")
    return (
        Move("Finding the marked line.", f"grep -n {mark} {name}"),
        Move("Reading the function it belongs to.", f"grep -B2 {mark} {name}"),
    )


def _ls_moves(task: Task) -> tuple[Move, ...]:
    """List, then list again with the extension as a filter.

    The second move looks redundant — the answer is already in the first move's
    output — and it is there for the shaping. `ON_TARGET` is charged against the
    *command*, and `ls -la` names nothing the task asked about, so a
    one-move demonstration would be a trajectory the reward function pays less
    for than a wandering one. The demonstration and the shaping have to agree
    about what progress is; that disagreement is the bug `_grep_moves` already
    had once.
    """
    return (
        Move("Listing the workspace.", "ls -la"),
        Move("Narrowing to the extension the question names.", f"ls *{task.target}"),
    )


def _ls_nested_moves(task: Task) -> tuple[Move, ...]:
    """The same, recursively: the file is in a subdirectory, so `ls` alone
    cannot see it and `find` is the move that scales."""
    return (
        Move("Listing the workspace.", "ls -la"),
        Move("Searching every directory for it.", f"find . -name '*{task.target}'"),
    )


def _ls_mark_moves(task: Task) -> tuple[Move, ...]:
    """Filter on the *name* rather than the contents — the whole point of this
    level, and the reason `-name` takes the mark wrapped in two globs."""
    return (
        Move("Listing the workspace.", "ls -la"),
        Move("Matching the marker against filenames.", f"find . -name '*{task.target}*'"),
    )


def _read_moves(task: Task) -> tuple[Move, ...]:
    """One command. The prompt names the file, the file holds one token, and
    anything before the `cat` would be teaching a search the task does not
    contain — the same reasoning `_csv_moves` gives for not listing first."""
    return (Move("Reading the file the question names.", f"cat {task.target}"),)


def _pairs_moves(task: Task) -> tuple[Move, ...]:
    """Read it, then take the line the question asked for. The second move is
    the one that generalises to a file too long to read whole, which is exactly
    what this family's hard tier is."""
    name = task.extra.get("file", "config.txt")
    key = task.extra.get("key", "")
    return (
        Move("Reading the file the question names.", f"cat {name}"),
        Move("Taking the key it asked for.", f"grep '^{key} ' {name}"),
    )


#: Task type to the moves that solve it. A table rather than a chain of `elif`
#: because it is now ten entries, and because an unmapped type has to raise
#: rather than fall through to a trajectory for a different task.
_MOVES = {
    "bash_grep": _grep_moves,
    "bash_find_easy": _grep_moves,
    "bash_find_medium": _grep_moves,
    "bash_csv": _csv_moves,
    "bash_code": _code_moves,
    "bash_ls_easy": _ls_moves,
    "bash_ls_medium": _ls_nested_moves,
    "bash_ls_hard": _ls_mark_moves,
    "bash_read_easy": _read_moves,
    "bash_read_medium": _pairs_moves,
}
