"""The demonstrated trajectory, run against the shell that will replay it.

A demonstration is training data, so the bar is higher than "it looks right":
every command has to actually run, the evidence has to actually appear in what
comes back, and the answer has to be derivable from what the model was shown.
A trajectory whose middle step prints nothing teaches the model to answer from
nowhere — which is the habit it already has.

Against the real sandbox rather than the in-process fake, for the reason
`test_sandbox.py` gives: the fake answers from wherever it is asked, so it
cannot tell a working `grep -rl` from one that matched nothing.
"""

import uuid

import pytest

from gyms.bash import expert, tasks
from gyms.registry import REGISTRY
from gyms.sandbox import WORKSPACE, SandboxError, run, seed

pytestmark = pytest.mark.sandbox

#: Every generator, deliberately not a hand-kept list: a demonstration is
#: training data, and a task type whose expert trajectory was never run against
#: a real shell is a batch of SFT rows teaching a command that does not work.
#: `tests/test_tasks.py` owns the same list for the same reason.
GENERATORS = [
    tasks.generate_ls_easy,
    tasks.generate_ls_medium,
    tasks.generate_ls_hard,
    tasks.generate_read_easy,
    tasks.generate_read_medium,
    tasks.generate_csv,
    tasks.generate_code,
    tasks.generate_find_easy,
    tasks.generate_find_medium,
    tasks.generate_grep,
]


def _reachable() -> bool:
    try:
        run(f"probe-{uuid.uuid4()}", "true", 2000)
        return True
    except SandboxError:
        return False


@pytest.fixture
def session() -> str:
    if not _reachable():
        pytest.skip("no just-bash at BASH_URL")
    return str(uuid.uuid4())


def _play(session: str, task: tasks.Task) -> list:
    seed(session, WORKSPACE, task.files)
    return [run(session, move.command) for move in expert.solve(task).moves]


@pytest.mark.parametrize("generate", GENERATORS)
def test_every_demonstrated_command_runs(generate, session):
    for ran in _play(session, generate(21)):
        assert ran.exit_code == 0, f"demonstration exits {ran.exit_code}"
        assert not ran.timed_out


@pytest.mark.parametrize("generate", GENERATORS)
def test_the_demonstration_puts_the_evidence_on_screen(generate, session):
    """`SAW_TARGET` has to fire on a trajectory that is meant to be exemplary —
    otherwise the shaping and the demonstration disagree about what progress is.
    """
    task = generate(22)
    printed = "\n".join(ran.stdout for ran in _play(session, task))
    assert task.evidence in printed


@pytest.mark.parametrize("generate", GENERATORS)
def test_the_answer_is_visible_in_what_the_model_was_shown(generate, session):
    """The step that matters: a demonstration whose final answer cannot be read
    off its own transcript is teaching the model to guess."""
    task = generate(23)
    printed = "\n".join(ran.stdout for ran in _play(session, task))
    assert task.answer in printed


@pytest.mark.parametrize("generate", GENERATORS)
def test_the_demonstrated_answer_grades_correct(generate, session):
    """The end of it: the trajectory this module writes must score 1 under the
    grader every model is scored by. Anything else is a demonstration of how to
    fail."""
    task = generate(24)
    _play(session, task)
    score, _ = REGISTRY.grade(task.task_type, expert.solve(task).answer, task.answer)
    assert score == 1.0
