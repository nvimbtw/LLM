"""Seeding, against the shell that actually runs the commands.

Everything else here stands in for just-bash in-process, which is the right
trade for testing the wire — and is exactly why it could not catch the thing
this file exists for. The fake answers whatever it is asked, from wherever it is
asked; the real sandbox opens every command in `/home/user` and resets the
working directory between them. A task seeded anywhere else is a task the model
has to navigate to, in every command, forever — and the first model to try
simply ran `ls -la`, saw an empty directory, and said so.

Skipped when there is no sandbox to talk to, the same bargain
`test_migrations.py` strikes with Postgres.
"""

import uuid

import pytest

from gyms.bash import tasks
from gyms.sandbox import WORKSPACE, SandboxError, clear, run, seed

pytestmark = pytest.mark.sandbox


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


def test_the_shell_opens_in_the_workspace(session):
    """The premise everything else here rests on."""
    assert run(session, "pwd").stdout.strip() == WORKSPACE


def test_a_bare_ls_finds_a_seeded_task(session):
    """The command the model actually reached for.

    The filename comes off the task rather than being spelled here: a task type
    varies its surface across seeds, so a constant in a test is a constant that
    is only true of one of them.
    """
    task = tasks.generate_csv(3)
    seed(session, WORKSPACE, task.files)
    assert task.extra["file"] in run(session, "ls").stdout


def test_a_relative_path_reaches_a_seeded_file(session):
    task = tasks.generate_code(3)
    seed(session, WORKSPACE, task.files)
    command = f"grep -B2 {task.extra['mark']} {task.extra['file']}"
    assert task.answer in run(session, command).stdout


def test_a_nested_task_file_is_seeded_with_its_directory(session):
    seed(session, WORKSPACE, {"sub/deep/a.txt": "needle"})
    assert run(session, "cat sub/deep/a.txt").stdout.strip() == "needle"


def test_content_that_looks_like_the_heredoc_delimiter_is_still_written(session):
    """The delimiter is drawn per file and checked against the content, so a
    task whose text happens to contain one cannot end its own write early."""
    body = "GYM_EOF_0123456789abcdef\nand a second line"
    seed(session, WORKSPACE, {"tricky.txt": body})
    assert run(session, "cat tricky.txt").stdout.strip() == body


def test_a_working_directory_does_not_survive_the_command_that_set_it(session):
    """Why the workspace is where the shell already is, stated as a test: each
    command gets a fresh shell, so a `cd` buys the next command nothing."""
    run(session, "cd /tmp")
    assert run(session, "pwd").stdout.strip() == WORKSPACE


@pytest.mark.parametrize("escaping", ["/etc/passwd", "../outside.txt", "sub/../../x"])
def test_a_path_that_leaves_the_workspace_is_refused(session, escaping):
    """Filenames are generated here today, and this is what keeps that a fact
    about today rather than something to remember."""
    with pytest.raises(SandboxError):
        seed(session, WORKSPACE, {escaping: "x"})


def test_a_workspace_that_did_not_fill_is_an_error(session, monkeypatch):
    """The count check at the end of `seed`. Without it a partial write is a
    healthy-looking episode over a directory that is missing a file, which is
    indistinguishable from a model that could not find it."""
    import gyms.sandbox as module

    real = module.run
    monkeypatch.setattr(
        module, "run",
        lambda s, c, t=0: module.Ran("0\n", "", 0, False) if c.startswith("find") else real(s, c, t),
    )
    with pytest.raises(SandboxError, match="workspace holds"):
        seed(session, WORKSPACE, {"a.txt": "x"})


def test_clearing_a_workspace_empties_it_for_the_next_task(session):
    """`sandbox.clear` against the real shell, because the in-process fake is
    free to agree with whatever it is told. A batch of demonstrations reuses one
    session, so a clear that leaves anything behind seeds the next task over the
    last one — which the seeding verification catches as a count mismatch, and
    a caller reads as the sandbox being broken."""
    task = tasks.generate_grep(4)
    seed(session, WORKSPACE, task.files)
    assert run(session, "ls").stdout.strip()

    clear(session)
    assert run(session, "ls").stdout.strip() == ""

    # And the workspace is still usable afterwards, not merely empty.
    seed(session, WORKSPACE, {"after.txt": "still works"})
    assert run(session, "cat after.txt").stdout.strip() == "still works"
