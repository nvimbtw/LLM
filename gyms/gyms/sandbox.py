"""The shell an episode runs in, and how a task's files get into it.

`deploy/just-bash/` is the sandbox — bash simulated over a virtual filesystem,
no `execve` under any of it — reached over HTTP at `BASH_URL`. It keys a
filesystem by the `session` string it is handed, so **the episode id is the
session key**: one episode is one workspace, and a file written in step 3 is
still there in step 4.

The backend's chat tool talks to the same service (`backend/crates/tools/src/bash.rs`)
and this module deliberately does not try to share anything with it beyond the
wire shape. What it must not do is *render* — the observation the model reads is
built on the Rust side, by the identical function chat uses, so a gym cannot
teach a layout serving does not show. This module returns the raw fields.
"""

import os
import secrets
from dataclasses import dataclass

import httpx

#: Where the shell starts, and therefore where a task's files go.
#:
#: **Not a directory of our choosing.** just-bash opens every command in
#: `/home/user`, and — this is the part that bites — *each command gets a fresh
#: shell*: the filesystem is shared but the working directory is not, so a `cd`
#: in one step is gone by the next. A model that has to navigate to its files
#: therefore has to do it again in every single command, using an absolute path,
#: forever.
#:
#: Seeding somewhere else cost exactly that. The files landed in
#: `/workspace/files`, the model ran `ls -la`, saw an empty `/home/user`, and
#: answered that the directory was empty and could it please be given a path.
#: It was not wrong. So the workspace is wherever the shell already is, and
#: `ls` is a command that works.
WORKSPACE = "/home/user"

#: The sandbox's own ceiling for one command, and ours for waiting on it. The
#: second is the larger of the two on purpose: a timeout the sandbox reports is
#: an answer the model can read ("still running"), while a timeout on this side
#: is an episode that failed.
COMMAND_TIMEOUT_MS = 10_000
SEED_TIMEOUT_MS = 5_000
HTTP_TIMEOUT_SECS = 15.0


class SandboxError(RuntimeError):
    """The sandbox could not be reached, or refused what it was asked.

    Raised rather than swallowed. A reset that cannot seed its files must not
    hand back a healthy-looking episode over an empty workspace: every command
    the model then runs finds nothing, and the run reports a model that failed
    every task it was never actually given.
    """


@dataclass(frozen=True)
class Ran:
    """One command, as the sandbox reported it."""

    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


def url() -> str:
    return os.environ.get("BASH_URL", "http://localhost:8899").rstrip("/")


def run(session: str, command: str, timeout_ms: int = COMMAND_TIMEOUT_MS) -> Ran:
    """Run one command in `session`'s filesystem."""
    try:
        response = httpx.post(
            f"{url()}/run",
            json={"session": session, "command": command, "timeout_ms": timeout_ms},
            timeout=HTTP_TIMEOUT_SECS,
        )
    except httpx.HTTPError as e:
        raise SandboxError(f"just-bash at {url()} did not answer: {e}") from e
    if response.status_code != 200:
        raise SandboxError(f"just-bash answered {response.status_code}: {response.text[:200]}")
    body = response.json()
    return Ran(
        stdout=body.get("stdout", ""),
        stderr=body.get("stderr", ""),
        exit_code=int(body.get("exit_code", 0)),
        timed_out=bool(body.get("timed_out", False)),
    )


def clear(session: str, root: str = WORKSPACE) -> None:
    """Empty a workspace so its session can hold the next task.

    The sandbox has no "forget this session" call and keeps a bounded number of
    filesystems, so a caller generating many workspaces in a row — building
    demonstrations, not running episodes — has to reuse one rather than mint a
    key per task. An episode must not use this: its id *is* its filesystem, and
    a workspace that outlives its episode is the bug the id keys against.
    """
    run(session, f"rm -rf {root}/* {root}/.[!.]* 2>/dev/null; true")


def seed(session: str, root: str, files: dict[str, str]) -> None:
    """Write `files` under `root`, then prove they are there.

    One command per file: a single script would be one long line the sandbox
    body limit could cut, and a partial write is the failure this whole module
    exists to make impossible. The heredoc delimiter is drawn per call and
    checked against the content, because a task's file could otherwise contain
    the delimiter and end its own write early.
    """
    for relative, content in files.items():
        ran = run(session, _write(root, relative, content), SEED_TIMEOUT_MS)
        if ran.exit_code != 0 or ran.timed_out:
            raise SandboxError(
                f"could not seed {relative!r}: exit {ran.exit_code} {ran.stderr[:200]}"
            )
    _verify(session, root, len(files))


def _write(root: str, relative: str, content: str) -> str:
    if relative.startswith("/") or ".." in relative.split("/"):
        raise SandboxError(f"refusing to seed outside the workspace: {relative!r}")
    delimiter = f"GYM_EOF_{secrets.token_hex(8)}"
    while delimiter in content:
        delimiter = f"GYM_EOF_{secrets.token_hex(8)}"
    path = _quote(f"{root}/{relative}")
    parent = _quote(f"{root}/{relative}".rsplit("/", 1)[0])
    return f"mkdir -p {parent} && cat > {path} << '{delimiter}'\n{content}\n{delimiter}"


def _verify(session: str, root: str, expected: int) -> None:
    ran = run(session, f"find {_quote(root)} -type f | wc -l", SEED_TIMEOUT_MS)
    counted = ran.stdout.strip()
    if ran.exit_code != 0 or not counted.isdigit() or int(counted) != expected:
        raise SandboxError(
            f"seeded {expected} files but the workspace holds {counted or '?'}"
        )


def _quote(text: str) -> str:
    """POSIX single-quoting. Filenames are generated here, but the quoting is
    what makes that a fact about today rather than a thing to remember."""
    return "'" + text.replace("'", "'\\''") + "'"
