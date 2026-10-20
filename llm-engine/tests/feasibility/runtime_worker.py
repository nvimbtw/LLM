"""ISS-48 Linux supervision experiment; deliberately not a production executor.

A trusted guardian remains outside the custom-code process group and reaps it
when its engine socket closes. This proves ordinary descendant cleanup, not a
sandbox or containment of descendants which escape the group. No service code
imports this module.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import os
from pathlib import Path
import resource
import select
import signal
import socket
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.nn import Module
    from torch.optim import Optimizer

MAX_PACKET_BYTES = 4096
MAX_SOURCE_BYTES = 1 << 20
MAX_ARTIFACT_BYTES = 1 << 20
MAX_PROGRESS_FRAMES = 64
STARTUP_SECONDS = 30.0
GRACE_SECONDS = 0.15
REAP_SECONDS = 2.0


@dataclass(frozen=True)
class Frame:
    """A bounded interpreted IPC record with no tensor, service or actor handles."""

    kind: str
    stage: str = ""
    value: int = 0
    detail: str = ""

    def encode(self) -> bytes:
        """Serialize this typed record, refusing excessive strings or packet bytes."""
        if not isinstance(self.kind, str) or self.kind not in {
            "ready", "train", "decode", "stop", "cancel", "progress",
            "done", "effect", "error", "clean",
        }:
            raise ValueError("unknown IPC kind")
        if type(self.value) is not int or not 0 <= self.value < (1 << 31):
            raise ValueError("invalid IPC integer")
        if not isinstance(self.stage, str) or len(self.stage) > 64:
            raise ValueError("invalid stage")
        if not isinstance(self.detail, str) or len(self.detail) > 1024:
            raise ValueError("invalid detail")
        encoded = json.dumps(asdict(self), allow_nan=False).encode()
        if len(encoded) > MAX_PACKET_BYTES:
            raise ValueError("IPC packet budget exceeded")
        return encoded

    @classmethod
    def decode(cls, raw: bytes) -> Frame:
        """Validate an untrusted IPC datagram before interpreting any of its fields."""
        if not raw or len(raw) > MAX_PACKET_BYTES:
            raise ValueError("IPC packet budget exceeded")
        value = json.loads(raw, object_pairs_hook=unique_object)
        if not isinstance(value, dict) or set(value) != {"kind", "stage", "value", "detail"}:
            raise ValueError("invalid IPC schema")
        frame = cls(**value)
        frame.encode()
        return frame


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object fields instead of interpreting an ambiguous record."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate IPC field")
        result[key] = value
    return result


def receive(channel: socket.socket) -> Frame:
    """Receive exactly one bounded datagram; reject truncation and peer closure."""
    raw, _, flags, _ = channel.recvmsg(MAX_PACKET_BYTES)
    if flags & socket.MSG_TRUNC:
        raise ValueError("truncated IPC packet")
    if not raw:
        raise EOFError("IPC peer closed")
    return Frame.decode(raw)


def send(channel: socket.socket, frame: Frame) -> None:
    """Send one bounded datagram under the channel's configured send deadline."""
    channel.sendall(frame.encode())


class Admission:
    """Parent-only single lease; uncertain cleanup permanently closes this instance."""

    def __init__(self) -> None:
        """Initialize a free parent lease with no worker-side lock copy."""
        self.owner_pid = os.getpid()
        self.busy = False
        self.quarantined = False

    def acquire(self) -> None:
        """Refuse competing work, worker-side use, and quarantined admission."""
        if os.getpid() != self.owner_pid or self.busy or self.quarantined:
            raise RuntimeError("admission closed")
        self.busy = True

    def finish(self, clean: bool) -> None:
        """Release only verified cleanup; otherwise retain a closed admission gate."""
        if os.getpid() != self.owner_pid:
            raise RuntimeError("admission has a different owner")
        self.quarantined = self.quarantined or not clean
        self.busy = not clean


class EffectBroker:
    """Engine-owned actor and at-most-once effect issue ledger for the Rust seam."""

    def __init__(self, actor: str, via_rust: Callable[[str, str], None]) -> None:
        """Retain the initiating actor and Rust adapter outside the worker."""
        self.actor = actor
        self.via_rust = via_rust
        self.issued: set[str] = set()

    def issue(self, operation: str) -> None:
        """Record before issuing a write; uncertain completion is never replayed."""
        if operation in self.issued:
            raise RuntimeError("effect already issued; completion may be uncertain")
        if len(self.issued) >= MAX_PROGRESS_FRAMES:
            raise ValueError("effect budget exceeded")
        if not operation or len(operation) > 64:
            raise ValueError("invalid effect identity")
        self.issued.add(operation)
        self.via_rust(self.actor, operation)


class Resident:
    """Own a fresh guardian, resident model worker, and one parent admission lease."""

    def __init__(self, gate: Admission, directory: Path, mode: str = "normal") -> None:
        """Acquire before spawning; pass only an artifact directory and bounded mode."""
        if mode not in {
            "normal", "import", "construction", "forward", "backward", "decode",
            "descendant", "flood", "effect", "allocation", "output",
        }:
            raise ValueError("unknown experiment mode")
        gate.acquire()
        self.gate = gate
        self.closed = False
        self.frames: deque[Frame] = deque(maxlen=MAX_PROGRESS_FRAMES)
        self.channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        self.channel.settimeout(STARTUP_SECONDS)
        environment = {"PATH": os.defpath, "LANG": "C.UTF-8", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        try:
            self.guardian = subprocess.Popen(
                [
                    sys.executable, "-I", str(Path(__file__).resolve()), "guardian",
                    str(child.fileno()), str(directory), mode,
                ],
                pass_fds=(child.fileno(),), close_fds=True, env=environment,
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
        except BaseException:
            self.channel.close()
            gate.finish(True)
            raise
        finally:
            child.close()

    def next(self, timeout: float = STARTUP_SECONDS) -> Frame:
        """Receive bounded progress without resetting any caller-owned absolute deadline."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise TimeoutError("worker deadline exceeded")
        self.channel.settimeout(timeout)
        frame = receive(self.channel)
        self.frames.append(frame)
        return frame

    def await_ready(self, timeout: float = STARTUP_SECONDS) -> Frame:
        """Bound the complete import/construction interval, including progress flooding."""
        deadline = time.monotonic() + timeout
        for _ in range(MAX_PROGRESS_FRAMES):
            frame = self.next(deadline - time.monotonic())
            if frame.kind == "ready":
                return frame
            if frame.kind in {"error", "clean"}:
                raise RuntimeError(f"worker startup failed: {frame.detail}")
        raise ValueError("startup progress budget exceeded")

    def run(self, request: Frame, timeout: float, broker: EffectBroker | None = None) -> Frame:
        """Run one whole training/generation command under a fixed absolute deadline."""
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("invalid execution deadline")
        send(self.channel, request)
        deadline = time.monotonic() + timeout
        for _ in range(MAX_PROGRESS_FRAMES):
            frame = self.next(deadline - time.monotonic())
            if frame.kind == "effect":
                if broker is None:
                    raise ValueError("effect has no parent owner")
                broker.issue(frame.detail)
                send(self.channel, Frame("done", "effect"))
            if frame.kind in {"done", "error", "clean"}:
                return frame
        raise ValueError("progress budget exceeded")

    def close(self) -> bool:
        """Request cooperative stop, then let the guardian terminate/reap the group.

        Missing cleanup acknowledgement or guardian exit quarantines admission.
        This does not certify descendants that escaped the process group.
        """
        if self.closed:
            return not self.gate.quarantined
        self.closed = True
        clean = False
        try:
            self.channel.settimeout(GRACE_SECONDS)
            send(self.channel, Frame("cancel"))
            deadline = time.monotonic() + GRACE_SECONDS + REAP_SECONDS
            while time.monotonic() < deadline:
                self.channel.settimeout(max(0.001, deadline - time.monotonic()))
                frame = receive(self.channel)
                if frame.kind == "clean":
                    clean = frame.value == 1
                    break
        except (OSError, EOFError, ValueError):
            clean = False
        finally:
            self.channel.close()
            try:
                self.guardian.wait(timeout=REAP_SECONDS)
            except subprocess.TimeoutExpired:
                self.guardian.kill()
                self.guardian.wait(timeout=REAP_SECONDS)
                clean = False
            self.gate.finish(clean and self.guardian.returncode == 0)
        return not self.gate.quarantined


def configure_subreaper() -> None:
    """Make the trusted Linux guardian reap orphaned ordinary worker descendants."""
    library = ctypes.CDLL(None, use_errno=True)
    if library.prctl(36, 1, 0, 0, 0) != 0:
        raise OSError(ctypes.get_errno(), "PR_SET_CHILD_SUBREAPER failed")


def reap_group(worker: subprocess.Popen[bytes]) -> bool:
    """Kill the owned group and reap adopted children; bound verification to two seconds."""
    try:
        os.killpg(worker.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + REAP_SECONDS
    worker.wait(timeout=REAP_SECONDS)
    while time.monotonic() < deadline:
        try:
            child, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return True
        if child == 0:
            time.sleep(0.005)
    return False


def guardian(channel: socket.socket, directory: str, mode: str) -> None:
    """Bridge bounded packets while watching engine liveness, then kill/reap children."""
    configure_subreaper()
    worker_channel, child = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    worker = subprocess.Popen(
        [sys.executable, "-I", str(Path(__file__).resolve()), "worker", str(child.fileno()), directory, mode],
        pass_fds=(child.fileno(),), close_fds=True, start_new_session=True,
        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    child.close()
    channel.settimeout(GRACE_SECONDS)
    worker_channel.settimeout(GRACE_SECONDS)
    stopping_at: float | None = None
    try:
        while stopping_at is None or time.monotonic() < stopping_at:
            readable, _, _ = select.select([channel, worker_channel], [], [], 0.02)
            for source in readable:
                frame = receive(source)
                if source is channel:
                    send(worker_channel, frame)
                    if frame.kind in {"cancel", "stop"}:
                        stopping_at = time.monotonic() + GRACE_SECONDS
                else:
                    send(channel, frame)
    except (OSError, EOFError, ValueError):
        pass
    finally:
        clean = reap_group(worker)
        try:
            send(channel, Frame("clean", value=int(clean)))
        except OSError:
            pass
        channel.close()
        worker_channel.close()


def hang() -> None:
    """Represent custom Python that never reaches another cooperative boundary."""
    while True:
        time.sleep(0.1)


def stage(channel: socket.socket, name: str, mode: str) -> None:
    """Report the execution phase to let the parent choose its deadline policy."""
    send(channel, Frame("progress", name))


def hang_in_hook(*args: object) -> None:
    """Inject an uncooperative custom module inside torch forward/backward execution."""
    hang()


def construct_model(mode: str) -> Module:
    """Create custom model parameters, with a constructor hang failure injection."""
    import torch

    if mode == "construction":
        hang()
    return torch.nn.Linear(4, 4)


def validate_then_inject(channel: socket.socket, model: Module, mode: str) -> None:
    """Check a finite eager forward before injecting later runtime-only failures.

    This is a fixture acceptance check, not the production capture validator.
    """
    import torch

    with torch.no_grad():
        if not torch.isfinite(model(torch.ones(2, 4))).all():
            raise ValueError("fixture validation failed")
    send(channel, Frame("progress", "validated"))
    if mode == "forward":
        model.register_forward_hook(hang_in_hook)
    if mode == "backward":
        model.register_full_backward_hook(hang_in_hook)


def apply_limits() -> None:
    """Bound per-process virtual address space, CPU seconds, files and inherited handles.

    These are experiment limits, not aggregate descendant/GPU budgets. CUDA wheels
    map many GiB of libraries even for CPU execution, hence the 16 GiB address cap.
    """
    resource.setrlimit(resource.RLIMIT_AS, (16 << 30, 16 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_ARTIFACT_BYTES, MAX_ARTIFACT_BYTES))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def write_checkpoint(model: Module, optimizer: Optimizer, directory: str, steps: int) -> int:
    """Publish one complete bounded recovery file atomically, retaining its predecessor on failure."""
    import torch

    target = Path(directory) / "recovery.pt"
    pending = target.with_suffix(".pending")
    with pending.open("wb") as output:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "steps": steps}, output)
        output.flush()
        os.fsync(output.fileno())
    if pending.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError("artifact budget exceeded")
    os.replace(pending, target)
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return target.stat().st_size


def train(
    channel: socket.socket,
    model: Module,
    optimizer: Optimizer,
    request: Frame,
    directory: str,
    mode: str,
    completed: int,
) -> int:
    """Keep the full autograd/optimizer lifecycle resident; IPC carries whole-job reports."""
    import torch

    if not 1 <= request.value <= 8:
        raise ValueError("training step budget exceeded")
    model.train()
    for _ in range(request.value):
        stage(channel, "forward", mode)
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.ones(2, 4)).square().mean()
        stage(channel, "backward", mode)
        loss.backward()
        optimizer.step()
        completed += 1
        size = write_checkpoint(model, optimizer, directory, completed)
        send(channel, Frame("progress", "checkpoint", completed, str(size)))
    send(channel, Frame("done", "train", completed, str(os.getpid())))
    return completed


def decode(channel: socket.socket, model: Module, request: Frame, mode: str) -> None:
    """Keep request-local cache in the reusable model worker and bound token count."""
    import torch

    if not 1 <= request.value <= 8:
        raise ValueError("decode token budget exceeded")
    model.eval()
    cache = torch.zeros(1, 4)
    with torch.no_grad():
        for step in range(request.value):
            if mode == "decode" and step == 3:
                stage(channel, "decode", mode)
                hang()
            cache = model(cache)
            send(channel, Frame("progress", "token", step + 1))
    del cache
    send(channel, Frame("done", "decode", request.value, str(os.getpid())))


def worker(channel: socket.socket, directory: str, mode: str) -> None:
    """Execute a tiny torch lifecycle with sanitized inheritance and explicit injections."""
    apply_limits()
    channel.settimeout(1.0)
    send(channel, Frame("progress", "environment", detail=",".join(sorted(os.environ))))
    stage(channel, "import", mode)
    if mode == "import":
        source = "import time\nwhile True:\n    time.sleep(0.1)\n"
        if len(source.encode()) > MAX_SOURCE_BYTES:
            raise ValueError("source budget exceeded")
        exec(compile(source, "fixture-revision.py", "exec"), {})
    import torch

    torch.set_num_threads(1)
    stage(channel, "construction", mode)
    torch.manual_seed(17)
    model = construct_model(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    validate_then_inject(channel, model, mode)
    completed = 0
    send(channel, Frame("ready", value=os.getpid(), detail=torch.__version__))
    while True:
        channel.settimeout(None)
        request = receive(channel)
        channel.settimeout(1.0)
        if request.kind in {"cancel", "stop"}:
            return
        if mode == "descendant":
            child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], close_fds=True)
            send(channel, Frame("progress", "descendant", child.pid))
            hang()
        if mode == "flood":
            for _ in range(MAX_PROGRESS_FRAMES * 2):
                send(channel, Frame("progress", "flood"))
            hang()
        if mode == "allocation":
            bytearray(32 << 30)
        if mode == "output":
            with (Path(directory) / "excessive-output").open("wb") as output:
                output.write(b"x" * (MAX_ARTIFACT_BYTES + 1))
        if mode == "effect":
            send(channel, Frame("effect", detail="gym-write-1"))
            receive(channel)
            hang()
        if request.kind == "train":
            completed = train(channel, model, optimizer, request, directory, mode, completed)
        elif request.kind == "decode":
            decode(channel, model, request, mode)
        else:
            raise ValueError("invalid worker command")


def main() -> None:
    """Run one explicitly selected subprocess role, reporting only bounded failures."""
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=("guardian", "worker"))
    parser.add_argument("descriptor", type=int)
    parser.add_argument("directory")
    parser.add_argument("mode")
    args = parser.parse_args()
    channel = socket.socket(fileno=args.descriptor)
    try:
        (guardian if args.role == "guardian" else worker)(channel, args.directory, args.mode)
    except Exception as error:
        try:
            send(channel, Frame("error", detail=type(error).__name__))
        except OSError:
            pass
        raise


if __name__ == "__main__":
    main()
