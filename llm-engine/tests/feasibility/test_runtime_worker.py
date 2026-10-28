"""Executable ISS-48 CPU worker evidence, independent of application fixtures.

Run from the repository root:
engine/.venv/bin/python -m unittest discover -s llm-engine/tests/feasibility -v
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from runtime_worker import (
    Admission,
    EffectBroker,
    Frame,
    MAX_ARTIFACT_BYTES,
    MAX_PACKET_BYTES,
    MAX_PROGRESS_FRAMES,
    Resident,
    receive,
)


def wait_gone(pid: int, timeout: float = 5.0) -> bool:
    """Wait a bounded interval for a worker/helper PID to be fully reaped."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not Path(f"/proc/{pid}").exists():
            return True
        time.sleep(0.02)
    return False


class WorkerEvidence(unittest.TestCase):
    """Observe actual spawned torch workers, process lifetimes and IPC failures."""

    def setUp(self) -> None:
        """Allocate an isolated artifact directory and parent admission owner."""
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name)
        self.gate = Admission()
        self.workers: list[Resident] = []

    def tearDown(self) -> None:
        """Drain every owned subprocess before deleting its disposable artifacts."""
        for worker in self.workers:
            worker.close()
        self.directory.cleanup()

    def spawn(self, mode: str = "normal", ready: bool = True) -> Resident:
        """Start one admitted worker and optionally await bounded model construction."""
        worker = Resident(self.gate, self.path, mode)
        self.workers.append(worker)
        if ready:
            worker.await_ready()
        return worker

    def test_one_parent_owned_lease_and_resident_optimizer_serving(self) -> None:
        """Training and two generations reuse one PID/optimizer, with bounded file output."""
        import torch

        worker = self.spawn()
        with self.assertRaisesRegex(RuntimeError, "admission closed"):
            Resident(self.gate, self.path)
        first = worker.run(Frame("train", value=2), timeout=10)
        second = worker.run(Frame("train", value=1), timeout=10)
        self.assertEqual((first.value, second.value), (2, 3))
        self.assertEqual(first.detail, second.detail)
        saved = torch.load(self.path / "recovery.pt", weights_only=True)
        self.assertEqual(saved["steps"], 3)
        self.assertEqual({int(state["step"]) for state in saved["optimizer"]["state"].values()}, {3})
        self.assertLess((self.path / "recovery.pt").stat().st_size, MAX_ARTIFACT_BYTES)
        first_decode = worker.run(Frame("decode", value=5), timeout=10)
        second_decode = worker.run(Frame("decode", value=2), timeout=10)
        self.assertEqual((first_decode.value, second_decode.value), (5, 2))
        self.assertEqual(first_decode.detail, first.detail)
        self.assertEqual(second_decode.detail, first.detail)
        time.sleep(1.1)
        self.assertEqual(worker.run(Frame("decode", value=1), timeout=10).detail, first.detail)
        self.assertTrue(worker.close())
        self.assertTrue(wait_gone(int(first.detail)))
        replacement = self.spawn()
        self.assertTrue(replacement.close())

    def test_source_inherits_no_service_credentials_or_unrelated_handles(self) -> None:
        """A clean interpreter receives only the explicit environment and IPC descriptor."""
        with patch.dict(os.environ, {"BROWSER_TOKEN": "test-token", "SERVICE_SECRET": "test-secret"}):
            worker = self.spawn()
        environment = next(frame.detail for frame in worker.frames if frame.stage == "environment")
        self.assertEqual(set(environment.split(",")), {"PATH", "LANG", "OMP_NUM_THREADS", "MKL_NUM_THREADS"})
        pid = next(frame.value for frame in worker.frames if frame.kind == "ready")
        descriptors = list(Path(f"/proc/{pid}/fd").iterdir())
        self.assertLessEqual(len(descriptors), 5)
        self.assertTrue(worker.close())

    def test_import_and_construction_hangs_are_terminated(self) -> None:
        """Source import and constructor failures cannot outlive cancellation grace."""
        for mode in ("import", "construction"):
            with self.subTest(mode=mode):
                worker = self.spawn(mode, ready=False)
                while worker.next().stage != mode:
                    pass
                started = time.monotonic()
                with self.assertRaises(TimeoutError):
                    worker.next(timeout=0.1)
                self.assertTrue(worker.close())
                self.assertLess(time.monotonic() - started, 3)

    def test_forward_and_actual_autograd_backward_hangs_are_terminated(self) -> None:
        """After successful construction, hung forward/backward cannot release an active lease."""
        for mode in ("forward", "backward"):
            with self.subTest(mode=mode):
                worker = self.spawn(mode)
                self.assertTrue(any(frame.stage == "validated" for frame in worker.frames))
                with self.assertRaises(TimeoutError):
                    worker.run(Frame("train", value=1), timeout=0.2)
                self.assertTrue(self.gate.busy)
                self.assertTrue(worker.close())
                self.assertFalse(self.gate.busy)
                self.assertFalse((self.path / "recovery.pt").exists())

    def test_late_decode_hang_retains_complete_recovery_and_releases_tree(self) -> None:
        """A hang after three decoded tokens preserves the last complete training checkpoint."""
        worker = self.spawn("decode")
        trained = worker.run(Frame("train", value=2), timeout=10)
        complete = (self.path / "recovery.pt").read_bytes()
        with self.assertRaises(TimeoutError):
            worker.run(Frame("decode", value=8), timeout=0.2)
        tokens = [frame.value for frame in worker.frames if frame.stage == "token"]
        self.assertEqual(tokens, [1, 2, 3])
        self.assertTrue(worker.close())
        self.assertTrue(wait_gone(int(trained.detail)))
        self.assertEqual((self.path / "recovery.pt").read_bytes(), complete)

    def test_ordinary_descendants_are_killed_and_reaped(self) -> None:
        """Guardian cleanup includes an actual helper subprocess created by the worker."""
        worker = self.spawn("descendant")
        worker.channel.sendall(Frame("decode", value=1).encode())
        helper = worker.next()
        self.assertEqual(helper.stage, "descendant")
        self.assertTrue(Path(f"/proc/{helper.value}").exists())
        self.assertTrue(worker.close())
        self.assertTrue(wait_gone(helper.value))

    def test_memory_and_artifact_limits_fail_bounded(self) -> None:
        """Actual over-budget host allocation/file output fails without unbounded growth."""
        for mode, expected in (("allocation", "MemoryError"), ("output", "OSError")):
            with self.subTest(mode=mode):
                worker = self.spawn(mode)
                frame = worker.run(Frame("decode", value=1), timeout=10)
                self.assertEqual((frame.kind, frame.detail), ("error", expected))
                self.assertTrue(worker.close())
                output = self.path / "excessive-output"
                if output.exists():
                    self.assertLessEqual(output.stat().st_size, MAX_ARTIFACT_BYTES)

    def test_frame_schema_and_kernel_datagram_truncation(self) -> None:
        """Malformed, excessive and truncated records never become interpreted IPC."""
        invalid = [b"{}", b"[]", b"null", b'{"kind":"stop","kind":"train"}', b"x" * (MAX_PACKET_BYTES + 1)]
        for raw in invalid:
            with self.assertRaises(ValueError):
                Frame.decode(raw)
        for frame in (Frame("unknown"), Frame("train", value=True), Frame("train", detail="x" * 1025)):
            with self.assertRaises(ValueError):
                frame.encode()
        first, second = socket.socketpair(socket.AF_UNIX, socket.SOCK_SEQPACKET)
        try:
            first.sendall(b"x" * (MAX_PACKET_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "truncated"):
                receive(second)
        finally:
            first.close()
            second.close()

    def test_progress_is_bounded_even_when_worker_floods(self) -> None:
        """A fast producer cannot grow parent report retention indefinitely."""
        worker = self.spawn("flood")
        with self.assertRaisesRegex(ValueError, "progress budget exceeded"):
            worker.run(Frame("decode", value=1), timeout=5)
        self.assertEqual(len(worker.frames), MAX_PROGRESS_FRAMES)
        self.assertTrue(worker.close())

    def test_uncertain_cleanup_keeps_admission_closed(self) -> None:
        """An externally killed guardian cannot falsely certify a released lease."""
        worker = self.spawn()
        pid = next(frame.value for frame in worker.frames if frame.kind == "ready")
        worker.guardian.kill()
        worker.guardian.wait(timeout=2)
        try:
            self.assertFalse(worker.close())
            with self.assertRaisesRegex(RuntimeError, "admission closed"):
                Resident(self.gate, self.path)
        finally:
            os.killpg(pid, signal.SIGKILL)
            self.assertTrue(wait_gone(pid))

    def test_actor_owned_rust_effect_is_never_replayed_after_uncertain_completion(self) -> None:
        """The worker cannot select an actor or replay an already-issued gym write."""
        calls: list[tuple[str, str]] = []
        broker = EffectBroker("initiating-user", lambda actor, operation: calls.append((actor, operation)))
        worker = self.spawn("effect")
        with self.assertRaises(TimeoutError):
            worker.run(Frame("decode", value=1), timeout=0.2, broker=broker)
        self.assertTrue(worker.close())
        self.assertEqual(calls, [("initiating-user", "gym-write-1")])
        with self.assertRaisesRegex(RuntimeError, "already issued"):
            broker.issue("gym-write-1")
        self.assertEqual(len(calls), 1)

    def test_lost_rust_reply_keeps_write_issued(self) -> None:
        """An adapter exception leaves an effect uncertain instead of making it retryable."""
        calls: list[str] = []

        def lost_reply(actor: str, operation: str) -> None:
            """Simulate Rust committing an effect and disconnecting before acknowledgement."""
            calls.append(operation)
            raise ConnectionError("lost acknowledgement")

        broker = EffectBroker("initiating-user", lost_reply)
        with self.assertRaises(ConnectionError):
            broker.issue("gym-write-1")
        with self.assertRaises(RuntimeError):
            broker.issue("gym-write-1")
        self.assertEqual(calls, ["gym-write-1"])

    def test_parent_sigkill_reaps_worker_and_helper_before_restart(self) -> None:
        """Killing the actual engine stand-in closes its sole guardian liveness socket."""
        program = """
import json,sys,time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from runtime_worker import Admission,Resident,Frame
w=Resident(Admission(),Path(sys.argv[2]),'descendant')
while (f:=w.next()).kind!='ready': pass
w.channel.sendall(Frame('decode',value=1).encode())
helper=w.next().value
print(json.dumps([w.guardian.pid,f.value,helper]),flush=True)
time.sleep(300)
"""
        parent = subprocess.Popen(
            [sys.executable, "-I", "-c", program, str(Path(__file__).parent.resolve()), str(self.path)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, close_fds=True,
        )
        try:
            readable, _, _ = select.select([parent.stdout], [], [], 30)
            self.assertTrue(readable, "engine stand-in did not become ready")
            guardian, worker, helper = json.loads(parent.stdout.readline())
            parent.kill()
            parent.wait(timeout=2)
            self.assertTrue(wait_gone(worker))
            self.assertTrue(wait_gone(helper))
            self.assertTrue(wait_gone(guardian))
            restarted = self.spawn()
            self.assertTrue(restarted.close())
        finally:
            if parent.poll() is None:
                parent.kill()
            parent.communicate(timeout=5)


if __name__ == "__main__":
    unittest.main()
