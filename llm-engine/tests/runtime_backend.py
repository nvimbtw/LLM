"""Trusted fake numerical backends for production worker lifecycle contract tests."""

import io
import json
import os
import subprocess
import sys
import time

import torch

from llm.runtime.protocol import (
    Completed, EffectData, Generate, GymFinishEffect, Load, Progress, Train,
)
from llm.runtime.worker import WorkerContext


def hang(*args: object) -> None:
    """Inject custom code that never reaches another cooperative boundary."""
    while True:
        time.sleep(0.05)


class ResidentBackend:
    """Keep real torch model/optimizer state in one child across complete commands."""

    mode = "normal"

    def __init__(self, context: WorkerContext) -> None:
        """Construct a tiny deterministic resident model without service/storage handles."""
        torch.set_num_threads(1)
        torch.manual_seed(52)
        self.model = torch.nn.Linear(4, 4).to(context.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.01)
        self.steps = 0
        self.cache = None

    def handle(self, payload, context: WorkerContext) -> Completed:
        """Execute one whole job/request; outputs are bounded materializations, never tensors."""
        if isinstance(payload, Load):
            self.model.eval()
            with torch.no_grad():
                assert torch.isfinite(self.model(torch.ones(1, 4, device=context.device))).all()
            if self.mode == "forward":
                self.model.register_forward_hook(hang)
            if self.mode == "backward":
                self.model.register_full_backward_hook(hang)
            return Completed(type="completed", outcome="ok")
        if isinstance(payload, Train):
            return self.train(payload, context)
        if isinstance(payload, Generate):
            return self.generate(payload, context)
        return Completed(type="completed", outcome="ok")

    def train(self, payload: Train, context: WorkerContext) -> Completed:
        """Run all optimizer steps locally and await parent adoption at completed boundaries."""
        config = json.loads(context.artifacts.read_bytes(payload.config))
        self.model.train()
        for _ in range(config.get("steps", 3)):
            context.progress(Progress(type="progress", phase="forward", completed=self.steps, unit="steps"), force=True)
            self.inject(context)
            self.optimizer.zero_grad(set_to_none=True)
            loss = self.model(torch.ones(2, 4, device=context.device)).square().mean()
            context.progress(Progress(type="progress", phase="backward", completed=self.steps, unit="steps"), force=True)
            loss.backward()
            self.optimizer.step()
            self.steps += 1
            if config.get("checkpoint", False):
                buffer = io.BytesIO()
                torch.save({"model": self.model.state_dict(), "optimizer": self.optimizer.state_dict(), "step": self.steps}, buffer)
                grant = context.artifacts.write_bytes(buffer.getvalue(), "checkpoint_v1")
                context.publish(grant, completed_step=self.steps)
            context.progress(Progress(type="progress", phase="backward", completed=self.steps, unit="steps"), force=True)
            time.sleep(config.get("delay", 0))
        result = context.artifacts.write_bytes(json.dumps({"steps": self.steps, "pid": os.getpid()}).encode(), "training_result_v1")
        return Completed(type="completed", outcome="ok", result=result)

    def inject(self, context: WorkerContext) -> None:
        """Exercise actual helper, allocation, stream flood, CPU hang and abrupt exit failures."""
        if self.mode == "crash":
            os._exit(17)
        if self.mode == "allocation":
            bytearray(32 << 30)
        if self.mode == "descendant":
            helper = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], close_fds=True)
            context.progress(Progress(type="progress", phase="forward", completed=helper.pid, unit="phases"), force=True)
            hang()
        if self.mode == "flood":
            for index in range(4096):
                context.progress(Progress(type="progress", phase="forward", completed=index, unit="phases"), force=True)
        if self.mode == "output":
            print("x" * (1 << 20), flush=True)
            hang()
        if self.mode == "effect":
            grant = context.artifacts.write_bytes(b'{"reply":"answer"}', "effect_data_v1")
            context.effect(GymFinishEffect(
                operation="gym_finish", episode_id=context.request_id,
                payload=EffectData.model_validate(grant.model_dump()),
            ))

    def generate(self, payload: Generate, context: WorkerContext) -> Completed:
        """Keep a request cache local, with a late-decode hang after several successful steps."""
        self.model.eval()
        self.cache = torch.zeros(1, 4, device=context.device)
        with torch.no_grad():
            for index in range(payload.max_tokens):
                context.progress(Progress(type="progress", phase="generation", completed=index, unit="tokens"), force=True)
                if self.mode == "decode" and index == 3:
                    hang()
                self.cache = self.model(self.cache)
        return Completed(type="completed", outcome="ok")

    def finish_request(self) -> None:
        """Drop request tensors while keeping the model and AdamW state resident."""
        self.cache = None

    def close(self) -> None:
        """Release all tensors before guardian process-tree cleanup."""
        self.cache = self.optimizer = self.model = None


class ConstructionHang(ResidentBackend):
    """A backend whose custom constructor never completes."""

    def __init__(self, context: WorkerContext) -> None:
        """Hang inside actual constructor execution after import completed."""
        hang()


class ForwardHang(ResidentBackend):
    """Fail during model forward after successful model loading/validation."""
    mode = "forward"


class BackwardHang(ResidentBackend):
    """Fail from an actual autograd backward callback after successful loading."""
    mode = "backward"


class DecodeHang(ResidentBackend):
    """Fail after three successful decode steps."""
    mode = "decode"


class Descendant(ResidentBackend):
    """Create one ordinary owned helper for cleanup/count evidence."""
    mode = "descendant"


class Flood(ResidentBackend):
    """Flood typed progress to exercise bounded backpressure."""
    mode = "flood"


class Allocation(ResidentBackend):
    """Request an address allocation larger than the per-process limit."""
    mode = "allocation"


class Crash(ResidentBackend):
    """Exit abruptly without running backend/model cleanup."""
    mode = "crash"


class Output(ResidentBackend):
    """Write excessive standard output, which is discarded rather than buffered."""
    mode = "output"


class Effects(ResidentBackend):
    """Request a real parent-side fake Rust adapter invocation."""
    mode = "effect"
