"""Parent-only actor-bound effect dispatch with durable issued-before-write protection."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
import json
import logging
import os
from pathlib import Path
import threading
import math
from uuid import UUID

from llm.runtime.artifacts import ArtifactStore, sync_directory
from llm.runtime.protocol import EffectReply, EffectRequest, RuntimeArtifact

EffectDispatch = Callable[[str, EffectRequest, ArtifactStore], RuntimeArtifact | None]
LOG = logging.getLogger(__name__)


class EffectJournal:
    """Bound a durable operation-ID ledger; a recorded write is never safe to replay."""

    def __init__(self, path: Path, max_bytes: int = 8 << 20) -> None:
        """Load an engine-owned ledger, refusing partial/corrupt or excessive prior evidence."""
        self.path = path
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        self._issued: set[UUID] = set()
        if path.exists():
            if path.stat().st_size > max_bytes:
                raise ValueError("runtime_effect_journal_budget")
            text = path.read_text()
            if text and not text.endswith("\n"):
                raise ValueError("runtime_effect_journal_incomplete")
            for line in text.splitlines():
                self._issued.add(UUID(json.loads(line)["operation_id"]))

    def issue(self, operation_id: UUID) -> None:
        """Fsync an operation identity before a remote write; duplicate/uncertain IDs fail closed."""
        with self._lock:
            if operation_id in self._issued:
                raise ValueError("runtime_effect_already_issued")
            line = json.dumps({"operation_id": str(operation_id)}, separators=(",", ":")) + "\n"
            size = self.path.stat().st_size if self.path.exists() else 0
            if size + len(line) > self.max_bytes:
                raise ValueError("runtime_effect_journal_budget")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            with os.fdopen(descriptor, "ab") as output:
                output.write(line.encode())
                output.flush()
                os.fsync(output.fileno())
            sync_directory(self.path.parent)
            self._issued.add(operation_id)


class EffectBroker:
    """Own the initiating actor and one outstanding bounded service-adapter invocation."""

    def __init__(self, actor: str, dispatch: EffectDispatch, journal: EffectJournal) -> None:
        """Retain actor/service handles only in the parent; requests cannot override either."""
        if not actor or len(actor) > 128:
            raise ValueError("runtime_effect_actor")
        self.actor = actor
        self.dispatch = dispatch
        self.journal = journal
        self._active: Future[EffectReply] | None = None
        self._lock = threading.Lock()
        self._closed = False

    def begin(self, request: EffectRequest, artifacts: ArtifactStore) -> Future[EffectReply]:
        """Start one parent callback while the guardian independently supervises deadlines.

        Production dispatchers must use the existing finite-timeout Rust Hop and
        validate operation-specific data. A late reply never reissues the effect.
        """
        with self._lock:
            if self._closed:
                raise ValueError("runtime_effect_broker_closed")
            if self._active is not None and not self._active.done():
                raise ValueError("runtime_effect_still_outstanding")
            if request.effect.operation not in {"dataset_read", "gym_read", "gym_catalogue"}:
                self.journal.issue(request.operation_id)
            future: Future[EffectReply] = Future()
            self._active = future
            thread = threading.Thread(
                target=self._dispatch, args=(request, artifacts, future),
                name="llm-runtime-effect", daemon=True,
            )
            thread.start()
            return future

    def close(self, timeout: float) -> None:
        """Stop new effects and drain the actual parent callback before its grant/session resources can retire.

        Timeout never certifies that an external write stopped; the session
        owner must retain artifacts and quarantine admission on that outcome.
        Issued writes keep their journal identity regardless of the reply.
        """
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("runtime_effect_drain_timeout")
        with self._lock:
            self._closed = True
            active = self._active
        if active is not None:
            active.result(timeout=timeout)

    def _dispatch(self, request: EffectRequest, artifacts: ArtifactStore, future: Future[EffectReply]) -> None:
        """Translate a callback result without exposing internal exception text or retrying writes."""
        try:
            result = self.dispatch(self.actor, request, artifacts)
            if result is not None:
                artifacts.verify(result)
            reply = EffectReply(type="effect_reply", operation_id=request.operation_id, outcome="completed", result=result)
        except Exception:
            write = request.effect.operation not in {"dataset_read", "gym_read", "gym_catalogue"}
            LOG.exception("Parent effect %s (%s) failed; write_issued=%s", request.operation_id, request.effect.operation, write)
            reply = EffectReply(
                type="effect_reply", operation_id=request.operation_id,
                outcome="uncertain" if write else "failed", error_code="remote_effect_failed",
            )
        future.set_result(reply)
