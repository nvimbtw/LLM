"""Session-scoped immutable byte grants; durable publication belongs to parent adapters."""

from collections.abc import Callable, Iterator
import hashlib
import os
from pathlib import Path
import stat
from uuid import uuid4

from llm.runtime.protocol import RuntimeArtifact, RuntimeResources


class ArtifactStore:
    """Own bounded flat materializations shared only by one engine session and its worker."""

    def __init__(self, directory: Path, resources: RuntimeResources) -> None:
        """Create the private grant directory under a caller-owned runtime session root."""
        self.directory = directory
        self.resources = resources
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    def _path(self, artifact: RuntimeArtifact) -> Path:
        """Resolve a validated UUID grant without accepting worker-supplied filesystem paths."""
        return self.directory / f"{artifact.artifact_id}.blob"

    def total_bytes(self) -> int:
        """Bound file count and cumulative stored bytes, rejecting links and special files."""
        total = 0
        for count, path in enumerate(self.directory.iterdir(), 1):
            if count > 4096:
                raise ValueError("runtime_artifact_count")
            info = path.lstat()
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("runtime_artifact_not_regular")
            total += info.st_size
        return total

    def write_bytes(self, data: bytes, format: str) -> RuntimeArtifact:
        """Publish bounded bytes as a new immutable grant without changing an earlier file."""
        return self._write(iter((data,)), len(data), format)

    def stage_file(self, source: Path, format: str, *, check_cancelled: Callable[[], None] = lambda: None) -> RuntimeArtifact:
        """Snapshot a parent-authorized file in bounded cancellable chunks; source paths never cross IPC."""
        size = source.stat().st_size
        with source.open("rb") as stream:
            def chunks():
                """Observe the owning request before each finite read without abandoning a partial output file."""
                while True:
                    check_cancelled()
                    block = stream.read(1 << 20)
                    if not block:
                        return
                    yield block

            return self._write(chunks(), size, format)

    def _write(self, chunks: Iterator[bytes], expected_bytes: int, format: str) -> RuntimeArtifact:
        """Atomically materialize a pre-budgeted input/output and retain prior complete grants on failure."""
        if expected_bytes < 1 or expected_bytes > self.resources.artifact_bytes:
            raise ValueError("runtime_artifact_byte_budget")
        if self.total_bytes() + expected_bytes > self.resources.total_artifact_bytes:
            raise ValueError("runtime_total_artifact_budget")
        artifact_id = uuid4()
        pending = self.directory / f"{artifact_id}.pending"
        target = self.directory / f"{artifact_id}.blob"
        digest = hashlib.sha256()
        written = 0
        try:
            with pending.open("xb") as output:
                for chunk in chunks:
                    written += len(chunk)
                    if written > expected_bytes:
                        raise ValueError("runtime_artifact_changed")
                    digest.update(chunk)
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            if written != expected_bytes:
                raise ValueError("runtime_artifact_changed")
            artifact = RuntimeArtifact(
                artifact_id=artifact_id, sha256=digest.hexdigest(), size_bytes=written, format=format,
            )
            os.chmod(pending, 0o400)
            os.replace(pending, target)
            sync_directory(self.directory)
            return artifact
        finally:
            pending.unlink(missing_ok=True)

    def verify(self, artifact: RuntimeArtifact) -> Path:
        """Verify regular-file identity, length and SHA before any parent adoption or worker load."""
        path = self._path(artifact)
        descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
        digest = hashlib.sha256()
        try:
            info = os.fstat(descriptor)
            if not stat.S_ISREG(info.st_mode) or info.st_size != artifact.size_bytes:
                raise ValueError("runtime_artifact_identity")
            if info.st_size > self.resources.artifact_bytes:
                raise ValueError("runtime_artifact_byte_budget")
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
        finally:
            os.close(descriptor)
        if digest.hexdigest() != artifact.sha256:
            raise ValueError("runtime_artifact_digest")
        return path

    def read_bytes(self, artifact: RuntimeArtifact) -> bytes:
        """Read a verified bounded grant; large training inputs should use verified_path instead."""
        if artifact.size_bytes > 8 << 20:
            raise ValueError("runtime_in_memory_artifact_budget")
        return self.verify(artifact).read_bytes()

    def remove(self, artifact: RuntimeArtifact) -> None:
        """Retire a session grant after its owning consumer has adopted or discarded it."""
        self._path(artifact).unlink(missing_ok=True)


def sync_directory(directory: Path) -> None:
    """Make atomic publication durable by syncing its parent directory descriptor."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
