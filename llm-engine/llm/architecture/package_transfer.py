"""Uncompressed bounded model_package_v1 containers for private runtime artifact grants."""

import os
from pathlib import Path
import shutil
import stat
import tarfile
import tempfile
from collections.abc import Callable
from typing import BinaryIO

from llm.architecture import package
from llm.architecture.package_integrity import (
    PackageBudget, enumerate_files, package_file, publish_directory, sync_directory,
)


def archive_limit(budget: PackageBudget) -> int:
    """Bound fixed USTAR headers/padding/trailer in addition to the admitted complete file payload."""
    return budget.total_bytes + (budget.files + 1) * 1024 + tarfile.RECORDSIZE


class ArchiveInput:
    """Expose cancellable bounded archive reads without changing declared member sizes or container bytes."""

    def __init__(self, stream: BinaryIO, check_cancelled: Callable[[], None]):
        """Borrow an already owned regular input stream and its request cancellation boundary."""
        self.stream, self.check_cancelled = stream, check_cancelled

    def read(self, size: int) -> bytes:
        """Observe cancellation before tarfile's finite read; the surrounding publication owner removes partial output."""
        self.check_cancelled()
        return self.stream.read(size)


def write_archive(directory: Path, destination: Path, *, budget: PackageBudget = PackageBudget(),
                  check_cancelled: Callable[[], None] = lambda: None) -> Path:
    """Snapshot a verified model package into a data-only USTAR runtime grant with atomic file publication.

    Containers are uncompressed so declared sizes bound disk and copy work before
    reading payloads. Package file names are engine-owned short UUID/hash paths.
    """
    check_cancelled()
    checked = package.verify(directory, budget=budget, check_cancelled=check_cancelled)
    if destination.resolve().is_relative_to(directory.resolve()):
        raise ValueError("package container must be outside its source directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".pending", dir=destination.parent)
    pending = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            with tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
                for name in sorted(set(checked.integrity.files) | {"integrity.json"}):
                    check_cancelled()
                    path = package_file(directory, name)
                    item = tarfile.TarInfo(name)
                    item.size = path.stat().st_size
                    item.mode = 0o400
                    with path.open("rb") as source:
                        archive.addfile(item, ArchiveInput(source, check_cancelled))
            output.flush()
            os.fsync(output.fileno())
        os.replace(pending, destination)
        sync_directory(destination.parent)
    finally:
        pending.unlink(missing_ok=True)
    return destination


def copy_member(source, target: Path, size_bytes: int) -> None:
    """Copy exactly one pre-budgeted regular member without extraction helpers or unbounded reads."""
    target.parent.mkdir(parents=True, exist_ok=True)
    remaining = size_bytes
    with target.open("xb") as output:
        while remaining:
            chunk = source.read(min(1 << 20, remaining))
            if not chunk:
                raise ValueError("truncated package container member")
            remaining -= len(chunk)
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    padding = source.read((-size_bytes) % tarfile.BLOCKSIZE)
    if len(padding) != (-size_bytes) % tarfile.BLOCKSIZE or any(padding):
        raise ValueError("invalid package container padding")


def unpack_members(source, destination: Path, budget: PackageBudget) -> None:
    """Read fixed-size USTAR headers and admit count/size/path/type before allocating or copying data."""
    total = 0
    seen = set()
    while True:
        header = source.read(tarfile.BLOCKSIZE)
        if len(header) != tarfile.BLOCKSIZE:
            raise ValueError("truncated package container header")
        if not any(header):
            tail = source.read(tarfile.RECORDSIZE + 1)
            if (len(tail) < tarfile.BLOCKSIZE or len(tail) % tarfile.BLOCKSIZE
                    or any(tail) or len(tail) > tarfile.RECORDSIZE or source.read(1)):
                raise ValueError("invalid package container trailer")
            return
        if header[257:265] != b"ustar\x0000":
            raise ValueError("package container requires fixed USTAR headers")
        item = tarfile.TarInfo.frombuf(header, "utf-8", "strict")
        if item.type not in (tarfile.REGTYPE, tarfile.AREGTYPE) or item.linkname or item.name in seen:
            raise ValueError("package container contains links, extensions or duplicate paths")
        if item.size < 0 or item.size > budget.file_bytes:
            raise ValueError("package container member exceeds byte budget")
        seen.add(item.name)
        total += item.size
        if len(seen) > budget.files + 1 or total > budget.total_bytes:
            raise ValueError("package container exceeds aggregate resource budget")
        copy_member(source, package_file(destination, item.name), item.size)


def materialize(archive: Path, destination: Path, *, budget: PackageBudget = PackageBudget()) -> package.VerifiedPackage:
    """Verify a private granted container into one immutable worker-local package before any source load.

    The caller verifies its RuntimeArtifact grant first. No archive path or
    destination comes from the browser. Failure removes all partial staging.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("package materialization destination already exists")
    descriptor = os.open(archive, os.O_RDONLY | os.O_NOFOLLOW)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".package-", dir=destination.parent))
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > archive_limit(budget):
            raise ValueError("package container exceeds byte budget")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            unpack_members(source, staging, budget)
        package.verify(staging, budget=budget)
        package.sync_tree(staging)
        publish_directory(staging, destination)
        sync_directory(destination.parent)
        return package.verify(destination, budget=budget)
    finally:
        os.close(descriptor)
        if staging.exists():
            shutil.rmtree(staging)
