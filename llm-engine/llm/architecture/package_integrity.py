"""Package-local artifact inventories and bounded data-only integrity primitives."""

from dataclasses import dataclass
from collections.abc import Callable
import hashlib
import json
import os
from pathlib import Path
import stat
from typing import Annotated, Literal

from pydantic import Field, field_validator

from llm.architecture.spec import Contract, NonnegativeInt, Sha256, checked_relative_path

MANIFEST_BYTES = 4 << 20
METADATA_BYTES = 8 << 20
RECEIPT_BYTES = 8 << 20


@dataclass(frozen=True)
class PackageBudget:
    """Explicit parent-owned disk/copy budgets, separate from model execution admission."""

    total_bytes: int = 32 << 30
    file_bytes: int = 16 << 30
    files: int = 1024

    def __post_init__(self) -> None:
        """Reject ineffective or negative copy budgets before opening package artifacts."""
        if min(self.total_bytes, self.file_bytes, self.files) < 1 or self.files > 4096:
            raise ValueError("invalid package budget")


class FileIdentity(Contract):
    """Name exact published artifact bytes without loading tensors or Python objects."""

    sha256: Sha256
    size_bytes: NonnegativeInt


class PackageIntegrity(Contract):
    """Local storage metadata; not a duplicate public model-definition wire schema."""

    format: Literal["slate.package-integrity"] = "slate.package-integrity"
    version: Literal[1] = 1
    definition_id: Sha256
    torch_version: Annotated[str, Field(min_length=1, max_length=64)]
    files: Annotated[dict[str, FileIdentity], Field(min_length=1, max_length=4096)]

    @field_validator("files")
    @classmethod
    def validate_paths(cls, files: dict[str, FileIdentity]) -> dict[str, FileIdentity]:
        """Reject traversal, duplicate aliases and a recursively hashed integrity inventory."""
        for path in files:
            checked_relative_path(path)
            if path == "integrity.json":
                raise ValueError("integrity inventory cannot hash itself")
        return files


def encode_json(value: object) -> bytes:
    """Encode canonical UTF-8 JSON so content identities agree with version-one contracts."""
    if isinstance(value, Contract):
        value = value.model_dump(mode="json")
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate package JSON fields rather than silently changing interpreted identities."""
    result = {}
    for name, value in pairs:
        if name in result:
            raise ValueError("duplicate package JSON key")
        result[name] = value
    return result


def decode_json(data: bytes) -> object:
    """Decode data-only package JSON with duplicate and nonfinite number rejection."""
    return json.loads(data, object_pairs_hook=unique_fields, parse_constant=reject_constant)


def reject_constant(value: str) -> None:
    """Reject nonfinite constants that cannot participate in semantic package identity."""
    raise ValueError(f"nonfinite package JSON: {value}")


def package_file(directory: Path, relative: str) -> Path:
    """Resolve one package-local path, refusing links in the root or any existing component."""
    checked_relative_path(relative)
    current = directory
    if current.is_symlink():
        raise ValueError("package root cannot be a symbolic link")
    for component in relative.split("/"):
        current = current / component
        if current.is_symlink():
            raise ValueError("package artifacts cannot be symbolic links")
    return current


def read_bytes(path: Path, max_bytes: int) -> bytes:
    """Read one regular no-follow file only after checking its byte budget."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("package artifact exceeds byte budget or is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            data = source.read(max_bytes + 1)
        if len(data) > max_bytes:
            raise ValueError("package artifact changed beyond byte budget")
        return data
    finally:
        os.close(descriptor)


def file_identity(path: Path, max_bytes: int, *, check_cancelled: Callable[[], None] = lambda: None) -> FileIdentity:
    """Hash bounded regular bytes incrementally with optional owning-request cancellation and no tensor/source execution."""
    check_cancelled()
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    digest = hashlib.sha256()
    total = 0
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("package artifact exceeds byte budget or is not regular")
        with os.fdopen(descriptor, "rb", closefd=False) as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                check_cancelled()
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("package artifact changed beyond byte budget")
                digest.update(chunk)
    finally:
        os.close(descriptor)
    return FileIdentity(sha256=digest.hexdigest(), size_bytes=total)


def sync_directory(directory: Path) -> None:
    """Persist directory entries after atomic file or package publication."""
    descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_bytes(path: Path, data: bytes) -> None:
    """Write a new staging artifact and fsync it before its directory can be published."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())


def enumerate_files(directory: Path, budget: PackageBudget) -> set[str]:
    """Bound complete tree traversal and refuse links/special files or unaccounted artifact paths."""
    if directory.is_symlink():
        raise ValueError("package root cannot be a symbolic link")
    pending = [(directory, 0)]
    files = set()
    entries = 0
    while pending:
        current, depth = pending.pop()
        if depth > 16:
            raise ValueError("package path nesting exceeds budget")
        with os.scandir(current) as children:
            for child in children:
                entries += 1
                if entries > budget.files * 4 + 1 or child.is_symlink():
                    raise ValueError("package tree exceeds budget or contains links")
                path = Path(child.path)
                if child.is_dir(follow_symlinks=False):
                    pending.append((path, depth + 1))
                elif child.is_file(follow_symlinks=False):
                    files.add(path.relative_to(directory).as_posix())
                    if len(files) > budget.files + 1:
                        raise ValueError("package file count exceeds budget")
                else:
                    raise ValueError("package contains non-regular artifacts")
    return files


def publish_directory(source: Path, destination: Path) -> None:
    """Atomically rename a staged Linux directory without replacing even an empty destination."""
    import ctypes

    library = ctypes.CDLL(None, use_errno=True)
    rename = library.renameat2
    rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(source), -100, os.fsencode(destination), 1) != 0:
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code), str(destination))
