"""Read and atomically save known custom processor files under catalogue ownership."""

import hashlib
import os
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from processors.loader import (
    MAX_CUSTOM_FILES,
    MAX_SOURCE_BYTES,
    Catalog,
    load_file,
    source_bytes,
)


@dataclass(frozen=True)
class Source:
    """Saved source identity alongside the separately active callback identity."""

    name: str
    title: str
    file: str
    version_hash: str
    active_version_hash: str
    source: str
    editable: bool
    reload_error: str | None


def read(catalog: Catalog, name: str) -> Source:
    """Read saved UTF-8 bytes and active metadata consistently; unknown names raise KeyError."""
    with catalog.lock:
        entry = catalog[name]
        path = Path(entry.file)
        content = source_bytes(path)
        editable = catalog.editable(entry)
        return Source(
            name=entry.name,
            title=entry.title,
            file=f"processors/{'custom/' if editable else ''}{path.name}",
            version_hash=hashlib.sha256(content).hexdigest(),
            active_version_hash=entry.version_hash,
            source=content.decode("utf-8"),
            editable=editable,
            reload_error=catalog.error(path),
        )


def _replace(path: Path, content: bytes) -> None:
    """Replace source atomically, preserving permissions and cleaning temporary files."""
    original = path.stat()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, suffix=".tmp", delete=False
        ) as output:
            temporary = Path(output.name)
            _file_metadata(output.fileno(), original, stat.S_IMODE(original.st_mode))
            output.write(content)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _file_metadata(descriptor: int, owner: os.stat_result, mode: int) -> None:
    """Preserve host ownership when a root container writes an operator file."""
    if os.geteuid() == 0:
        os.fchown(descriptor, owner.st_uid, owner.st_gid)
    os.fchmod(descriptor, mode)


def _sync_directory(directory: Path) -> None:
    """Flush a source directory's entry changes before acknowledging a save."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _encoded_source(text: str) -> bytes:
    """Validate the UTF-8 byte budget before writing operator source."""
    content = text.encode("utf-8")
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(f"processor source exceeds {MAX_SOURCE_BYTES} UTF-8 bytes")
    return content


def create(catalog: Catalog, text: str) -> Source:
    """Create a new processor from valid source, deriving its identity from the decorator.

    Unlike edits, creation needs a working first version. Collisions raise
    FileExistsError without overwriting bytes; failed publication removes only
    this operation's newly created file. The detail route reserves new-processor.
    """
    content = _encoded_source(text)
    with catalog.lock:
        try:
            entry = load_file(catalog.directory / "new_processor.py", content)
        except Exception as error:
            raise ValueError(f"{type(error).__name__}: {error}") from error
        path = catalog.directory / f"{entry.name}.py"
        if entry.name in catalog or path.exists() or path.is_symlink():
            raise FileExistsError(f"a processor named {entry.name} already exists")
        if len(catalog.paths()) >= MAX_CUSTOM_FILES:
            raise ValueError("custom processor file limit reached")
        output = path.open("xb")
        try:
            with output:
                _file_metadata(output.fileno(), catalog.directory.stat(), 0o644)
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            _sync_directory(path.parent)
            catalog.reload()
            if entry.name not in catalog:
                raise ValueError(
                    catalog.error(path) or "processor could not be activated"
                )
            return read(catalog, entry.name)
        except Exception:
            path.unlink(missing_ok=True)
            _sync_directory(path.parent)
            catalog.reload()
            raise


def save(catalog: Catalog, name: str, text: str) -> Source:
    """Persist bytes as supplied, then reload; invalid code retains the active entry.

    Builtins raise PermissionError. Size, path and UTF-8 are transport/storage
    boundaries; Python syntax and import policy are handled only after saving.
    """
    content = _encoded_source(text)
    with catalog.lock:
        entry = catalog[name]
        if not catalog.editable(entry):
            raise PermissionError("builtin processors are read-only")
        path = Path(entry.file)
        if path.is_symlink():
            raise PermissionError("processor source must not be a symlink")
        _replace(path, content)
        catalog.reload()
        return read(catalog, name)
