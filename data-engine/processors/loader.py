"""Atomic catalogues for trusted operator files, with import policy and active hashes.

The import policy catches accidental storage/service access. Python executes in
process; this is not a security sandbox for untrusted code.
"""

import ast
import builtins
import hashlib
import inspect
import json
import logging
import re
import sys
import threading
from collections.abc import Iterator, Mapping
from dataclasses import asdict, replace
from itertools import islice
from pathlib import Path
from types import MappingProxyType, ModuleType

from processors.spec import Param, Processor

LOGGER = logging.getLogger(__name__)
MAX_SOURCE_BYTES = 1_048_576
MAX_CUSTOM_FILES = 128
POLL_SECONDS = 0.2
DEBOUNCE_SECONDS = 0.2
ALLOWED_IMPORTS = frozenset(
    {
        "re",
        "html",
        "unicodedata",
        "json",
        "urllib.parse",
        "dataclasses",
        "math",
        "collections",
        "itertools",
        "processors.helpers",
    }
)
SAFE_BUILTINS = (
    "abs",
    "all",
    "any",
    "bool",
    "bytes",
    "callable",
    "chr",
    "dict",
    "enumerate",
    "filter",
    "float",
    "frozenset",
    "int",
    "isinstance",
    "issubclass",
    "iter",
    "len",
    "list",
    "map",
    "max",
    "min",
    "next",
    "object",
    "ord",
    "range",
    "repr",
    "reversed",
    "round",
    "set",
    "slice",
    "sorted",
    "str",
    "sum",
    "tuple",
    "type",
    "zip",
    "Exception",
    "ValueError",
    "TypeError",
    "RuntimeError",
    "__build_class__",
    "property",
    "staticmethod",
    "classmethod",
    "super",
)
PROCESSOR_NAME = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,99}$")


def _validate_metadata(entry: Processor) -> None:
    """Reject metadata that cannot safely populate the catalogue and parameter forms."""
    if not isinstance(entry.title, str) or not 1 <= len(entry.title.strip()) <= 150:
        raise ValueError("processor title must contain 1–150 characters")
    if not isinstance(entry.description, str):
        raise ValueError("processor description must be text")
    defaults = {"int": (int,), "float": (int, float), "bool": (bool,), "string": (str,)}
    names = set()
    for param in entry.params:
        if (
            not isinstance(param, Param)
            or not isinstance(param.name, str)
            or not param.name
        ):
            raise ValueError("processor parameters must be named Param declarations")
        if param.name in names:
            raise ValueError(f"duplicate parameter: {param.name}")
        names.add(param.name)
        if not isinstance(param.label, str) or not isinstance(param.description, str):
            raise ValueError("parameter labels and descriptions must be text")
        if type(param.default) not in defaults.get(param.type, ()):
            raise ValueError(f"invalid default or type for parameter: {param.name}")
        if any(
            bound is not None and type(bound) not in (int, float)
            for bound in (param.minimum, param.maximum)
        ):
            raise ValueError(f"invalid range for parameter: {param.name}")
        if (
            param.minimum is not None
            and param.maximum is not None
            and param.minimum > param.maximum
        ):
            raise ValueError(f"inverted range for parameter: {param.name}")
    json.dumps([asdict(param) for param in entry.params], allow_nan=False)


def source_bytes(path: Path) -> bytes:
    """Read one regular source file within the byte budget; reject symlinks."""
    if path.is_symlink():
        raise ValueError("processor source must not be a symlink")
    with path.open("rb") as source:
        content = source.read(MAX_SOURCE_BYTES + 1)
    if len(content) > MAX_SOURCE_BYTES:
        raise ValueError(f"processor source exceeds {MAX_SOURCE_BYTES} bytes")
    return content


def with_source(processor: Processor) -> Processor:
    """Attach a builtin's module path and hash to its loaded registry entry."""
    path = Path(processor.file or inspect.getfile(processor.apply)).resolve()
    return replace(
        processor,
        file=str(path),
        version_hash=hashlib.sha256(source_bytes(path)).hexdigest(),
    )


def _import(name: str, globals=None, locals=None, fromlist=(), level: int = 0):
    """Permit only declared absolute imports from custom processor code."""
    if level or name not in ALLOWED_IMPORTS:
        raise ImportError(f"processor import is not allowed: {name}")
    return builtins.__import__(name, globals, locals, fromlist, level)


def _lint(tree: ast.AST) -> None:
    """Reject storage imports and direct escape builtins before executing a file."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name not in ALLOWED_IMPORTS:
                    raise ImportError(f"processor import is not allowed: {alias.name}")
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS:
                raise ImportError(f"processor import is not allowed: {node.module}")
            if any(
                alias.name == "*" or alias.name.startswith("_") for alias in node.names
            ):
                raise ImportError(
                    "wildcard and private processor imports are not allowed"
                )
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            raise ValueError("private attributes are not allowed in custom processors")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise ValueError("private builtins are not allowed in custom processors")


def load_file(path: Path, content: bytes) -> Processor:
    """Compile exactly these bytes into one processor, or raise a load error."""
    tree = ast.parse(content, filename=str(path))
    _lint(tree)
    digest = hashlib.sha256(content).hexdigest()
    module_name = f"processors.custom.{path.stem}_{digest}"
    module = ModuleType(module_name)
    module.__dict__.update(
        {
            "__file__": str(path),
            "__builtins__": {
                **{key: getattr(builtins, key) for key in SAFE_BUILTINS},
                "__import__": _import,
            },
        }
    )
    sys.modules[module_name] = module
    try:
        exec(compile(tree, str(path), "exec"), module.__dict__)
    finally:
        sys.modules.pop(module_name, None)
    entries = {
        id(value): value
        for value in module.__dict__.values()
        if isinstance(value, Processor)
    }
    if len(entries) != 1:
        raise ValueError("a custom file must declare exactly one processor")
    processor = next(iter(entries.values()))
    if not PROCESSOR_NAME.fullmatch(processor.name):
        raise ValueError("processor name must be a short URL-safe identifier")
    if processor.name == "new-processor":
        raise ValueError("new-processor is reserved for the creation page")
    _validate_metadata(processor)
    return replace(processor, file=str(path), version_hash=digest)


class Catalog(Mapping[str, Processor]):
    """Own registry generations, quarantined-load errors and a serialized reload lock."""

    def __init__(self, builtins: Mapping[str, Processor], directory: Path) -> None:
        """Load custom files alongside immutable builtins; startup collisions raise."""
        self.directory = directory.resolve()
        self._builtins = dict(builtins)
        self._registry = MappingProxyType(dict(builtins))
        self._files: dict[Path, Processor] = {}
        self._observed: dict[Path, str] = {}
        self._errors: dict[Path, str] = {}
        self._reload_error: str | None = None
        self.lock = threading.RLock()
        self.reload(startup=True)

    def __getitem__(self, name: str) -> Processor:
        """Return a processor from the current complete generation, or raise KeyError."""
        return self._registry[name]

    def __iter__(self) -> Iterator[str]:
        """Iterate identities in one immutable generation."""
        return iter(self._registry)

    def __len__(self) -> int:
        """Return the current catalogue size."""
        return len(self._registry)

    def snapshot(self) -> Mapping[str, Processor]:
        """Return one immutable generation for a request or running pipeline."""
        return self._registry

    def error(self, path: Path) -> str | None:
        """Return the last reload failure for a saved file, when present."""
        return self._errors.get(path) or self._reload_error

    def editable(self, entry: Processor) -> bool:
        """Allow writes only to known custom files directly inside this catalogue."""
        return (
            entry.name not in self._builtins
            and Path(entry.file).parent == self.directory
        )

    def paths(self) -> list[Path]:
        """List bounded custom sources; helper package files are excluded."""
        candidates = (
            path for path in self.directory.glob("*.py") if path.name != "__init__.py"
        )
        paths = sorted(islice(candidates, MAX_CUSTOM_FILES + 1))
        if len(paths) > MAX_CUSTOM_FILES:
            raise ValueError(
                f"at most {MAX_CUSTOM_FILES} custom processor files are allowed"
            )
        return paths

    def _entry(
        self, path: Path, errors: dict[Path, str], observed: dict[Path, str]
    ) -> Processor | None:
        """Load a changed file or retain its last working entry after a failed edit."""
        try:
            content = source_bytes(path)
            digest = hashlib.sha256(content).hexdigest()
            observed[path] = digest
            if self._observed.get(path) == digest:
                if path in self._errors:
                    errors[path] = self._errors[path]
                return self._files.get(path)
            entry = load_file(path, content)
            previous = self._files.get(path)
            if previous is not None and previous.name != entry.name:
                raise ValueError("an edited processor must retain its registered name")
            return entry
        except Exception as error:
            errors[path] = f"{type(error).__name__}: {error}"
            LOGGER.warning("Quarantined processor %s: %s", path.name, errors[path])
            return self._files.get(path)

    def reload(self, *, startup: bool = False) -> None:
        """Publish a complete candidate registry; collisions retain the previous generation.

        Startup collisions raise ValueError. Invalid files are quarantined logically
        without moving their bytes, so an operator can repair the saved source.
        """
        with self.lock:
            try:
                paths = self.paths()
            except Exception as error:
                if startup:
                    raise
                self._reload_error = f"{type(error).__name__}: {error}"
                LOGGER.warning("Processor reload rejected: %s", self._reload_error)
                return
            registry = dict(self._builtins)
            files, errors, observed = {}, {}, {}
            for path in paths:
                entry = self._entry(path, errors, observed)
                if entry is None:
                    continue
                if entry.name in registry:
                    message = f"two processors named {entry.name} ({path.name})"
                    if startup:
                        raise ValueError(message)
                    errors[path] = message
                    self._errors = errors
                    self._reload_error = message
                    LOGGER.warning("Processor reload rejected: %s", message)
                    return
                registry[entry.name] = entry
                files[path] = entry
            self._files, self._errors, self._observed = files, errors, observed
            self._reload_error = None
            self._registry = MappingProxyType(registry)


class Watcher:
    """One stoppable polling thread owned and drained by the service lifespan."""

    def __init__(self, catalog: Catalog) -> None:
        """Create a watcher for one catalogue without starting background work."""
        self.catalog = catalog
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=self._watch, name="processor-watcher", daemon=True
        )

    def start(self) -> None:
        """Start polling custom source files until stop() joins the thread."""
        self._thread.start()

    def stop(self) -> None:
        """Request shutdown and join any in-flight reload before returning."""
        self._stop.set()
        self._thread.join()

    def _signature(self) -> tuple:
        """Return filesystem change indicators without executing custom code."""
        return tuple(
            (str(path), path.stat().st_mtime_ns, path.stat().st_size)
            for path in self.catalog.paths()
        )

    def _watch(self) -> None:
        """Reload after 200 ms of stable filesystem state; log failures and keep polling."""
        previous = None
        while not self._stop.wait(POLL_SECONDS):
            try:
                current = self._signature()
                if current != previous:
                    if self._stop.wait(DEBOUNCE_SECONDS):
                        return
                    stable = self._signature()
                    if stable != current:
                        continue
                    self.catalog.reload()
                    previous = stable
            except Exception:
                LOGGER.exception("Processor watcher could not reload sources")
