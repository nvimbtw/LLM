"""What a processor is.

Curated processors and trusted operator files use the same Apply contract.
Administrators may edit known custom files through Rust's authenticated source
routes. Custom code executes in process under an import policy, so this is an
operator authoring interface, not an untrusted-code upload sandbox.
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from store.schema import DatasetKind

#: A document in flight: `content` for prose, `messages` for a conversation,
#: plus whatever `meta` came off the Hub. The same shape `store.queries`
#: writes, so a pipeline's output is insertable without translation.
Document = dict[str, Any]

#: Per-run scratch space, handed to every step. A processor that must remember
#: something across documents — a seen-hash set — keeps it here rather than in a
#: module global, so two runs cannot contaminate each other and a preview always
#: starts clean.
State = dict[str, Any]

#: Returns the document, changed or not, or `None` to drop it.
Apply = Callable[[Document, dict, State], Document | None]

ParamType = Literal["int", "float", "bool", "string"]


@dataclass(frozen=True)
class Param:
    name: str
    label: str
    type: ParamType
    default: Any
    description: str = ""
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class Processor:
    """One named transformation, and enough about it to build a form."""

    name: str
    title: str
    description: str
    applies_to: tuple[DatasetKind, ...]
    apply: Apply
    params: tuple[Param, ...] = field(default_factory=tuple)
    file: str = ""
    version_hash: str = ""

    def defaults(self) -> dict:
        """Return the declared parameter defaults for form initialization."""
        return {p.name: p.default for p in self.params}

    def resolve(self, given: dict | None) -> dict:
        """Declared parameters only, defaulted and range-checked.

        Silently ignoring an unknown key is deliberate: the frontend and this
        registry version separately, and a stale form field must not fail a run
        that is otherwise valid.
        """
        supplied = given or {}
        resolved = {}
        for param in self.params:
            resolved[param.name] = _coerce(
                param, supplied.get(param.name, param.default)
            )
        return resolved


def _coerce(param: Param, value: Any) -> Any:
    """A form value as the processor's type, clamped to the declared range.

    Clamped rather than rejected: these arrive from a slider the user is still
    dragging, and a preview that errors mid-drag is worse than one that shows
    the nearest legal answer.
    """
    try:
        if param.type == "int":
            value = int(value)
        elif param.type == "float":
            value = float(value)
        elif param.type == "bool":
            value = bool(value)
        else:
            value = str(value)
    except (TypeError, ValueError):
        return param.default

    if param.type in ("int", "float"):
        if param.minimum is not None:
            value = max(value, type(value)(param.minimum))
        if param.maximum is not None:
            value = min(value, type(value)(param.maximum))
    return value
