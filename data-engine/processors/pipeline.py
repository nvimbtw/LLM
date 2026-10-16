"""Running an ordered list of processors over documents.

Order is the user's, not ours, and it matters: stripping HTML before measuring
length keeps a page of markup from passing a minimum it only met in tags.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict

from processors.spec import Document, Processor, State


class Provenance(TypedDict):
    processor: str
    params: dict
    version_hash: str
    executed_version_hash: str
    version_mismatch: bool


@dataclass(frozen=True)
class Step:
    processor: Processor
    params: dict
    submitted_version_hash: str | None = None

    def provenance(self) -> Provenance:
        """Describe submitted and executing versions, including detected source drift."""
        submitted = self.submitted_version_hash or self.processor.version_hash
        return {
            "processor": self.processor.name,
            "params": self.params,
            "version_hash": submitted,
            "executed_version_hash": self.processor.version_hash,
            "version_mismatch": submitted != self.processor.version_hash,
        }


@dataclass
class Outcome:
    """What a pipeline did to one document."""

    before: Document
    after: Document | None
    #: The processor that dropped it, when one did.
    dropped_by: str | None = None

    @property
    def dropped(self) -> bool:
        return self.after is None

    @property
    def changed(self) -> bool:
        return self.after != self.before


def resolve(steps: list[dict], registry: Mapping[str, Processor]) -> list[Step]:
    """Turn the wire's `[{processor, params}, …]` into runnable steps.

    An unknown processor name raises rather than being skipped: a pipeline that
    quietly does less than it was asked to would produce a dataset whose
    recorded provenance is a lie.
    """
    resolved = []
    for step in steps:
        name = step.get("processor")
        processor = registry.get(str(name))
        if processor is None:
            raise KeyError(f"unknown processor: {name}")
        version = step.get("version_hash")
        if version is not None and (
            not isinstance(version, str)
            or len(version) != 64
            or any(ch not in "0123456789abcdef" for ch in version)
        ):
            raise ValueError("version_hash must be a SHA256 hex digest")
        resolved.append(Step(processor, processor.resolve(step.get("params")), version))
    return resolved


def run(steps: list[Step], doc: Document, state: State) -> Outcome:
    """Apply every step in turn, stopping at the first one that drops it."""
    current: Document = doc
    for step in steps:
        result = step.processor.apply(current, step.params, state)
        if result is None:
            return Outcome(before=doc, after=None, dropped_by=step.processor.name)
        current = result
    return Outcome(before=doc, after=current)
