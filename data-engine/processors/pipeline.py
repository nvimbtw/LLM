"""Running an ordered list of processors over documents.

Order is the user's, not ours, and it matters: stripping HTML before measuring
length keeps a page of markup from passing a minimum it only met in tags.
"""

from dataclasses import dataclass

from processors.spec import Document, Processor, State


@dataclass(frozen=True)
class Step:
    processor: Processor
    params: dict


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


def resolve(steps: list[dict], registry: dict[str, Processor]) -> list[Step]:
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
        resolved.append(Step(processor, processor.resolve(step.get("params"))))
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
