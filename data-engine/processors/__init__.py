"""The curated set of transformations a dataset can be cleaned with.

One registry, assembled from the modules that group them by what they do to a
document: `text` rewrites prose, `filter` decides what to keep, `chat` reshapes
and renders conversations, `markdown` converts inline emphasis, and `hermes` and
`smoltalk` convert one named source each. `markup` holds no processors — it is
the block renderer the converters share.
"""

from pathlib import Path

from processors.chat import PROCESSORS as CONVERSATION
from processors.filter import PROCESSORS as SELECTION
from processors.hermes import PROCESSORS as HERMES
from processors.markdown import PROCESSORS as MARKUP
from processors.pipeline import Outcome, Step, resolve, run
from processors.smoltalk import PROCESSORS as CONVERTERS
from processors.spec import Document, Param, Processor, State
from processors.text import PROCESSORS as REWRITING
from processors.loader import Catalog, Watcher, with_source
from store.schema import DatasetKind

__all__ = [
    "Document",
    "Outcome",
    "Param",
    "Processor",
    "State",
    "Step",
    "REGISTRY",
    "for_kind",
    "resolve",
    "run",
]


def _registry() -> dict[str, Processor]:
    """Build the curated catalogue with source hashes, rejecting duplicate names."""
    registry: dict[str, Processor] = {}
    for processor in (
        *REWRITING,
        *SELECTION,
        *CONVERSATION,
        *MARKUP,
        *CONVERTERS,
        *HERMES,
    ):
        if processor.name in registry:
            raise ValueError(f"two processors named {processor.name}")
        registry[processor.name] = with_source(processor)
    return registry


REGISTRY = Catalog(_registry(), Path(__file__).parent / "custom")


def reload() -> None:
    """Reload custom files, retaining the last working entries for invalid edits."""
    REGISTRY.reload()


def for_kind(kind: DatasetKind) -> list[Processor]:
    """The processors that can be offered for a dataset of this kind, in the
    order they are declared — which is roughly the order they want running."""
    return [p for p in REGISTRY.snapshot().values() if kind in p.applies_to]
