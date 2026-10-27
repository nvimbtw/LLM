"""Bounded immutable worker inputs/results for architecture validation, outside HTTP/storage."""

import hashlib
import json
from typing import Annotated, Literal
from uuid import UUID, uuid5

from pydantic import Field, model_validator

from llm.architecture.spec import Contract, MAX_DEPENDENCIES, MAX_SOURCE_BYTES, MAX_YAML_BYTES, RevisionRef, Sha256
from llm.architecture.wire import ValidationReport, ValidationRequest
from llm.architecture.dependencies import recipe_dependencies
from llm.architecture.parse import parse_recipe
from llm.architecture.source import imports_public_builtins

VALIDATION_PHASES = ("inspection","resolution","construction","capture","numerical","publication")


def canonical_bytes(value: dict) -> bytes:
    """Serialize canonical JSON so raw-byte SHA agrees with the shared canonical digest contract."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


class PinnedSource(Contract):
    """Exact immutable custom source or subrecipe bytes checked before import/resolution."""

    reference: RevisionRef
    source: str

    @model_validator(mode="after")
    def exact_source(self):
        """Require bounded UTF-8 bytes with a matching revision content hash."""
        raw = self.source.encode("utf-8")
        if len(raw) > MAX_SOURCE_BYTES or hashlib.sha256(raw).hexdigest() != self.reference.content_hash:
            raise ValueError("dependency source byte budget/hash mismatch")
        return self


class ValidationSnapshot(Contract):
    """Parent-captured request and closure; no mutable catalog lookups happen inside the worker."""

    schema_version: Literal[1] = 1
    run_id: UUID
    request: ValidationRequest
    source: str | None = None
    recipe: str | None = None
    builtin_root: RevisionRef | None = None
    dependencies: Annotated[tuple[PinnedSource, ...], Field(max_length=MAX_DEPENDENCIES)] = ()
    builtins: Annotated[tuple[RevisionRef, ...], Field(max_length=MAX_DEPENDENCIES)] = ()

    @model_validator(mode="after")
    def exact_subject(self):
        """Keep saved bytes/profile subject exact and reject ambiguous or oversized work snapshots."""
        if (self.source is None) == (self.recipe is None):
            raise ValueError("provide exactly one source or recipe")
        raw = (self.source if self.source is not None else self.recipe).encode("utf-8")
        if len(raw) > (MAX_SOURCE_BYTES if self.source is not None else MAX_YAML_BYTES):
            raise ValueError("subject source byte budget exceeded")
        subject = self.request.subject
        expected = subject.content_hash if subject.kind == "draft" else subject.reference.content_hash
        if self.builtin_root is not None:
            resource_id = subject.resource_id if subject.kind == "draft" else subject.reference.resource_id
            if self.source is None or self.builtin_root.kind != "block" or self.builtin_root.resource_id != resource_id:
                raise ValueError("builtin validation root must match its block subject")
            if subject.kind == "revision" and subject.reference != self.builtin_root:
                raise ValueError("builtin validation revision differs from its frozen implementation")
        if not (self.builtin_root is not None and subject.kind == "revision") and hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("subject source hash mismatch")
        references = [value.reference for value in self.dependencies] + list(self.builtins)
        if len(references) > MAX_DEPENDENCIES or len({value.revision_id for value in references}) != len(references):
            raise ValueError("duplicate/oversized dependency closure")
        block_sources = [value.source for value in self.dependencies if value.reference.kind == "block"]
        if self.source is not None:
            block_sources.append(self.source)
        if not self.builtins and any(imports_public_builtins(source) for source in block_sources):
            raise ValueError("public builtin source requires frozen implementation pins")
        return self


def subject_reference(snapshot: ValidationSnapshot) -> RevisionRef:
    """Give exact draft bytes a stable private import/location identity without accepting a catalog revision."""
    subject = snapshot.request.subject
    if subject.kind == "revision":
        return subject.reference
    return RevisionRef(kind="block" if snapshot.source is not None else "architecture",
        resource_id=subject.resource_id,
        revision_id=uuid5(subject.resource_id, f"draft:{subject.draft_version}:{subject.content_hash}"),
        content_hash=subject.content_hash)


def snapshot_dependencies(snapshot: ValidationSnapshot) -> dict[str,RevisionRef]:
    """Identify root recipe pins or direct public-builtin implementation pins for a copied source block."""
    if snapshot.recipe is not None:
        return recipe_dependencies(parse_recipe(snapshot.recipe))
    if snapshot.builtin_root is not None:
        return {"builtin":snapshot.builtin_root}
    return {f"builtins/{index}":reference for index,reference in
            enumerate(sorted(snapshot.builtins,key=lambda value:str(value.revision_id)))}


class ValidationOutput(Contract):
    """Complete typed report plus canonical graph blobs awaiting atomic parent publication."""

    schema_version: Literal[1] = 1
    report: ValidationReport
    graph_blobs: dict[Sha256, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def exact_artifacts(self):
        """Check every complete blob and receipt manifest reference before returning worker success."""
        total = 0
        for digest, source in self.graph_blobs.items():
            raw = source.encode("utf-8")
            total += len(raw)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("graph blob hash mismatch")
        if total > 8 << 20:
            raise ValueError("validation graph aggregate byte budget exceeded")
        receipt = self.report.receipt
        if receipt is not None and receipt.graph_manifest_hash is not None and receipt.graph_manifest_hash not in self.graph_blobs:
            raise ValueError("receipt graph manifest is absent")
        return self
