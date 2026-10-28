"""Retained validation text is independent of validity, mutable catalogs and installed code."""

import hashlib
from uuid import uuid4

import pytest

from llm.architecture.builtins.catalog import builtin_definitions, readable_source
from llm.architecture.errors import ArchitectureError
from llm.architecture.spec import RevisionRef, RevisionSubject
from llm.architecture.validation_artifacts import PublishedValidation
from llm.architecture.validation_records import PinnedSource, ValidationOutput, ValidationSnapshot
from llm.architecture.wire import ValidationReport, ValidationRequest
from tests.test_architecture_validation import source_snapshot


def invalid_publication(snapshot: ValidationSnapshot) -> PublishedValidation:
    """Build an explicitly invalid data-only publication; this fixture claims no construction or capture evidence."""
    return PublishedValidation(snapshot=snapshot, output=ValidationOutput(report=ValidationReport(
        schema_version=1, run_id=snapshot.run_id, verdict="invalid", stages=(), diagnostics=())))


@pytest.mark.parametrize("kind", ["block", "architecture"])
def test_invalid_dependency_text_remains_exactly_readable(kind):
    """A diagnostic can open frozen invalid Python/YAML without requiring acceptance or reparsing the text."""
    source = "invalid Python/YAML ž 🎯 [\n"
    reference = RevisionRef(kind=kind, resource_id=uuid4(), revision_id=uuid4(),
        content_hash=hashlib.sha256(source.encode("utf-8")).hexdigest())
    original = source_snapshot(inspect=True)
    snapshot = ValidationSnapshot(**{**original.model_dump(),
        "dependencies": (PinnedSource(reference=reference, source=source),)})
    retained = invalid_publication(snapshot).source(reference.revision_id)
    assert retained.source == source
    assert retained.content_hash == reference.content_hash
    assert retained.language == ("python" if kind == "block" else "yaml")
    assert retained.definition_id is None


def test_builtin_root_reads_snapshot_bytes_and_never_confuses_manifest_with_raw_source_hash(monkeypatch):
    """Builtin identity hashes a manifest while source integrity hashes the retained Python bytes; no installed read occurs on presentation."""
    definitions = tuple(builtin_definitions().values())
    reference = definitions[0].reference
    source = readable_source(reference)
    snapshot = ValidationSnapshot(run_id=uuid4(), source=source, builtin_root=reference,
        builtins=tuple(item.reference for item in definitions),
        request=ValidationRequest(schema_version=1, request_id=uuid4(), mode="inspect",
            subject=RevisionSubject(kind="revision", reference=reference)))
    publication = invalid_publication(snapshot)

    def forbidden(*args, **kwargs):
        """Reject any attempt to substitute current code or execute a source read."""
        pytest.fail("retained source presentation attempted installed/source execution")

    monkeypatch.setattr("llm.architecture.builtins.catalog.readable_source", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    retained = publication.source(reference.revision_id)
    assert retained.source == source
    assert retained.content_hash == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert retained.content_hash != reference.content_hash
    with pytest.raises(ArchitectureError, match="not retained"):
        publication.source(definitions[1].reference.revision_id)
