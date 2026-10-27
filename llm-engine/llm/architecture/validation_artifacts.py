"""Immutable parent-owned validation publication, independent of database transactions and custom imports."""

import hashlib
from pathlib import Path
import shutil
import tempfile
from typing import Literal
from uuid import UUID

from pydantic import model_validator

from llm import storage
from llm.architecture.dependencies import dependency_digest
from llm.architecture.errors import ArchitectureError
from llm.architecture.graphs import verify_receipt_graphs
from llm.architecture.package_integrity import package_file, publish_directory, read_bytes, sync_directory, write_bytes
from llm.architecture.parse import parse_recipe
from llm.architecture.resolve import parameter_values
from llm.architecture.spec import Contract, canonical_digest
from llm.architecture.validation_records import ValidationOutput, ValidationSnapshot, canonical_bytes, snapshot_dependencies, subject_reference
from llm.architecture.wire import RetainedSource

PUBLICATION_BYTES = 16 << 20


class PublishedValidation(Contract):
    """Retain exact source/profile snapshot, bounded report and complete graph bytes after verified cleanup."""

    schema_version: Literal[1] = 1
    snapshot: ValidationSnapshot
    output: ValidationOutput

    def source(self, revision_id: UUID) -> RetainedSource:
        """Read selected published snapshot text, including invalid drafts, without consulting mutable catalog or executing Python."""
        snapshot = self.snapshot
        root = subject_reference(snapshot)
        if revision_id == root.revision_id:
            reference, source = root, snapshot.source if snapshot.source is not None else snapshot.recipe
        else:
            entry = next((entry for entry in snapshot.dependencies if entry.reference.revision_id == revision_id), None)
            if entry is None:
                raise ArchitectureError("not_found", "source revision is not retained by this validation")
            reference, source = entry.reference, entry.source
        language = "python" if reference.kind == "block" else "yaml"
        return RetainedSource(revision_id=revision_id, content_hash=hashlib.sha256(source.encode("utf-8")).hexdigest(),
            file=f"{'blocks' if language == 'python' else 'architectures'}/{revision_id}.{'py' if language == 'python' else 'yaml'}",
            language=language, source=source)

    @model_validator(mode="after")
    def matching_result(self):
        """Reject another job/subject/profile or incomplete graph evidence before any publication/adoption."""
        snapshot, report = self.snapshot, self.output.report
        if report.run_id != snapshot.run_id:
            raise ValueError("validation output belongs to another job")
        if snapshot.request.mode == "inspect" and report.verdict not in {"inspection_only","invalid"}:
            raise ValueError("inspection cannot publish full validation evidence")
        receipt = report.receipt
        if receipt is None:
            if self.output.graph_blobs:
                raise ValueError("unvalidated report cannot publish capture graphs")
            return self
        identity, request = receipt.identity, snapshot.request
        if identity.subject != request.subject or identity.tokenizer != request.tokenizer:
            raise ValueError("validation receipt source/tokenizer differs from its queued snapshot")
        if request.profile is None or identity.profile.model_copy(update={"export_guards":()}) != request.profile.model_copy(update={"export_guards":()}):
            raise ValueError("validation receipt runtime differs from its queued profile")
        if request.profile.export_guards and request.profile.export_guards != identity.profile.export_guards:
            raise ValueError("validation receipt guards differ from supplied expectations")
        recipe = parse_recipe(snapshot.recipe) if snapshot.recipe is not None else None
        declarations = recipe.parameters if recipe else report.declaration.parameters if report.declaration else None
        if declarations is None:
            raise ValueError("validated block report omits its constructor declarations")
        parameters = parameter_values(declarations,request.parameters)
        context = recipe.context_window if recipe else request.context_window
        if isinstance(context,str):
            context = parameters[context[2:-1]]
        inputs = {"tensor_inputs":{name:value.model_dump(mode="json") for name,value in request.tensor_inputs.items()},
                  "domains":[domain.model_dump(mode="json") for domain in request.profile.domains]}
        if (identity.parameter_digest != canonical_digest(parameters) or identity.context_window != context
                or identity.dependency_digest != dependency_digest(snapshot_dependencies(snapshot))
                or identity.input_profile_digest != canonical_digest(inputs)):
            raise ValueError("validation receipt bindings differ from its exact queued snapshot")
        if report.verdict == "valid":
            verify_receipt_graphs(receipt,self.output.graph_blobs)
        elif self.output.graph_blobs or receipt.graph_manifest_hash is not None:
            raise ValueError("provisional validation cannot claim published full capture evidence")
        return self


def read_publication(run_id: UUID) -> PublishedValidation:
    """Verify complete bounded published bytes before serving graphs, accepting revisions or reconciling a job."""
    directory = storage.validation_dir(run_id)
    raw = read_bytes(package_file(directory,"validation.json"),PUBLICATION_BYTES)
    expected = read_bytes(package_file(directory,"sha256"),64).decode("ascii")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("published validation bytes are corrupt")
    publication = PublishedValidation.model_validate_json(raw)
    if publication.snapshot.run_id != run_id:
        raise ValueError("published validation directory belongs to another job")
    return publication


def publish_after_cleanup(snapshot: ValidationSnapshot, output: ValidationOutput) -> PublishedValidation:
    """Stage/fsync verified report/graphs only after the caller has verified worker termination.

    A database failure after rename leaves a complete orphan for startup adoption.
    Existing publications are immutable. This function neither completes a job
    nor retains an authorization receipt in the database.
    """
    publication = PublishedValidation(snapshot=snapshot,output=output)
    raw = canonical_bytes(publication.model_dump(mode="json"))
    if len(raw) > PUBLICATION_BYTES:
        raise ValueError("validation publication exceeds byte budget")
    destination = storage.validation_dir(snapshot.run_id)
    if destination.exists():
        existing = read_publication(snapshot.run_id)
        if existing != publication:
            raise ValueError("validation publication identity conflict")
        return existing
    root = storage.validation_staging_dir()
    root.mkdir(parents=True,exist_ok=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{snapshot.run_id}.",dir=root))
    try:
        write_bytes(staging/"validation.json",raw)
        write_bytes(staging/"sha256",hashlib.sha256(raw).hexdigest().encode("ascii"))
        sync_directory(staging)
        publish_directory(staging,destination)
        sync_directory(destination.parent)
        sync_directory(root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return publication


def remove_pending() -> None:
    """Remove unpublished staging only while startup holds the exclusive reconciled runtime owner."""
    root = storage.validation_staging_dir()
    if root.exists():
        shutil.rmtree(root)
        sync_directory(root.parent)
