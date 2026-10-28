"""Self-contained model publication and data-only verification, independent of the live catalog."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import tempfile
from uuid import UUID

from llm import storage
from llm.architecture.dependencies import dependency_digest, recipe_dependencies
from llm.architecture.graphs import verify_receipt_graphs
from llm.architecture.package_integrity import (
    MANIFEST_BYTES, METADATA_BYTES, RECEIPT_BYTES, PackageBudget, PackageIntegrity, decode_json,
    FileIdentity, encode_json, enumerate_files, file_identity, package_file, publish_directory,
    read_bytes, sync_directory, write_bytes,
)
from llm.architecture.parse import parse_recipe
from llm.architecture.spec import (
    MAX_GRAPH_BYTES, MAX_NESTING, MAX_SOURCE_BYTES, DefinitionPackage, Recipe,
    RevisionRef, RevisionSubject, ValidationReceipt, canonical_digest, checked_relative_path,
)


@dataclass(frozen=True)
class VerifiedPackage:
    """Owned validated data snapshots; this record contains no imported custom classes or tensors."""

    directory: Path
    manifest: DefinitionPackage
    integrity: PackageIntegrity
    receipts: tuple[ValidationReceipt, ...]
    sources: Mapping[UUID, bytes]


@dataclass(frozen=True)
class Publication:
    """Complete verified publication awaiting or having completed caller-owned index registration."""

    directory: Path
    manifest: DefinitionPackage


def current_runtime_version() -> str:
    """Return PyTorch's exact build identity, including CUDA/CPU suffixes absent from distribution metadata."""
    import torch

    return str(torch.__version__)


def read_manifest(directory: Path) -> DefinitionPackage:
    """Read bounded summary/definition metadata without source imports or tensor deserialization."""
    return DefinitionPackage.model_validate(decode_json(read_bytes(
        package_file(directory, "package.json"), MANIFEST_BYTES)))


def dependency_sources(manifest: DefinitionPackage, directory: Path) -> dict[UUID, bytes]:
    """Verify every immutable source revision's exact bytes before any custom code is imported."""
    result = {}
    for entry in manifest.definition.dependencies:
        if not entry.path.startswith("sources/"):
            raise ValueError("dependency artifacts must be under sources/")
        data = read_bytes(package_file(directory, entry.path), MAX_SOURCE_BYTES)
        if len(data) != entry.bytes or hashlib.sha256(data).hexdigest() != entry.reference.content_hash:
            raise ValueError("packaged dependency source identity mismatch")
        result[entry.reference.revision_id] = data
    return result


def validate_closure(manifest: DefinitionPackage, sources: Mapping[UUID, bytes]) -> None:
    """Require the exact bounded transitive recipe closure, including immutable child recipes."""
    available = {entry.reference.revision_id: entry.reference for entry in manifest.definition.dependencies}
    found = {}
    pending = [(manifest.definition.recipe, frozenset({manifest.definition.architecture.revision_id}), 0)]
    while pending:
        recipe, ancestry, depth = pending.pop()
        if depth > MAX_NESTING:
            raise ValueError("package dependency nesting exceeds budget")
        for reference in recipe_dependencies(recipe).values():
            if reference.revision_id in ancestry:
                raise ValueError("recursive packaged architecture")
            if available.get(reference.revision_id) != reference:
                raise ValueError("missing or conflicting packaged dependency")
            if reference.revision_id in found:
                continue
            found[reference.revision_id] = reference
            if reference.kind == "architecture":
                child = parse_recipe(sources[reference.revision_id].decode("utf-8"))
                pending.append((child, ancestry | {reference.revision_id}, depth + 1))
    if found != available:
        raise ValueError("package contains dependencies outside its recipe closure")


def validate_receipt(manifest: DefinitionPackage, receipt: ValidationReceipt, torch_version: str) -> None:
    """Require a valid receipt for these exact saved semantics/runtime without executing source or reading tensors."""
    definition = manifest.definition
    pins_digest = dependency_digest(recipe_dependencies(definition.recipe))
    identity = receipt.identity
    receipt.require_current(identity)
    subject = identity.subject
    source_matches = (subject.reference == definition.architecture if isinstance(subject, RevisionSubject)
                      else subject.resource_id == definition.architecture.resource_id and subject.content_hash == definition.architecture.content_hash)
    if (receipt.receipt_id not in manifest.receipt_ids or not source_matches
            or identity.parameter_digest != canonical_digest(definition.parameters)
            or identity.dependency_digest != pins_digest or identity.tokenizer != definition.tokenizer
            or identity.context_window != definition.context_window
            or identity.profile.torch_version != torch_version
            or receipt.graph_manifest_hash not in manifest.graph_hashes):
        raise ValueError("package evidence differs from its saved definition/runtime")


def validate_evidence(manifest: DefinitionPackage, directory: Path, torch_version: str) -> tuple[ValidationReceipt, ...]:
    """Require retained exact profile receipts and their graph artifacts without live job/catalog reads."""
    receipts = []
    receipt_bytes = 0
    if len(set(manifest.receipt_ids)) != len(manifest.receipt_ids) or len(set(manifest.graph_hashes)) != len(manifest.graph_hashes):
        raise ValueError("duplicate package evidence identity")
    for receipt_id in manifest.receipt_ids:
        raw = read_bytes(package_file(directory, f"receipts/{receipt_id}.json"), min(MANIFEST_BYTES, RECEIPT_BYTES - receipt_bytes))
        receipt_bytes += len(raw)
        receipt = ValidationReceipt.model_validate(decode_json(raw))
        if receipt.receipt_id != receipt_id:
            raise ValueError("package receipt path differs from its identity")
        validate_receipt(manifest, receipt, torch_version)
        receipts.append(receipt)
    blobs = {}
    graph_bytes = 0
    for graph_hash in manifest.graph_hashes:
        raw = read_bytes(package_file(directory, f"graphs/{graph_hash}.json"), MAX_GRAPH_BYTES - graph_bytes)
        graph_bytes += len(raw)
        if hashlib.sha256(raw).hexdigest() != graph_hash:
            raise ValueError("package graph digest mismatch")
        blobs[graph_hash] = raw.decode("utf-8")
    for receipt in receipts:
        verify_receipt_graphs(receipt, blobs)
    return tuple(receipts)


def validate_compatibility(manifest: DefinitionPackage, sources: Mapping[UUID, bytes]) -> None:
    """Check every installed builtin/SDK identity before any dependency can execute custom source."""
    from llm.architecture.builtins.catalog import builtin_definitions, compatibility_manifest
    from llm.architecture.source import imports_public_builtins

    installed = builtin_definitions()
    current = compatibility_manifest()
    digest = canonical_digest(current)
    for entry in manifest.definition.dependencies:
        source = sources[entry.reference.revision_id]
        if entry.builtin_compatibility is not None and entry.builtin_compatibility != digest:
            raise ValueError("incompatible builtin implementation")
        if entry.path.endswith(".builtin.json"):
            definition = next((value for value in installed.values() if value.reference == entry.reference), None)
            expected = {"manifest": current, "declaration": definition.declaration.model_dump(mode="json")} if definition else None
            if (entry.reference.kind != "block" or entry.builtin_compatibility != digest
                    or source != encode_json(expected)):
                raise ValueError("incompatible builtin revision/declaration")
        elif entry.reference.kind == "block" and imports_public_builtins(source.decode("utf-8")):
            if entry.builtin_compatibility != digest:
                raise ValueError("custom builtin dependency is missing compatibility pin")


def verify(directory: Path, *, budget: PackageBudget = PackageBudget(), runtime_version: str | None = None,
           check_cancelled: Callable[[], None] = lambda: None) -> VerifiedPackage:
    """Check complete package bytes, semantic/source/evidence identities and runtime before execution.

    This parent-safe operation never unpickles tensors or imports custom source.
    Published directories must remain immutable throughout verification and use.
    """
    check_cancelled()
    manifest = read_manifest(directory)
    integrity = PackageIntegrity.model_validate(decode_json(read_bytes(
        package_file(directory, "integrity.json"), MANIFEST_BYTES)))
    if integrity.definition_id != manifest.definition_id or integrity.torch_version != (runtime_version or current_runtime_version()):
        raise ValueError("incompatible package definition or PyTorch runtime")
    required = {"package.json", "architecture.yaml", "model.pt", "tokenizer.json", "meta.json"}
    required.update(entry.path for entry in manifest.definition.dependencies)
    required.update(f"receipts/{identity}.json" for identity in manifest.receipt_ids)
    required.update(f"graphs/{digest}.json" for digest in manifest.graph_hashes)
    if manifest.training_state_path:
        required.add(manifest.training_state_path)
    if not required <= integrity.files.keys():
        raise ValueError("package inventory omits required artifacts")
    if enumerate_files(directory, budget) != set(integrity.files) | {"integrity.json"}:
        raise ValueError("package inventory differs from its complete artifact tree")
    integrity_bytes = file_identity(directory / "integrity.json", MANIFEST_BYTES).size_bytes
    if len(integrity.files) > budget.files or sum(item.size_bytes for item in integrity.files.values()) + integrity_bytes > budget.total_bytes:
        raise ValueError("package inventory exceeds resource budget")
    for name, expected in integrity.files.items():
        if file_identity(package_file(directory, name), budget.file_bytes, check_cancelled=check_cancelled) != expected:
            raise ValueError("package artifact integrity mismatch")
    root_source = read_bytes(package_file(directory, "architecture.yaml"), MAX_SOURCE_BYTES)
    if (hashlib.sha256(root_source).hexdigest() != manifest.definition.architecture.content_hash
            or parse_recipe(root_source.decode("utf-8")) != manifest.definition.recipe):
        raise ValueError("package root recipe identity mismatch")
    tokenizer = integrity.files["tokenizer.json"]
    if tokenizer.sha256 != manifest.definition.tokenizer.fingerprint:
        raise ValueError("package tokenizer fingerprint mismatch")
    decode_json(read_bytes(package_file(directory, "meta.json"), METADATA_BYTES))
    sources = dependency_sources(manifest, directory)
    validate_closure(manifest, sources)
    validate_compatibility(manifest, sources)
    receipts = validate_evidence(manifest, directory, integrity.torch_version)
    return VerifiedPackage(directory, manifest, integrity, receipts, sources)


def copy_artifact(source: Path | bytes, target: Path, max_bytes: int) -> None:
    """Snapshot a caller-authorized artifact in bounded chunks and fsync its staging bytes."""
    if isinstance(source, bytes):
        if len(source) > max_bytes:
            raise ValueError("package artifact exceeds byte budget")
        write_bytes(target, source)
        return
    expected = file_identity(source, max_bytes)
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with source.open("rb") as input_file, target.open("xb") as output:
        for chunk in iter(lambda: input_file.read(1 << 20), b""):
            total += len(chunk)
            if total > expected.size_bytes:
                raise ValueError("package input changed during publication")
            output.write(chunk)
        output.flush()
        os.fsync(output.fileno())
    if file_identity(target, max_bytes) != expected:
        raise ValueError("package input changed during publication")


def sync_tree(directory: Path) -> None:
    """Fsync every staged directory bottom-up before publishing the immutable tree."""
    for root, directories, _ in os.walk(directory, topdown=False):
        for name in directories:
            sync_directory(Path(root) / name)
        sync_directory(Path(root))


def publication_metadata(
    manifest: DefinitionPackage, files: Mapping[str, Path | bytes], budget: PackageBudget,
) -> tuple[bytes, bytes]:
    """Preflight every input identity plus generated metadata against cumulative disk budgets."""
    manifest_bytes = encode_json(manifest)
    if len(manifest_bytes) > MANIFEST_BYTES:
        raise ValueError("package manifest exceeds byte budget")
    inventory = {}
    for name, source in files.items():
        checked_relative_path(name)
        if isinstance(source, bytes):
            if len(source) > budget.file_bytes:
                raise ValueError("package artifact exceeds byte budget")
            identity = FileIdentity(sha256=hashlib.sha256(source).hexdigest(), size_bytes=len(source))
        else:
            identity = file_identity(source, budget.file_bytes)
        inventory[name] = identity
        if sum(item.size_bytes for item in inventory.values()) > budget.total_bytes:
            raise ValueError("package inputs exceed cumulative byte budget")
    inventory["package.json"] = FileIdentity(sha256=hashlib.sha256(manifest_bytes).hexdigest(), size_bytes=len(manifest_bytes))
    integrity = PackageIntegrity(definition_id=manifest.definition_id, torch_version=current_runtime_version(), files=inventory)
    integrity_bytes = encode_json(integrity)
    if len(integrity_bytes) > MANIFEST_BYTES or len(integrity_bytes) + sum(item.size_bytes for item in inventory.values()) > budget.total_bytes:
        raise ValueError("package metadata exceeds cumulative byte budget")
    return manifest_bytes, integrity_bytes


def publish(
    destination: Path, manifest: DefinitionPackage, files: Mapping[str, Path | bytes], *,
    budget: PackageBudget = PackageBudget(), register: Callable[[Publication], None] | None = None,
    staging_root: Path | None = None,
) -> Publication:
    """Stage/fsync/verify a complete immutable package before rename, then register it.

    Registration failure leaves one complete discoverable package for reconciliation;
    callers must stop publication rather than accumulating unregistered generations.
    Existing destinations are immutable and are never replaced, including on retries.
    Workers supply their private session staging root; parent callers use the
    authoritative storage staging root.
    """
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("model package destination already exists")
    if "package.json" in files or "integrity.json" in files or len(files) + 1 > budget.files:
        raise ValueError("reserved or excessive package artifact paths")
    manifest_bytes, integrity_bytes = publication_metadata(manifest, files, budget)
    staging_root = staging_root if staging_root is not None else storage.package_staging_dir()
    staging_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"{destination.name}.", dir=staging_root))
    written = len(manifest_bytes) + len(integrity_bytes)
    try:
        for name, source in files.items():
            target = package_file(staging, name)
            copy_artifact(source, target, min(budget.file_bytes, budget.total_bytes - written))
            written += target.stat().st_size
        write_bytes(staging / "package.json", manifest_bytes)
        write_bytes(staging / "integrity.json", integrity_bytes)
        verify(staging, budget=budget)
        sync_tree(staging)
        if destination.exists():
            raise FileExistsError("model package destination already exists")
        publish_directory(staging, destination)
        sync_directory(destination.parent)
        sync_directory(staging_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    publication = Publication(destination, manifest)
    if register is not None:
        register(publication)
    return publication


def reconcile(directory: Path, register: Callable[[Publication], None], *, budget: PackageBudget = PackageBudget()) -> Publication:
    """Re-adopt one complete verified orphan using an idempotent caller-owned registration transaction."""
    checked = verify(directory, budget=budget)
    publication = Publication(directory, checked.manifest)
    register(publication)
    return publication


def remove_staging() -> None:
    """Remove abandoned unpublished package trees only while the engine holds its exclusive startup lock."""
    staging = storage.package_staging_dir()
    if staging.exists():
        shutil.rmtree(staging)
        sync_directory(staging.parent)


def retain_definition(checked: VerifiedPackage, *, budget: PackageBudget = PackageBudget()) -> Path:
    """Deduplicate immutable semantic/source bytes without making model packages depend on this cache.

    No package deletion removes this cache; conservative retention avoids deleting
    source bytes referenced by another model or by retained database definitions.
    Receipts/graphs remain profile-specific files in each self-contained model.
    """
    destination = storage.definition_dir(checked.manifest.definition_id)
    files = {"definition.json": encode_json(checked.manifest.definition),
             "architecture.yaml": read_bytes(checked.directory / "architecture.yaml", MAX_SOURCE_BYTES)}
    files.update({entry.path: checked.sources[entry.reference.revision_id]
                  for entry in checked.manifest.definition.dependencies})
    if destination.exists():
        for name, expected in files.items():
            if read_bytes(package_file(destination, name), budget.file_bytes) != expected:
                raise ValueError("conflicting retained definition bytes")
        return destination
    staging_root = storage.package_staging_dir()
    staging_root.mkdir(parents=True, exist_ok=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f"definition-{destination.name}.", dir=staging_root))
    try:
        for name, data in files.items():
            write_bytes(package_file(staging, name), data)
        sync_tree(staging)
        try:
            publish_directory(staging, destination)
        except OSError:
            if not destination.is_dir():
                raise
            for name, expected in files.items():
                if read_bytes(package_file(destination, name), budget.file_bytes) != expected:
                    raise ValueError("conflicting concurrent definition publication")
        sync_directory(destination.parent)
        sync_directory(staging_root)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination
