"""Bounded saved-definition and graph reads without tensor loading, source execution or recapture."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from uuid import UUID

from llm.architecture import package
from llm.architecture.errors import ArchitectureError
from llm.architecture.graphs import read_fragment
from llm.architecture.package_integrity import (
    MANIFEST_BYTES, METADATA_BYTES, RECEIPT_BYTES, PackageIntegrity, decode_json, package_file, read_bytes,
)
from llm.architecture.spec import MAX_GRAPH_BYTES, MAX_SOURCE_BYTES, DefinitionPackage, ValidationReceipt, Capability
from llm.architecture.wire import GraphFragment, GraphQuery, ModelDefinitionSummary, RetainedSource

@dataclass(frozen=True)
class SavedMetadata:
    """Own bounded verified metadata only; numerical loading must still verify the complete package."""

    directory: Path
    manifest: DefinitionPackage
    integrity: PackageIntegrity
    metadata: dict
    receipts: tuple[ValidationReceipt, ...]

    def read(self, relative: str, ceiling: int) -> bytes:
        """Read one inventoried metadata/graph artifact, checking exact declared length and SHA-256 before decoding."""
        expected = self.integrity.files.get(relative)
        if expected is None or expected.size_bytes > ceiling:
            raise ValueError("saved metadata artifact is missing or exceeds its byte budget")
        raw = read_bytes(package_file(self.directory, relative), ceiling)
        if len(raw) != expected.size_bytes or hashlib.sha256(raw).hexdigest() != expected.sha256:
            raise ValueError("saved metadata artifact integrity mismatch")
        return raw

    def summary(self) -> ModelDefinitionSummary:
        """Return authoritative saved context/tokenizer/parameters and measured unique counts, never a guessed legacy shape."""
        definition = self.manifest.definition
        result = self.metadata.get("result", {})
        count = result.get("parameters") if isinstance(result, dict) else None
        if type(count) is not int or not 0 <= count <= 2**53 - 1:
            count = None
        return ModelDefinitionSummary(definition_id=self.manifest.definition_id, architecture=definition.architecture,
            parameters=definition.parameters, context_window=definition.context_window,
            vocabulary_size=definition.tokenizer.vocabulary_size, tokenizer_fingerprint=definition.tokenizer.fingerprint,
            dependencies=tuple(entry.reference for entry in definition.dependencies), capabilities=self.capabilities(),
            parameter_count=count, legacy_architecture=None, validation_receipts=self.receipts)

    def capabilities(self) -> tuple[Capability, ...]:
        """Report supported optional behavior only when every retained profile reports the same evidence-backed capability."""
        declared = {value.name: value for receipt in self.receipts for value in receipt.capabilities}
        return tuple(value if all(value in receipt.capabilities for receipt in self.receipts)
            else Capability(name=name, supported=False, reason="retained validation profiles do not share this capability evidence")
            for name, value in sorted(declared.items()))

    def source(self, revision_id: UUID) -> RetainedSource:
        """Read only a selected inventoried frozen root/dependency; builtin descriptors remain explicitly JSON, never current Python."""
        definition = self.manifest.definition
        if revision_id == definition.architecture.revision_id:
            reference, path, language = definition.architecture, "architecture.yaml", "yaml"
        else:
            entry = next((entry for entry in definition.dependencies if entry.reference.revision_id == revision_id), None)
            if entry is None or not entry.path.startswith("sources/"):
                raise ArchitectureError("not_found", "source revision is not retained by this model")
            reference, path = entry.reference, entry.path
            language = "json" if path.endswith(".builtin.json") else "python" if reference.kind == "block" else "yaml"
        raw = self.read(path, MAX_SOURCE_BYTES)
        if hashlib.sha256(raw).hexdigest() != reference.content_hash:
            raise ValueError("saved source differs from its frozen revision")
        return RetainedSource(revision_id=revision_id, definition_id=self.manifest.definition_id,
            content_hash=reference.content_hash, file=path, language=language, source=raw.decode("utf-8"))

    def graph(self, query: GraphQuery) -> GraphFragment:
        """Read only the requested immutable scope under its retained receipt identity; stale/cross-profile cursors refuse."""
        receipt = next((item for item in self.receipts if item.identity_hash == query.identity_hash), None)
        if receipt is None:
            raise ArchitectureError("graph_identity", "graph query does not identify a retained model receipt")
        digest = receipt.graph_manifest_hash
        manifest_raw = self.read(f"graphs/{digest}.json", MAX_GRAPH_BYTES)
        manifest = decode_json(manifest_raw)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("graphs"), list):
            raise ArchitectureError("graph_identity", "saved graph manifest is invalid")
        selected = next((entry for entry in manifest["graphs"] if isinstance(entry, dict)
            and all(entry.get(key) == getattr(query, key) for key in ("view", "mode", "signature", "scope"))), None)
        if selected is None or selected.get("sha256") not in self.manifest.graph_hashes:
            raise ArchitectureError("graph_scope", "graph scope/signature is unavailable")
        scope_hash = selected["sha256"]
        scope = self.read(f"graphs/{scope_hash}.json", MAX_GRAPH_BYTES - len(manifest_raw))
        return read_fragment(digest, {digest: manifest_raw.decode("utf-8"), scope_hash: scope.decode("utf-8")}, query)


def read_saved_metadata(directory: Path, *, definition_id: str | None = None) -> SavedMetadata:
    """Verify bounded saved metadata/evidence without reading weights, source files, optimizer state or live catalog rows."""
    manifest = package.read_manifest(directory)
    integrity = PackageIntegrity.model_validate(decode_json(read_bytes(package_file(directory, "integrity.json"), MANIFEST_BYTES)))
    if manifest.definition_id != integrity.definition_id or definition_id is not None and manifest.definition_id != definition_id:
        raise ValueError("saved definition differs from indexed model identity")
    shell = SavedMetadata(directory, manifest, integrity, {}, ())
    if DefinitionPackage.model_validate(decode_json(shell.read("package.json", MANIFEST_BYTES))) != manifest:
        raise ValueError("saved manifest changed during metadata read")
    metadata = decode_json(shell.read(manifest.metadata_path, METADATA_BYTES))
    if not isinstance(metadata, dict):
        raise ValueError("saved model metadata must be an object")
    if len(set(manifest.receipt_ids)) != len(manifest.receipt_ids):
        raise ValueError("duplicate saved receipt identity")
    receipts, consumed = [], 0
    for receipt_id in manifest.receipt_ids:
        raw = shell.read(f"receipts/{receipt_id}.json", min(MANIFEST_BYTES, RECEIPT_BYTES - consumed))
        consumed += len(raw)
        receipt = ValidationReceipt.model_validate(decode_json(raw))
        if receipt.receipt_id != receipt_id:
            raise ValueError("saved receipt path differs from its identity")
        package.validate_receipt(manifest, receipt, integrity.torch_version)
        receipts.append(receipt)
    return SavedMetadata(directory, manifest, integrity, metadata, tuple(receipts))
