"""Data-only archive, graph-evidence and crash tests without model execution or network fixtures."""

import hashlib
import io
import json
from pathlib import Path
import tarfile

import pytest
import torch

from llm.architecture import package, package_transfer
from llm.architecture.graphs import verify_receipt_graphs
from llm.architecture.package_integrity import PackageBudget
from llm.architecture.package_execution import validate_saved_tensors
from tests.test_architecture_packages import saved


def archive_bytes(name: str, *, kind: bytes = tarfile.REGTYPE, link: str = "", duplicate: bool = False, size: int = 1) -> bytes:
    """Build deliberately invalid fixed-header containers without extracting any paths."""
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        for _ in range(2 if duplicate else 1):
            item = tarfile.TarInfo(name)
            item.type = kind
            item.linkname = link
            item.size = size if kind == tarfile.REGTYPE else 0
            archive.addfile(item, io.BytesIO(b"x" * item.size) if item.size else None)
    return stream.getvalue()


def test_archive_roundtrip_is_self_contained_and_preserves_every_byte(saved, tmp_path):
    """Verified uncompressed transfer retains receipts, sources and opaque tensors without deserializing them."""
    archive = package_transfer.write_archive(saved, tmp_path / "model.tar")
    checked = package_transfer.materialize(archive, tmp_path / "restored")
    assert checked.manifest == package.read_manifest(saved)
    for path in saved.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (checked.directory / path.relative_to(saved)).read_bytes()
    with pytest.raises(FileExistsError):
        package_transfer.materialize(archive, checked.directory)
    assert package.verify(checked.directory).manifest == checked.manifest


def test_cancelled_archive_verification_preserves_existing_destination(saved, tmp_path):
    """Interactive cancellation during bounded input hashing leaves the previous complete archive and no partial publication."""
    destination = tmp_path / "cancelled.tar"
    destination.write_bytes(b"previous complete archive")
    calls = 0

    def cancelled():
        """Interrupt a later verification chunk rather than failing before the archive operation begins."""
        nonlocal calls
        calls += 1
        if calls == 4:
            raise InterruptedError("request disconnected")

    with pytest.raises(InterruptedError, match="disconnected"):
        package_transfer.write_archive(saved, destination, check_cancelled=cancelled)
    assert destination.read_bytes() == b"previous complete archive"
    assert not list(tmp_path.glob(".cancelled.tar.*.pending"))


def test_even_zero_padding_truncation_of_a_complete_package_is_refused(saved, tmp_path):
    """A complete payload cannot conceal a truncated USTAR trailer merely because missing bytes were zero."""
    archive = package_transfer.write_archive(saved,tmp_path/"model.tar")
    archive.write_bytes(archive.read_bytes()[:-1])
    with pytest.raises(ValueError,match="trailer"):
        package_transfer.materialize(archive,tmp_path/"restored")
    assert not (tmp_path/"restored").exists()


@pytest.mark.parametrize("data", [
    b"", b"\x00" * 512, archive_bytes("../escape"), archive_bytes("/absolute"),
    archive_bytes("linked", kind=tarfile.SYMTYPE, link="../escape"),
    archive_bytes("linked", kind=tarfile.LNKTYPE, link="other"),
    archive_bytes("duplicate", duplicate=True), archive_bytes("payload")[:513],
    archive_bytes("payload")[:1023], archive_bytes("payload")[:-1] + b"x",
])
def test_invalid_containers_leave_no_materialization_or_traversal(tmp_path, data):
    """Truncation, links, duplicates, trailer corruption and traversal fail with staging cleanup."""
    archive = tmp_path / "bad.tar"
    archive.write_bytes(data)
    destination = tmp_path / "restored"
    with pytest.raises((ValueError, tarfile.HeaderError)):
        package_transfer.materialize(archive, destination)
    assert not destination.exists()
    assert not (tmp_path / "escape").exists()
    assert not list(tmp_path.glob(".package-*"))


def test_container_budget_rejects_members_before_copying(tmp_path, monkeypatch):
    """Header resource limits stop payload copying even before package metadata can be interpreted."""
    archive = tmp_path / "large.tar"
    archive.write_bytes(archive_bytes("payload", size=2))
    monkeypatch.setattr(package_transfer, "copy_member", lambda *args: pytest.fail("must preflight member"))
    with pytest.raises(ValueError, match="byte budget"):
        package_transfer.materialize(archive, tmp_path / "restored", budget=PackageBudget(total_bytes=1, file_bytes=1))
    assert not (tmp_path / "restored").exists()


@pytest.mark.parametrize("point", ["copy_artifact", "sync_tree", "publish_directory"])
def test_publication_failure_before_rename_preserves_prior_generation(saved, tmp_path, monkeypatch, point):
    """Injected copy/fsync/rename errors leave no destination or pending partial package."""
    manifest = package.read_manifest(saved)
    files = {path.relative_to(saved).as_posix(): path for path in saved.rglob("*")
             if path.is_file() and path.name not in {"package.json", "integrity.json"}}
    def fail(*_args):
        """Inject a storage failure at a selected publication boundary."""
        raise OSError("injected storage failure")
    monkeypatch.setattr(package, point, fail)
    with pytest.raises(OSError, match="injected"):
        package.publish(tmp_path / "interrupted", manifest, files)
    assert not (tmp_path / "interrupted").exists()
    assert package.verify(saved).manifest == manifest
    assert not list(package.storage.package_staging_dir().iterdir())


@pytest.mark.parametrize("mutation", ["identity", "incomplete", "missing", "modes", "dangling_edge"])
def test_graph_evidence_rejects_inconsistent_complete_inventories(saved, mutation):
    """Rehashed but contradictory graph metadata cannot be substituted for retained receipt evidence."""
    checked = package.verify(saved)
    receipt = checked.receipts[0]
    blobs = {digest:(saved / f"graphs/{digest}.json").read_text() for digest in checked.manifest.graph_hashes}
    manifest = json.loads(blobs.pop(receipt.graph_manifest_hash))
    if mutation == "identity":
        manifest["identity_hash"] = "b" * 64
    elif mutation == "incomplete":
        manifest["complete"] = False
    elif mutation == "missing":
        blobs.pop(manifest["graphs"][0]["sha256"])
    elif mutation == "modes":
        manifest["graphs"] = manifest["graphs"][:1]
    else:
        entry = manifest["graphs"][0]
        graph = json.loads(blobs[entry["sha256"]])
        graph["edges"] = [{"source":"missing","source_port":"value","target":"fixture","target_port":"value"}]
        encoded = json.dumps(graph, sort_keys=True, separators=(",", ":"))
        entry["sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
        blobs[entry["sha256"]] = encoded
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    blobs[digest] = encoded
    with pytest.raises(ValueError):
        verify_receipt_graphs(receipt.model_copy(update={"graph_manifest_hash":digest}), blobs)


def test_saved_tensor_profiles_refuse_implicit_precision_cast_and_alias_conflicts():
    """Safe deserialization cannot silently reinterpret a tensor precision or overwrite a tied parameter."""
    model = torch.nn.Module()
    model.first = torch.nn.Linear(2, 2, bias=False)
    model.second = torch.nn.Linear(2, 2, bias=False)
    model.second.weight = model.first.weight
    state = {name:value.clone() for name,value in model.state_dict().items()}
    validate_saved_tensors(model,state)
    with pytest.raises(ValueError,match="precision"):
        validate_saved_tensors(model,{name:value.to(torch.bfloat16) for name,value in state.items()})
    state["second.weight"].add_(1)
    with pytest.raises(ValueError,match="aliases"):
        validate_saved_tensors(model,state)
