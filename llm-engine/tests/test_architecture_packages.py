"""Self-contained package integrity, supervised round trips and publication crash boundaries."""

import copy
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4

import pytest
import torch

from llm import storage
from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import PackageBudget, decode_json, encode_json
from llm.architecture.spec import DefinitionPackage, ValidationReceipt, canonical_digest
from llm.model import checkpoint
from llm.model.device import DeviceGate
from llm.runtime.protocol import Load, RuntimePins
from llm.runtime.supervisor import WorkerManager
from tests.package_fixtures import payload
from tests.test_runtime_supervision import policy


def roundtrip(tmp_path: Path, configurations: list[dict]) -> list[dict]:
    """Run source construction and package loading only in an actually supervised numerical child."""
    gate = DeviceGate()
    with WorkerManager(tmp_path/"runtime",gate) as manager, gate.admit() as lease:
        with manager.session(lease,policy(construction_seconds=20),"tests.package_backend:PackageBackend") as session:
            descriptor = session.artifacts.write_bytes(encode_json({"packages":configurations}),"definition_v1")
            manifest = DefinitionPackage.model_validate(configurations[0]["manifest"])
            receipt = ValidationReceipt.model_validate(json.loads(configurations[0]["files"][f"receipts/{manifest.receipt_ids[0]}.json"]))
            pins = RuntimePins(definition_id=manifest.definition_id,bindings_digest=receipt.identity.parameter_digest,
                               profile_digest=canonical_digest(receipt.identity.profile),receipt_id=receipt.receipt_id)
            result = session.run(Load(type="load",pins=pins,definition=descriptor))
            report = json.loads(session.artifacts.read_bytes(result.result))
            assert not isinstance(report,dict), report.get("test_error") if isinstance(report,dict) else report
            return report


@pytest.fixture
def saved(tmp_path,monkeypatch):
    """Publish synthetic data-only evidence for byte/publication checks; numerical roundtrips use workers."""
    monkeypatch.setenv("DATA_DIR",str(tmp_path/"data"))
    config = payload()
    destination = tmp_path/"model"
    files = {name:source.encode() for name,source in config["files"].items()}
    files.update({"model.pt":b"opaque synthetic weights", "train_state.pt":b"opaque synthetic optimizer"})
    package.publish(destination,DefinitionPackage.model_validate(config["manifest"]),files)
    return destination


@pytest.mark.parametrize("name",["standard","looped"])
def test_builtin_packages_roundtrip_logits_aliases_optimizer_and_rng(tmp_path,monkeypatch,name):
    """New package loaders preserve causal logits, shared identity and saved data-only training state."""
    monkeypatch.setenv("DATA_DIR",str(tmp_path/"data"))
    config = payload(name)
    config["destination"] = str(tmp_path/"model")
    [result] = roundtrip(tmp_path,[config])
    assert result["same_logits"] and result["tied"] and result["rng"]
    assert result["optimizer"]>0 and result["step"]==1 and result["cursor"]=={"next_batch":7}
    checked = package.verify(Path(config["destination"]))
    assert checked.manifest.definition.recipe == DefinitionPackage.model_validate(config["manifest"]).definition.recipe


def test_custom_revisions_with_same_python_names_coexist_and_do_not_import_in_parent(tmp_path,monkeypatch):
    """Packaged source revisions remain independent while live catalog/history are absent."""
    monkeypatch.setenv("DATA_DIR",str(tmp_path/"data"))
    configs = [payload(custom=True,scale=scale) for scale in (1,2)]
    for index,config in enumerate(configs):
        config["destination"] = str(tmp_path/f"model-{index}")
    before = {name for name in sys.modules if name.startswith("_slate_block_")}
    first,second = roundtrip(tmp_path,configs)
    assert first["same_logits"] and second["same_logits"]
    assert first["logits"] != second["logits"]
    assert set(first["imports"]).isdisjoint(second["imports"])
    assert {name for name in sys.modules if name.startswith("_slate_block_")} == before
    for config in configs:
        package.verify(Path(config["destination"]))


def test_load_package_grants_restore_saved_definition_in_a_new_worker(tmp_path,monkeypatch):
    """Actual private archive/definition grants rebuild a model without catalog access after worker restart."""
    monkeypatch.setenv("DATA_DIR",str(tmp_path/"data"))
    config = payload("looped",custom=True)
    config["destination"] = str(tmp_path/"model")
    [expected] = roundtrip(tmp_path,[config])
    checked = package.verify(Path(config["destination"]))
    receipt = checked.receipts[0]
    pins = RuntimePins(definition_id=checked.manifest.definition_id,bindings_digest=receipt.identity.parameter_digest,
        profile_digest=canonical_digest(receipt.identity.profile),receipt_id=receipt.receipt_id)
    archive = package_transfer.write_archive(checked.directory,tmp_path/"model.tar")
    gate = DeviceGate()
    with WorkerManager(tmp_path/"runtime-restarted",gate) as manager, gate.admit() as lease:
        with manager.session(lease,policy(construction_seconds=20),"tests.package_backend:PackageBackend") as session:
            definition = session.artifacts.write_bytes(encode_json(checked.manifest.definition),"definition_v1")
            artifact = session.artifacts.stage_file(archive,"model_package_v1")
            result = session.run(Load(type="load",pins=pins,definition=definition,package=artifact))
            actual = json.loads(session.artifacts.read_bytes(result.result))
            assert actual.get("logits") == expected["logits"],actual
            assert actual["definition_id"] == pins.definition_id


@pytest.mark.parametrize("artifact",["model.pt","tokenizer.json","meta.json","architecture.yaml","package.json","train_state.pt"])
def test_tampered_artifacts_are_refused_before_tensor_loading(saved,artifact,monkeypatch):
    """Every required artifact is covered by integrity before any weights-only decode occurs."""
    path = saved/artifact
    path.write_bytes(path.read_bytes()+b"tampered")
    monkeypatch.setattr(torch,"load",lambda *args,**kwargs:pytest.fail("must not deserialize before verification"))
    with pytest.raises(ValueError):
        package.verify(saved)


def test_missing_and_unlisted_files_and_symbolic_links_are_refused(saved):
    """Inventory accounting covers the complete directory, not only the files named by the manifest."""
    extra = saved/"unlisted"
    extra.write_bytes(b"hidden")
    with pytest.raises(ValueError,match="inventory"):
        package.verify(saved)
    extra.unlink()
    (saved/"meta.json").unlink()
    (saved/"meta.json").symlink_to(saved/"package.json")
    with pytest.raises(ValueError,match="symbolic|links"):
        package.verify(saved)


def test_runtime_and_stale_receipt_rejection(saved):
    """Installed runtime and profile-specific retained evidence remain mandatory for loading."""
    with pytest.raises(ValueError,match="runtime"):
        package.verify(saved,runtime_version="not-installed")
    manifest = package.read_manifest(saved)
    receipt = saved/f"receipts/{manifest.receipt_ids[0]}.json"
    receipt.unlink()
    with pytest.raises(ValueError):
        package.verify(saved)


def test_publication_preflights_total_budget_before_copy(saved,tmp_path,monkeypatch):
    """Aggregate file admission rejects excessive inputs before staging or copying any artifact."""
    manifest = package.read_manifest(saved)
    monkeypatch.setattr(package,"copy_artifact",lambda *args:pytest.fail("copy must not start"))
    with pytest.raises(ValueError,match="cumulative"):
        package.publish(tmp_path/"too-large",manifest,{"one":b"x"*64,"two":b"y"*64},
                        budget=PackageBudget(total_bytes=100,file_bytes=80))
    assert not (tmp_path/"too-large").exists()


def test_registration_failure_retains_complete_package_for_idempotent_reconciliation(saved,tmp_path):
    """A post-rename DB failure leaves one verifiable orphan that can be registered on restart."""
    manifest = package.read_manifest(saved)
    files = {path.relative_to(saved).as_posix():path for path in saved.rglob("*")
             if path.is_file() and path.name not in {"package.json","integrity.json"}}
    destination = tmp_path/"orphan"
    def fail(_publication):
        """Inject a database failure after package rename completed."""
        raise RuntimeError("registration failed")
    with pytest.raises(RuntimeError,match="registration"):
        package.publish(destination,manifest,files,register=fail)
    assert package.verify(destination).manifest == manifest
    registered = set()
    def register(publication):
        """Simulate idempotent index registration using immutable package identity."""
        registered.add((publication.directory.name,publication.manifest.definition_id))
    package.reconcile(destination,register)
    package.reconcile(destination,register)
    assert len(registered)==1


def test_interrupted_copy_and_concurrent_destination_creation_preserve_prior_packages(saved,tmp_path,monkeypatch):
    """No interrupted or racing writer can replace an existing complete or empty destination."""
    manifest = package.read_manifest(saved)
    files = {path.relative_to(saved).as_posix():path for path in saved.rglob("*")
             if path.is_file() and path.name not in {"package.json","integrity.json"}}
    destination = tmp_path/"raced"
    real_sync = package.sync_tree
    def race(directory):
        """Create the destination in the interval between staging and atomic rename."""
        real_sync(directory)
        destination.mkdir()
    monkeypatch.setattr(package,"sync_tree",race)
    with pytest.raises(FileExistsError):
        package.publish(destination,manifest,files)
    assert list(destination.iterdir())==[]
    assert package.verify(saved).manifest==manifest
    assert not list(storage.package_staging_dir().glob("*"))


def test_definition_dedupe_is_independent_and_retained_when_model_package_is_removed(saved,tmp_path):
    """Semantic source retention never deletes another package's required files or cached definitions."""
    checked = package.verify(saved)
    first = package.retain_definition(checked)
    assert package.retain_definition(checked)==first
    copied = tmp_path/"copy"
    shutil.copytree(saved,copied)
    shutil.rmtree(saved)
    assert first.is_dir()
    assert package.verify(copied).manifest.definition_id==checked.manifest.definition_id


def test_legacy_loader_refuses_modular_construction_before_deserialization(saved,monkeypatch):
    """Parent weights/state readers refuse modular bytes before invoking any tensor deserialization."""
    monkeypatch.setattr(torch,"load",lambda *args,**kwargs:pytest.fail("must refuse before deserialization"))
    with pytest.raises(ValueError,match="worker"):
        checkpoint.load(saved/"model.pt")
    for name in ("train_state.pt","model.pt"):
        with pytest.raises(ValueError,match="worker"):
            checkpoint.load_state(saved/name)
