"""Actual bounded numeric input scans and serialization, plus explicit parent admission-policy boundaries."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from api import training_admission as admission
from llm.model import checkpoint
from llm.config import TrainConfig
from llm.train import data_identity, mapped_data
from llm.train.plan import DataPlan, Stage


def original_fingerprint(*arrays, plan):
    """Retain the prior complete-byte identity formula as independent compatibility evidence."""
    digest = hashlib.sha256()
    digest.update(json.dumps([plan.eot_id, plan.shuffle, plan.seed, plan.documents.names], sort_keys=True).encode())
    arrays = (*arrays, plan.documents.rows)
    for stage in plan.stages:
        digest.update(json.dumps([stage.name, stage.epochs, stage.documents.names]).encode())
        arrays = (*arrays, stage.documents.rows)
    for array in arrays:
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def test_blocked_identity_matches_the_original_schedule_and_numeric_byte_stream(monkeypatch):
    """Chunking preserves headers/order/dtypes exactly across multiple stages and arbitrary block boundaries."""
    corpus = np.arange(129, dtype=np.uint16)
    plan = DataPlan.whole_corpus(len(corpus))
    plan.stages = [Stage("first", 2, plan.documents), Stage("repeat", 3, plan.documents)]
    expected = original_fingerprint(corpus, plan=plan)
    monkeypatch.setattr(data_identity, "HASH_BLOCK_BYTES", 17)
    assert data_identity.fingerprint(corpus, plan=plan) == expected
    assert all(len(part.data) <= 17 for part in data_identity.parts(corpus) if part.array is not None)


def resident_bytes() -> int:
    """Read this actual Linux process's current resident pages, excluding reclaimable unmapped file cache."""
    import os
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def test_readonly_fingerprint_releases_real_mapped_pages_and_preserves_identity(tmp_path):
    """A 128 MiB actual file mapping drops resident pages and hashes with a bounded scan rather than retaining the corpus."""
    path = tmp_path / "large.bin"
    size = 128 << 20
    with path.open("wb") as output:
        output.truncate(size)
    corpus = np.memmap(path, dtype=np.uint16, mode="r")
    plan = DataPlan.whole_corpus(len(corpus))
    expected = data_identity.fingerprint(corpus, plan=plan)
    touched = resident_bytes()
    mapped_data.release_pages(corpus)
    baseline = resident_bytes()
    assert touched - baseline >= size // 2
    samples = []

    def observe():
        """Sample real RSS between bounded identity blocks without bypassing the actual page-release operation."""
        samples.append(resident_bytes())

    assert mapped_data.fingerprint(corpus, plan=plan, check_cancelled=observe) == expected
    assert max(samples) - baseline < 32 << 20
    assert len(samples) >= size // data_identity.HASH_BLOCK_BYTES
    assert int(corpus[0]) == 0


def test_mapped_hash_cancellation_stops_before_the_next_numeric_block(tmp_path, monkeypatch):
    """Cancellation interrupts a real chunked scan without completing the remaining mapped input."""
    path = tmp_path / "input.bin"
    np.arange(4096, dtype=np.uint16).tofile(path)
    corpus = np.memmap(path, dtype=np.uint16, mode="r")
    calls = []
    monkeypatch.setattr(data_identity, "HASH_BLOCK_BYTES", 64)

    def cancelled():
        """Cancel after several actual data/header boundaries."""
        calls.append(True)
        if len(calls) == 5:
            raise InterruptedError("cancelled input scan")

    with pytest.raises(InterruptedError, match="cancelled input scan"):
        mapped_data.fingerprint(corpus, plan=DataPlan.whole_corpus(len(corpus)), check_cancelled=cancelled)
    assert len(calls) == 5


def test_tensor_serialization_quota_preserves_prior_bytes_and_removes_partial_output(tmp_path):
    """PyTorch archive writes cannot exceed their admitted bytes or replace a prior complete tensor file on refusal."""
    model = torch.nn.Linear(8, 8)
    path = tmp_path / "model.pt"
    path.write_bytes(b"previous complete output")
    with pytest.raises(ValueError, match="serialized file byte budget"):
        checkpoint.save_modular_weights(model, path, "a" * 64, max_bytes=64)
    assert path.read_bytes() == b"previous complete output"
    assert list(tmp_path.iterdir()) == [path]
    checkpoint.save_modular_weights(model, path, "a" * 64, max_bytes=1 << 20)
    blob = torch.load(path, weights_only=True)
    torch.testing.assert_close(blob["state_dict"]["weight"], model.weight)


@pytest.mark.parametrize("free", [0, 64 << 20, admission.DISK_RESERVE_BYTES])
def test_packing_refuses_insufficient_observed_disk_capacity(free):
    """A configured maximum cannot authorize acquisition when the observed filesystem cannot accommodate it."""
    with pytest.raises(ValueError, match="capacity"):
        admission.packing_budget(admission.Capacity(8 << 30, free))


def test_parent_admission_accounts_for_input_and_generation_copies(tmp_path):
    """The runtime aggregate ceiling fits observed free space while retaining parent-copy/headroom reservations."""
    paths = []
    for name in ("corpus", "spans", "tokenizer"):
        path = tmp_path / name
        path.write_bytes(b"actual fixture bytes")
        paths.append(path)
    prepared = SimpleNamespace(corpus=paths[0], spans=paths[1], tokenizer=paths[2])
    available = admission.Capacity(4 << 30, 4 << 30)
    value = admission.admit(prepared, None, available, "cpu")
    assert value.policy.resources.device_allocator_bytes == 0
    assert value.policy.resources.total_artifact_bytes < available.disk_bytes - admission.DISK_RESERVE_BYTES
    assert value.storage.package_bytes < available.disk_bytes // 6
    assert admission.packing_budget(available).max_corpus_bytes <= available.disk_bytes // 6


@pytest.mark.parametrize("changes", [
    {"batch_size": 0}, {"epochs": 0}, {"seed": 2**64}, {"lr": float("nan")},
    {"max_vocab": 100000}, {"curriculum": [{"epochs": 1}] * 129},
    {"datasets": ["one"] * 4097},
])
def test_invalid_preparation_limits_are_refused_before_data_acquisition(changes):
    """Shape/schedule/input cardinality and effective fitting limits are checked before a parent opens corpus access."""
    with pytest.raises(ValueError):
        admission.validate_configuration(TrainConfig(model_name="fixture", **changes))


def test_admission_allows_the_largest_input_grant_even_when_spans_exceed_corpus_and_output(tmp_path):
    """Per-file admission includes actual span/tokenizer sizes rather than assuming the corpus is the largest input."""
    paths = []
    for name, size in (("corpus", 1 << 20), ("spans", 96 << 20), ("tokenizer", 1024)):
        path = tmp_path / name
        with path.open("wb") as output:
            output.truncate(size)
        paths.append(path)
    prepared = SimpleNamespace(corpus=paths[0], spans=paths[1], tokenizer=paths[2])
    value = admission.admit(prepared, None, admission.Capacity(4 << 30, 700 << 20), "cpu")
    assert value.storage.package_bytes < paths[1].stat().st_size
    assert value.policy.resources.artifact_bytes >= paths[1].stat().st_size


def test_sft_admission_charges_mask_probes_and_both_package_copy_lifetimes(tmp_path):
    """SFT extra inputs and the original frozen base reduce generation admission and fit the aggregate runtime allowance."""
    paths = []
    for name, size in (("corpus", 1 << 20), ("spans", 1024), ("tokenizer", 1024), ("mask", 1 << 19), ("probe", 8 << 20)):
        path = tmp_path / name
        with path.open("wb") as source:
            source.truncate(size)
        paths.append(path)
    prepared = SimpleNamespace(corpus=paths[0], spans=paths[1], tokenizer=paths[2])
    saved = SimpleNamespace(integrity=SimpleNamespace(files={"weights": SimpleNamespace(size_bytes=64 << 20)}))
    available = admission.Capacity(4 << 30, 4 << 30)
    pretrain = admission.admit(prepared, saved, available, "cpu")
    supervised = admission.admit(prepared, saved, available, "cpu", additional_inputs=tuple(paths[3:]), reference=saved)
    assert supervised.storage.package_bytes < pretrain.storage.package_bytes
    assert supervised.policy.resources.artifact_bytes >= paths[4].stat().st_size
    assert supervised.policy.resources.total_artifact_bytes < available.disk_bytes - admission.DISK_RESERVE_BYTES
