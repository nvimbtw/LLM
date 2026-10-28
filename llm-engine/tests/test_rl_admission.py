"""Finite parent ledger/history/disk admission and its shared package-copy boundary."""

from uuid import uuid4

import pytest

from api import rl_admission, training_admission
from llm import storage
from llm.config import RlConfig


def config(**changes):
    """Build a normal UUID-owned RL request, allowing each test to vary actual schedule inputs."""
    return RlConfig(model_name=str(uuid4()), base_model=str(uuid4()), task_types=["fixture"], **changes)


def test_default_rl_schedule_has_finite_admitted_effect_and_parent_memory_allowances():
    """The ordinary public default fits explicit journal ceilings and reserves parent memory outside the worker tree."""
    budget = rl_admission.effect_budget(config())
    assert 0 < budget.ownership_bytes <= rl_admission.MAX_LOG_BYTES
    assert 0 < budget.issued_bytes <= rl_admission.MAX_LOG_BYTES
    assert budget.disk_bytes < rl_admission.MAX_EFFECT_ROOT_BYTES
    before = training_admission.Capacity(4 << 30, 10 << 30)
    after = rl_admission.worker_capacity(before, budget)
    assert after.host_bytes == before.host_bytes - budget.parent_memory_bytes
    assert after.disk_bytes == before.disk_bytes


def test_schedule_growth_cannot_hide_durable_operation_cost():
    """More admitted iterations reserve more evidence; a schedule that fits history can still exceed the explicit effect ceiling."""
    small = rl_admission.effect_budget(config(iterations=1))
    normal = rl_admission.effect_budget(config())
    assert normal.ownership_bytes > small.ownership_bytes
    with pytest.raises(ValueError, match="durable gym-effect allowance"):
        rl_admission.effect_budget(config(iterations=1000, tasks_per_batch=1, group_size=2, max_rounds=32))


def test_parent_memory_reservation_cannot_overassign_the_worker_tree():
    """Low host capacity is refused before worker creation instead of assigning parent evidence memory twice."""
    budget = rl_admission.effect_budget(config())
    with pytest.raises(ValueError, match="host admission"):
        rl_admission.worker_capacity(training_admission.Capacity(1 << 30, 10 << 30), budget)


@pytest.mark.parametrize("violation", ["jobs", "bytes", "symlink", "extra_file", "foreign_directory"])
def test_retained_effect_evidence_has_aggregate_job_disk_and_file_ownership_limits(tmp_path, monkeypatch, violation):
    """Uncertainty can retain evidence but cannot grow an unbounded root or cause traversal through foreign filesystem entries."""
    root = tmp_path / "effects"
    budget = rl_admission.EffectBudget(1024, 128, 1024)
    rl_admission.reserve_root(root, budget)
    job = root / str(uuid4())
    job.mkdir()
    if violation == "jobs":
        monkeypatch.setattr(rl_admission, "MAX_RETAINED_JOBS", 1)
    elif violation == "bytes":
        monkeypatch.setattr(rl_admission, "MAX_EFFECT_ROOT_BYTES", 2048)
        (job / "ownership.jsonl").write_bytes(b"x" * 1024)
    elif violation == "symlink":
        target = tmp_path / "foreign"
        target.write_text("foreign contents")
        (job / "ownership.jsonl").symlink_to(target)
    elif violation == "extra_file":
        (job / "commands.json").write_text("not ownership evidence")
    else:
        (root / "foreign-directory").mkdir()
    with pytest.raises(ValueError):
        rl_admission.reserve_root(root, budget)


def test_owned_job_evidence_paths_do_not_create_directories_or_accept_traversal(data_root):
    """All production evidence paths derive from storage UUID authority without filesystem creation effects."""
    identity = uuid4()
    target = storage.gym_effect_dir(identity)
    assert target.parent == storage.gym_effects_dir() and target.name == str(identity)
    assert not target.exists()
    with pytest.raises(ValueError):
        storage.gym_effect_dir("../another-job")


def test_shared_input_admission_reserves_parent_effect_disk_before_generation_copies(tmp_path):
    """RL admits actual tokenizer files and separate parent evidence without fabricating a training corpus plan."""
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_bytes(b"{}")
    capacity = training_admission.Capacity(2 << 30, 10 << 30)
    plain = training_admission.admit_files((tokenizer,), (), capacity, "cpu")
    reserved = training_admission.admit_files((tokenizer,), (), capacity, "cpu", reserved_disk_bytes=1 << 30)
    assert reserved.storage.package_bytes < plain.storage.package_bytes
    assert reserved.policy.resources.total_artifact_bytes < plain.policy.resources.total_artifact_bytes
    with pytest.raises(ValueError, match="insufficient disk"):
        training_admission.admit_files((tokenizer,), (), capacity, "cpu", reserved_disk_bytes=10 << 30)
