"""Exclusive parent gym cleanup from real durable ownership evidence, without requiring job history or retrying uncertain writes."""

import shutil
from uuid import UUID

import httpx
import pytest

from api import rl_effects, rl_recovery
from llm import storage
from llm.gym.effect_client import EffectClient
from tests.test_rl_effect_client import public_run
from tests.test_rl_gym_dispatch import create, dispatched


def canonical_evidence(dispatched, tmp_path, monkeypatch):
    """Move this drained fixture's actual journals to production-owned UUID paths and retain only transport instrumentation."""
    context, _cfg, log, _replies, _calls = dispatched
    context.broker.close(timeout=2)
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "restart-data"))
    directory = storage.gym_effect_dir(log.authority.job_id)
    directory.mkdir(parents=True)
    shutil.copyfile(log.path, directory / "ownership.jsonl")
    if context.journal.path.exists():
        shutil.copyfile(context.journal.path, directory / "issued.jsonl")
    original = rl_effects.GymResourceOwner

    def observe(restored, *, response_bytes):
        """Expose the actual reconstructed owner to the fixture's write-before-Rust assertions without replacing cleanup behavior."""
        owner = original(restored, response_bytes=response_bytes)
        context.broker.dispatch = owner
        return owner

    def forbidden(*args, **kwargs):
        """Ensure cleanup does not depend on surviving execution/model rows or import custom source."""
        pytest.fail("gym retirement read pruned history or executed source")

    monkeypatch.setattr(rl_recovery, "GymResourceOwner", observe)
    monkeypatch.setattr("store.session", forbidden)
    monkeypatch.setattr("torch.load", forbidden)
    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    return directory


def test_startup_retires_known_resources_after_job_history_was_pruned(dispatched, tmp_path, monkeypatch):
    """Header actor and owned IDs suffice for confirmed close/abandon cleanup, then fully settled evidence retires."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    run_id, episodes = create(EffectClient(context), cfg)
    directory = canonical_evidence(dispatched, tmp_path, monkeypatch)
    replies.extend([{"episode_id": episode.id, "status": "closed"} for episode in episodes])
    replies.append({"id": run_id, "status": "abandoned"})
    rl_recovery.reconcile()
    assert len(calls) == 4 and not directory.exists()
    assert context.broker.dispatch.log.authority == log.authority
    assert all(episode.terminal for episode in context.broker.dispatch.log.episodes.values())
    rl_recovery.reconcile()
    assert len(calls) == 4


def test_startup_preserves_lost_creation_evidence_without_reissuing_creation(dispatched, tmp_path, monkeypatch):
    """A run UUID lost with its creation response cannot be recovered by repeating the write, even after another restart."""
    context, cfg, log, replies, calls = dispatched
    replies.append(httpx.ReadError("creation response lost"))
    with pytest.raises(RuntimeError):
        create(EffectClient(context), cfg)
    directory = canonical_evidence(dispatched, tmp_path, monkeypatch)
    rl_recovery.reconcile()
    rl_recovery.reconcile()
    assert directory.is_dir() and len(calls) == 1
    assert context.broker.dispatch.log.pending.request.operation_id == log.pending.request.operation_id


def test_startup_closes_known_resources_but_keeps_the_uncertain_tool_identity(dispatched, tmp_path, monkeypatch):
    """New parent retirement writes do not clear a lost tool result or allow that command to be reissued."""
    context, cfg, log, replies, calls = dispatched
    replies.append(public_run(2))
    client = EffectClient(context)
    run_id, episodes = create(client, cfg)
    replies.append(httpx.ReadError("step response lost"))
    with pytest.raises(RuntimeError):
        client.step(episodes[0].id, "bash", "echo only-once", "reply")
    directory = canonical_evidence(dispatched, tmp_path, monkeypatch)
    replies.extend([{"episode_id": episode.id, "status": "closed"} for episode in episodes])
    replies.append({"id": run_id, "status": "abandoned"})
    rl_recovery.reconcile()
    assert directory.exists() and len(calls) == 5
    assert context.broker.dispatch.log.pending.request.operation_id == log.pending.request.operation_id
    assert context.broker.dispatch.log.runs[UUID(run_id)]
    rl_recovery.reconcile()
    assert len(calls) == 5


def test_startup_never_repeats_a_cleanup_write_with_a_lost_acknowledgement(dispatched, tmp_path, monkeypatch):
    """Cleanup's own durable terminal fences survive another restart independently of worker operation status."""
    context, cfg, _log, replies, calls = dispatched
    replies.append(public_run(2))
    run_id, episodes = create(EffectClient(context), cfg)
    directory = canonical_evidence(dispatched, tmp_path, monkeypatch)
    replies.extend([httpx.ReadError("close response lost"), {"episode_id": episodes[1].id, "status": "closed"},
        {"id": run_id, "status": "abandoned"}])
    rl_recovery.reconcile()
    assert len(calls) == 4 and directory.exists()
    rl_recovery.reconcile()
    assert len(calls) == 4 and directory.exists()


def test_startup_requires_memory_for_bounded_ownership_reconstruction(dispatched, tmp_path, monkeypatch, capsys):
    """Low parent capacity retains metadata without loading its body or issuing external cleanup."""
    context, cfg, _log, replies, calls = dispatched
    replies.append(public_run(2))
    create(EffectClient(context), cfg)
    directory = canonical_evidence(dispatched, tmp_path, monkeypatch)
    monkeypatch.setattr(rl_recovery.training_admission, "available_host_bytes", lambda: 0)
    rl_recovery.reconcile()
    assert directory.exists() and len(calls) == 1
    assert "insufficient parent memory" in capsys.readouterr().out


def test_startup_defers_remaining_resources_after_its_elapsed_allowance(dispatched, tmp_path, monkeypatch, capsys):
    """Startup is bounded across retained jobs; unissued cleanup remains available for a later reconciliation."""
    context, cfg, _log, replies, calls = dispatched
    replies.append(public_run(2))
    create(EffectClient(context), cfg)
    directory = canonical_evidence(dispatched, tmp_path, monkeypatch)
    clock = iter((0.0, 31.0))
    monkeypatch.setattr(rl_recovery.time, "monotonic", lambda: next(clock))
    rl_recovery.reconcile()
    assert directory.exists() and len(calls) == 1
    assert "startup cleanup allowance" in capsys.readouterr().out
