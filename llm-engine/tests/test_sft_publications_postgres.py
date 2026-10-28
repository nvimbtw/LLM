"""Required PostgreSQL checks for SFT derivation, immutable enqueue pins and post-rename recovery."""

import pytest
import store

from tests.test_sft_publications import sft_job
from tests.test_sft_publications import (
    test_sft_adoption_keeps_enqueue_pin_and_retains_derived_definition as _verify_derivation,
    test_sft_registration_failure_preserves_authority_for_startup as _verify_rollback,
    test_sft_authority_cleanup_keeps_unadopted_or_active_inputs as _verify_cleanup,
    test_sft_queue_owns_one_lease_and_finishes_after_cleanup as _verify_queue,
)
from tests.test_training_publications import published_job
from tests.test_architecture_store import sessions, sample

pytestmark = pytest.mark.postgres


@pytest.fixture(autouse=True)
def parent_database(sessions, data_root, monkeypatch):
    """Bind real parent operations to the uniquely owned migrated test database and isolated artifact tree."""
    monkeypatch.setattr(store, "_sessions", sessions)


def test_postgres_sft_derivation_retains_immutable_enqueue_pin(sft_job, tmp_path, monkeypatch):
    """Exercise derived-tokenizer evidence and model registration under actual immutable/FK constraints."""
    _verify_derivation(sft_job, tmp_path, monkeypatch)


def test_postgres_sft_post_rename_failure_recovers_with_retained_authority(sft_job, tmp_path, monkeypatch):
    """Verify transactional rollback plus durable authority/package adoption without changing the failed job outcome."""
    _verify_rollback(sft_job, tmp_path, monkeypatch)


def test_postgres_sft_authority_retirement_after_real_history_cascade(sft_job, tmp_path):
    """Pruning a terminal job retires only settled preparation and preserves the registered derived definition."""
    _verify_cleanup(sft_job, tmp_path)


async def test_postgres_waiting_sft_chain_retains_pending_enqueue_pin(sft_job, tmp_path, monkeypatch):
    """Finalize actual inherited saved bytes under PostgreSQL immutable-pin/FK constraints without rewriting a waiting chain's pending pin."""
    await _verify_queue(None, sft_job, tmp_path, monkeypatch, "pending_chain")
