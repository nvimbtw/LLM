"""Signed HTTP recipe admission, immutable retry pins and deliberate legacy refusal.

Accepted source/receipt fixtures are data-only storage evidence. These tests do
not claim supervised construction, numerical preflight or saved-model recovery.
"""

import copy
from threading import Event
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

import store
from api import handlers
from api.runner import _work
from llm import storage
from llm.architecture.spec import FinalTokenizer, RevisionSubject, ValidationReceipt, ReceiptIdentity, canonical_digest
from llm.architecture.tokenizer_binding import binding_from_bytes
from llm.architecture.wire import ArchitectureDraftSave, BlockDraftSave, MetadataPatch
from llm.config import LEGACY_STRUCTURE_FIELDS, RlConfig, SftConfig, TrainConfig
from llm.orchestrate import train as legacy
from llm.train.model_preflight import training_profile
from store import architecture_queries as catalog, queries as q
from store.schema import ArchitectureRevision, Run, RunArchitecturePin, RunKind, RunStatus, TrainingRun
from tests.test_architecture_store import sample
from tests.test_architecture_visibility import headers
from tests.test_validation_queue import client


@pytest.fixture
def sessions(store_session):
    """Reuse the isolated SQLite factory for accepted storage fixtures; PostgreSQL checks remain separate."""
    return store._sessions


def request(sample, **changes) -> dict:
    """Create a sparse selection over the fixture's accepted immutable recipe, preserving caller overrides."""
    return {"model_name": "selected", "architecture": {
        "revision_id": str(sample.architecture_revision), "parameters": {},
    }, **changes}


def training_count() -> int:
    """Count actual persisted training-domain rows without conflating validation fixtures with training."""
    with store.session() as db:
        return db.scalar(select(func.count()).select_from(TrainingRun))


def pin(run_id: str) -> dict:
    """Read the independently durable immutable selection rather than rebuilding current catalog state."""
    with store.session() as db:
        value = db.get(RunArchitecturePin, UUID(run_id))
        return copy.deepcopy({"evidence": value.evidence, "parameters": value.parameters, "snapshot": value.snapshot})


def final_tokenizer() -> FinalTokenizer:
    """Publish a small complete owned tokenizer fixture with actual contiguous IDs and its exact hash."""
    tokenizer_id = uuid4()
    raw = Tokenizer(WordLevel(vocab={"<|endoftext|>": 0, "one": 1, "two": 2},
                              unk_token="<|endoftext|>")).to_str().encode()
    directory = storage.tokenizer_build_dir(str(tokenizer_id))
    directory.mkdir(parents=True)
    (directory / "tokenizer.json").write_bytes(raw)
    (directory / "build.json").write_text('{"vocab_size":3}')
    return binding_from_bytes(tokenizer_id, raw)


def exact_evidence(sample, binding, *, precision: str = "fp32") -> UUID:
    """Retain typed exact-profile storage evidence without claiming new numerical execution."""
    cfg = TrainConfig.model_validate(request(sample, tokenizer_id=str(binding.tokenizer_id)))
    profile = training_profile(cfg, 8, "cpu").model_copy(update={"precision": precision})
    with store.session() as db:
        revision = db.get(ArchitectureRevision, sample.architecture_revision)
        identity = ReceiptIdentity(subject=RevisionSubject(kind="revision", reference=catalog.revision_reference(revision)),
            dependency_digest=revision.dependency_digest, parameter_digest=canonical_digest(sample.parameters),
            input_profile_digest=canonical_digest({"tensor_inputs": {}, "domains": [value.model_dump(mode="json") for value in profile.domains]}),
            tokenizer=binding, context_window=8, profile=profile)
        receipt = ValidationReceipt(schema_version=1, receipt_id=uuid4(), identity=identity,
            identity_hash=canonical_digest(identity), verdict="valid", graph_manifest_hash="a" * 64)
        run = q.enqueue(db, kind=RunKind.model_validation, label="retained fixture", config={}, status=RunStatus.done)
        catalog.record_evidence(db, run.id, receipt)
        return run.id


@pytest.mark.parametrize("field", sorted(LEGACY_STRUCTURE_FIELDS))
async def test_selected_recipe_rejects_every_explicit_flat_structure_field(client, sample, field):
    """Even restating a legacy default introduces a second structural authority and creates no job."""
    value = getattr(TrainConfig(model_name="legacy"), field)
    response = await client.post(f"/train?request_id={uuid4()}", headers=headers(["administration"]),
                                 json=request(sample, **{field: value}))
    assert response.status_code == 422
    assert training_count() == 0


async def test_resume_and_saved_consumers_reject_recipe_override(client, sample):
    """Resume/SFT/RL inherit their saved definition instead of silently ignoring a new selection."""
    response = await client.post(f"/train?request_id={uuid4()}", headers=headers(["administration"]),
                                 json=request(sample, resume_from=str(uuid4())))
    assert response.status_code == 422
    for model in (SftConfig, RlConfig):
        with pytest.raises(ValueError, match="inherits its definition"):
            model.model_validate(request(sample, base_model=str(uuid4()), task_types=["example"]))
    assert training_count() == 0


async def test_recipe_submission_requires_admin_actor_and_request_identity(client, sample):
    """A selected source cannot be admitted through ordinary visibility or an untracked retry identity."""
    assert (await client.post(f"/train?request_id={uuid4()}", headers=headers(["public"]), json=request(sample))).status_code == 403
    assert (await client.post("/train", headers=headers(["administration"]), json=request(sample))).status_code == 422
    assert training_count() == 0


async def test_pending_submission_freezes_sources_and_idempotency_survives_archive(client, sample, monkeypatch):
    """Identical retries return the original output/pins even after archive; fresh selection and conflicting retries fail."""
    def forbidden(*args, **kwargs):
        """Fail any source import or authenticated data acquisition during HTTP submission."""
        pytest.fail("submission executed source or opened a dataset client")

    monkeypatch.setattr("llm.architecture.source.SourceLoader.load", forbidden)
    monkeypatch.setattr(legacy.remote, "Client", forbidden)
    admin, operation = headers(["administration"]), uuid4()
    payload = request(sample)
    first = await client.post(f"/train?request_id={operation}", headers=admin, json=payload)
    assert first.status_code == 202, first.text
    original = first.json()
    frozen = pin(original["run_id"])
    assert frozen["parameters"] == sample.parameters
    assert frozen["evidence"]["tokenizer"] == {"kind": "pending", "build_request_id": original["run_id"]}
    assert frozen["evidence"]["final_preflight_required"] is True
    assert frozen["snapshot"]["dependencies"][0]["source"] == "class StoredBlock: pass\n"
    assert frozen["snapshot"]["dependencies"][0]["compatibility"] == {"sdk_version": 1}
    assert frozen["snapshot"]["dependencies"][0]["origin"] == "custom"
    UUID(original["config"]["model_name"])
    assert "ctx_window" not in original["config"]
    with store.session() as db:
        catalog.save_block_draft(db, sample.block_id, BlockDraftSave(expected_version=1, source="invalid edited source"))
        catalog.save_architecture_draft(db, sample.architecture_id, ArchitectureDraftSave(expected_version=1, yaml="invalid edited YAML"))
        catalog.update_metadata(db, "architecture", sample.architecture_id,
            MetadataPatch(expected_metadata_version=1, archived=True))
    same = await client.post(f"/train?request_id={operation}", headers=admin, json=payload)
    assert same.status_code == 202 and same.json()["run_id"] == original["run_id"]
    changed = await client.post(f"/train?request_id={operation}", headers=admin, json={**payload, "lr": 0.1})
    assert changed.status_code == 409 and changed.json()["error"]["code"] == "request_conflict"
    fresh = await client.post(f"/train?request_id={uuid4()}", headers=admin, json=payload)
    assert fresh.status_code == 409 and fresh.json()["error"]["code"] == "archived"
    assert training_count() == 1 and pin(original["run_id"]) == frozen


async def test_final_binding_and_retained_receipt_survive_validation_history_pruning(client, sample, data_root):
    """Actual owned tokenizer facts and matching evidence are pinned independently of the validation job."""
    binding = final_tokenizer()
    validation_id = exact_evidence(sample, binding)
    with store.session() as db:
        assert q.delete_run(db, validation_id)
    payload = request(sample, tokenizer_id=str(binding.tokenizer_id))
    payload["architecture"]["validation_run_id"] = str(validation_id)
    response = await client.post(f"/train?request_id={uuid4()}", headers=headers(["administration"]), json=payload)
    assert response.status_code == 202, response.text
    frozen = pin(response.json()["run_id"])
    assert frozen["evidence"]["tokenizer"] == binding.model_dump(mode="json")
    assert frozen["evidence"]["exact_receipt_id"] is not None
    assert frozen["evidence"]["final_preflight_required"] is True


async def test_stale_runtime_receipt_cannot_authorize_training(client, sample, data_root):
    """A retained structurally matching receipt for another arithmetic profile leaves no training row."""
    binding = final_tokenizer()
    payload = request(sample, tokenizer_id=str(binding.tokenizer_id))
    payload["architecture"]["validation_run_id"] = str(exact_evidence(sample, binding, precision="bf16"))
    response = await client.post(f"/train?request_id={uuid4()}", headers=headers(["administration"]), json=payload)
    assert response.status_code == 409 and response.json()["error"]["code"] == "stale_evidence"
    assert training_count() == 0


async def test_chain_pins_waiting_consumers_and_replays_all_original_job_ids(client, sample):
    """A complete recipe chain pins later SFT/RL source identity before queue insertion and deduplicates as one operation."""
    admin, operation = headers(["administration"]), uuid4()
    steps = [
        {"kind": "train", "config": request(sample)},
        {"kind": "train", "config": {"model_name": "tuned"}},
        {"kind": "rl", "config": {"task_types": ["example"]}},
        {"kind": "benchmark", "config": {}},
    ]
    response = await client.post(f"/runs/chain?request_id={operation}", headers=admin, json=steps)
    assert response.status_code == 202, response.text
    rows = response.json()
    frozen = [pin(value["run_id"]) for value in rows[:3]]
    assert frozen[0] == frozen[1] == frozen[2]
    assert [value["status"] for value in rows] == ["queued", "waiting", "waiting", "waiting"]
    assert len({value["config"]["model_name"] for value in rows[:3]}) == 3
    assert "base_model" not in rows[1]["config"]
    assert "base_model" not in rows[2]["config"]
    same = await client.post(f"/runs/chain?request_id={operation}", headers=admin, json=steps)
    assert [value["run_id"] for value in same.json()] == [value["run_id"] for value in rows]
    assert training_count() == 3
    steps[1]["config"]["lr"] = 0.0001
    assert (await client.post(f"/runs/chain?request_id={operation}", headers=admin, json=steps)).status_code == 409


async def test_invalid_waiting_override_does_not_even_insert_the_first_recipe_step(client, sample, monkeypatch):
    """All effective consumer configs are checked before the first queue/domain write."""
    def forbidden(*args, **kwargs):
        """Detect a premature queue insert during whole-chain validation."""
        pytest.fail("invalid chain inserted a job")

    monkeypatch.setattr(q, "enqueue", forbidden)
    steps = [{"kind": "train", "config": request(sample)},
             {"kind": "train", "config": {"ctx_window": 16}}]
    response = await client.post(f"/runs/chain?request_id={uuid4()}", headers=headers(["administration"]), json=steps)
    assert response.status_code == 422
    assert training_count() == 0


async def test_retry_preserves_old_pins_after_archive_and_original_job_pruning(client, sample):
    """A fresh output retries frozen sources without reselecting an archived root, then retains its own pin lifetime."""
    admin = headers(["administration"])
    first = (await client.post(f"/train?request_id={uuid4()}", headers=admin, json=request(sample))).json()
    frozen = pin(first["run_id"])
    assert (await client.post(f"/runs/{first['run_id']}/cancel", headers=admin)).status_code == 200
    with store.session() as db:
        catalog.update_metadata(db, "architecture", sample.architecture_id, MetadataPatch(expected_metadata_version=1, archived=True))
    operation = uuid4()
    retry = await client.post(f"/runs/{first['run_id']}/retry?request_id={operation}", headers=admin)
    assert retry.status_code == 202, retry.text
    retried = retry.json()
    assert retried["config"]["model_name"] != first["config"]["model_name"]
    assert pin(retried["run_id"]) == frozen
    same = await client.post(f"/runs/{first['run_id']}/retry?request_id={operation}", headers=admin)
    assert same.json()["run_id"] == retried["run_id"]
    with store.session() as db:
        assert q.delete_run(db, UUID(first["run_id"]))
    assert pin(retried["run_id"]) == frozen
    after_pruning = await client.post(f"/runs/{first['run_id']}/retry?request_id={operation}", headers=admin)
    assert after_pruning.status_code == 202 and after_pruning.json()["run_id"] == retried["run_id"]
    assert training_count() == 1


async def test_recovery_reserves_a_new_output_and_inherits_the_last_registered_generation(client, sample):
    """Recovery metadata uses the saved generation and frozen source pin, without claiming to load checkpoint tensors."""
    admin = headers(["administration"])
    first = (await client.post(f"/train?request_id={uuid4()}", headers=admin, json=request(sample))).json()
    frozen = pin(first["run_id"])
    output = UUID(first["config"]["model_name"])
    checkpoint_id = uuid4()
    with store.session() as db:
        db.get(Run, UUID(first["run_id"])).status = RunStatus.failed
        q.upsert_model(db, model_id=checkpoint_id, name="complete generation", published=False,
                       parent_id=output, run_id=UUID(first["run_id"]), checkpoint_index=1,
                       checkpoint_step=3, context=8, parameters={"result": {"recovery": True}})
        catalog.update_metadata(db, "architecture", sample.architecture_id, MetadataPatch(expected_metadata_version=1, archived=True))
    operation = uuid4()
    recovered = await client.post(f"/runs/{first['run_id']}/recover?request_id={operation}", headers=admin)
    assert recovered.status_code == 202, recovered.text
    value = recovered.json()
    assert value["config"]["resume_from"] == str(checkpoint_id)
    assert value["config"]["recover_exact"] is True
    assert "architecture" not in value["config"]
    assert value["config"]["model_name"] != str(output)
    assert pin(value["run_id"]) == frozen
    again = await client.post(f"/runs/{first['run_id']}/recover?request_id={operation}", headers=admin)
    assert again.json()["run_id"] == value["run_id"]
    with store.session() as db:
        assert q.delete_run(db, UUID(first["run_id"]))
    after_pruning = await client.post(f"/runs/{first['run_id']}/recover?request_id={operation}", headers=admin)
    assert after_pruning.status_code == 202 and after_pruning.json()["run_id"] == value["run_id"]


async def test_request_identity_survives_training_history_pruning(client, sample):
    """A pruned operation cannot be recreated under its old actor/request identity."""
    admin, operation = headers(["administration"]), uuid4()
    payload = request(sample)
    first = (await client.post(f"/train?request_id={operation}", headers=admin, json=payload)).json()
    with store.session() as db:
        db.get(Run, UUID(first["run_id"])).status = RunStatus.cancelled
    assert (await client.delete(f"/runs/{first['run_id']}", headers=admin)).status_code == 200
    response = await client.post(f"/train?request_id={operation}", headers=admin, json=payload)
    assert response.status_code == 404 and response.json()["error"]["code"] == "operation_pruned"
    assert training_count() == 0


async def test_chain_request_identity_never_recreates_a_pruned_waiting_step(client, sample):
    """A chain replay with incomplete retained history refuses the entire operation rather than returning a partial chain."""
    admin, operation = headers(["administration"]), uuid4()
    steps = [{"kind": "train", "config": request(sample)}, {"kind": "train", "config": {}}]
    first = (await client.post(f"/runs/chain?request_id={operation}", headers=admin, json=steps)).json()
    with store.session() as db:
        db.get(Run, UUID(first[1]["run_id"])).status = RunStatus.cancelled
        assert q.delete_run(db, UUID(first[1]["run_id"]))
    response = await client.post(f"/runs/chain?request_id={operation}", headers=admin, json=steps)
    assert response.status_code == 404 and response.json()["error"]["code"] == "operation_pruned"
    assert training_count() == 1


@pytest.mark.parametrize("parameters", [{"unknown": 1}, {"dimensions": "invalid"}, {"context_window": 0}])
async def test_bad_structural_parameters_leave_no_training_job(client, sample, parameters):
    """Resolve public declaration defaults/types/bounds before queue insertion, without importing a model."""
    payload = request(sample)
    payload["architecture"]["parameters"] = parameters
    response = await client.post(f"/train?request_id={uuid4()}", headers=headers(["administration"]), json=payload)
    assert response.status_code == 422, response.text
    assert training_count() == 0


async def test_pinned_waiting_consumer_cannot_enter_a_legacy_handler(client, sample, monkeypatch):
    """An inherited recipe never executes legacy authenticated acquisition while supervised training wiring is pending."""
    def forbidden(*args, **kwargs):
        """Detect any attempt to open a legacy data client or optimizer before a supervised preflight."""
        pytest.fail("pinned recipe entered legacy execution")

    admin = headers(["administration"])
    steps = [{"kind": "train", "config": request(sample)}, {"kind": "train", "config": {}}]
    response = await client.post(f"/runs/chain?request_id={uuid4()}", headers=admin, json=steps)
    assert response.status_code == 202, response.text
    waiting = response.json()[1]
    monkeypatch.setattr(legacy.remote, "Client", forbidden)
    monkeypatch.setattr("torch.optim.AdamW", forbidden)
    with pytest.raises(RuntimeError, match="supervised training adapter"):
        _work(handlers.train, waiting["config"], None, Event(), None, run_id=UUID(waiting["run_id"]))
    with pytest.raises(ValueError, match="supervised training adapter"):
        legacy.run(TrainConfig.model_validate(request(sample)))
