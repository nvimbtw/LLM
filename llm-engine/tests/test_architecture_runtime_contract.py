"""Shared resident-worker protocol fixtures and transport identity boundaries."""

import copy
import json
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from llm.runtime.protocol import (
    CONTRACT_CHECKS, CONTRACT_MODELS, MAX_RUNTIME_PACKET_BYTES,
    RuntimeEnvelope, RuntimePins,
)

FIXTURES = Path(__file__).resolve().parents[2] / "contracts/modular-architecture/v1/runtime"


def fixture_cases():
    """Load the versioned shared cases also consumed by other language adapters."""
    return [case for path in sorted(FIXTURES.glob("*.json")) for case in json.loads(path.read_text())["cases"]]


@pytest.mark.parametrize("case", fixture_cases(), ids=lambda case: case["name"])
def test_shared_runtime_fixtures(case):
    """Validate every positive and negative record, including stale pins and effective bytes."""
    model = CONTRACT_MODELS[case["model"]]

    def validate():
        """Apply duplicate-safe transport parsing and the optional semantic boundary hook."""
        if "raw" in case:
            value = RuntimeEnvelope.from_packet(case["raw"].encode())
        else:
            value = model.model_validate_json(json.dumps(case["value"]))
        if "check" in case:
            CONTRACT_CHECKS[case["check"]](value, case.get("context", {}))
        return value

    if case["valid"]:
        validated = validate()
        assert model.model_validate_json(validated.model_dump_json()) == validated
    else:
        with pytest.raises((ValueError, ValidationError)):
            validate()


def example_envelope(kind="generate"):
    """Return one representative message from shared fixtures without duplicating its schema."""
    for case in fixture_cases():
        if case["valid"] and case["model"] == "RuntimeEnvelope" and case["value"]["payload"]["type"] == kind:
            return RuntimeEnvelope.model_validate(case["value"])
    raise AssertionError(f"missing fixture: {kind}")


def test_runtime_packet_roundtrip_and_byte_boundaries():
    """Raw transport applies a UTF-8 byte budget before parsing or allocating interpreted payloads."""
    message = example_envelope()
    assert RuntimeEnvelope.from_packet(message.to_packet()) == message
    for packet in (b"", b" " * (MAX_RUNTIME_PACKET_BYTES + 1), b"not-json", b"\xff"):
        with pytest.raises(ValueError):
            RuntimeEnvelope.from_packet(packet)


@pytest.mark.parametrize("name", ["training-bootstrap-final-preflight", "sft-bootstrap-inherited-model-package", "rl-bootstrap-inherited-model-package"])
def test_fresh_training_bootstrap_cannot_replace_loaded_runtime_pins(name):
    """Only an empty worker may enter admitted pretrain/SFT/RL; null pins cannot bypass loaded model authorization."""
    from llm.runtime.worker import WorkerContext

    case = next(value for value in fixture_cases() if value["name"] == name)
    command = RuntimeEnvelope.model_validate(case["value"]).payload
    context = object.__new__(WorkerContext)
    context.pins = None
    context.require_pins(command)
    context.pins = example_envelope().payload.pins
    with pytest.raises(ValueError, match="requires_empty_worker"):
        context.require_pins(command)


def test_nested_duplicate_payload_fields_are_rejected():
    """Duplicate rejection applies inside interpreted command payloads as well as envelopes."""
    raw = example_envelope("shutdown").to_packet().replace(
        b'"reason":"eviction"', b'"reason":"eviction","reason":"idle"',
    )
    with pytest.raises(ValueError, match="duplicate_runtime_field"):
        RuntimeEnvelope.from_packet(raw)


def test_cleanup_certificate_cannot_arrive_from_model_worker():
    """Source/model execution cannot certify its own process-tree/resource cleanup."""
    certificate = example_envelope("cleanup_verified")
    certificate.require_sender("guardian")
    for sender in ("worker", "parent"):
        with pytest.raises(ValueError, match="runtime_sender_mismatch"):
            certificate.require_sender(sender)
    certificate.payload.require_lease(certificate.payload.lease_id)
    with pytest.raises(ValueError, match="stale_runtime_lease"):
        certificate.payload.require_lease(UUID("33333333-3333-4333-8333-333333333333"))


def test_worker_request_and_sequence_replay_are_rejected():
    """Valid schema alone cannot authorize stale worker/request identities or reordered events."""
    message = example_envelope()
    current = {"worker_id": message.worker_id, "request_id": message.request_id, "next_sequence": message.sequence}
    message.require_context(**current)
    for field in current:
        stale = {**current, field: message.sequence + 1 if field == "next_sequence" else UUID(int=0)}
        with pytest.raises(ValueError, match="stale_runtime_message"):
            message.require_context(**stale)


def test_effect_reply_and_actor_authority_belong_to_parent():
    """Parent replies and worker effect requests use distinct authenticated channel roles."""
    request = example_envelope("effect_request")
    request.require_sender("worker")
    reply = example_envelope("effect_reply")
    reply.require_sender("parent")
    with pytest.raises(ValueError):
        reply.require_sender("worker")
    bad = request.model_dump(mode="json")
    bad["payload"]["actor_id"] = "attacker"
    with pytest.raises(ValidationError):
        RuntimeEnvelope.model_validate(bad)


def test_full_training_command_carries_no_tensors_optimizer_or_service_handles():
    """The full lifecycle starts from immutable materializations, not per-operation tensor RPC."""
    train = example_envelope("train")
    train.require_sender("parent")
    assert train.payload.type == "train"
    fields = set(type(train.payload).model_fields)
    assert fields == {"type", "pins", "job_id", "training_kind", "config", "corpus", "recovery"}
    assert all(name not in fields for name in ("actor", "token", "service", "optimizer", "tensor", "url"))


def test_all_configuration_pins_are_checked_before_reuse():
    """Definition, bindings, profile and durable receipt must all match a resident model."""
    current = example_envelope().payload.pins
    assert isinstance(current, RuntimePins)
    current.require_current(current)
    for name in RuntimePins.model_fields:
        data = current.model_dump(mode="json")
        data[name] = "33333333-3333-4333-8333-333333333333" if name == "receipt_id" else "f" * 64
        with pytest.raises(ValueError, match="stale_runtime_identity"):
            RuntimePins.model_validate(data).require_current(current)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_runtime_measurements_and_probabilities_are_refused(value):
    """NaN/Infinity cannot bypass numeric bounds or enter progress/token messages."""
    for kind, field in (("progress", "loss"), ("tokens", "log_probabilities")):
        message = copy.deepcopy(example_envelope(kind).model_dump(mode="json"))
        message["payload"][field] = [value, value, value] if kind == "tokens" else value
        with pytest.raises(ValidationError):
            RuntimeEnvelope.model_validate(message)
