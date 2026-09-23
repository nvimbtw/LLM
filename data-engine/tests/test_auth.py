"""The service token seam: who gets in, as whom, and who is turned away."""

import json
import uuid
from pathlib import Path

import jwt
import pytest
from fastapi import HTTPException

from api import auth

SECRET = "a-test-secret"
CONTRACTS = Path(__file__).resolve().parents[2] / "contracts"


def bearer(**claims) -> str:
    payload = {"iss": "backend", "aud": auth.AUDIENCE, "exp": 4102444800, **claims}
    return f"Bearer {jwt.encode(payload, SECRET, algorithm='HS256')}"


def test_without_a_secret_the_service_is_open_and_still_reads_the_caller(monkeypatch):
    monkeypatch.delenv("DATA_ENGINE_SERVICE_SECRET", raising=False)
    who = str(uuid.uuid4())

    assert not auth.enforced()
    assert auth.require_service(bearer(act=who)) == who
    assert auth.require_service(None) == auth.SYSTEM


def test_a_signed_token_names_the_caller_it_acts_for(monkeypatch):
    monkeypatch.setenv("DATA_ENGINE_SERVICE_SECRET", SECRET)
    who = str(uuid.uuid4())

    assert auth.require_service(bearer(act=who)) == who


def test_an_enforcing_service_refuses_a_missing_or_unsigned_token(monkeypatch):
    monkeypatch.setenv("DATA_ENGINE_SERVICE_SECRET", SECRET)

    with pytest.raises(HTTPException) as missing:
        auth.require_service(None)
    assert missing.value.status_code == 401

    with pytest.raises(HTTPException):
        auth.require_service(f"Bearer {jwt.encode({'act': 'x'}, 'wrong-secret')}")


def test_a_token_minted_for_the_llm_engine_is_refused(monkeypatch):
    """The audience claim is the whole point of running two services on one
    secret-shaped seam: one service's token must not open the other."""
    monkeypatch.setenv("DATA_ENGINE_SERVICE_SECRET", SECRET)
    for_engine = jwt.encode(
        {"iss": "backend", "aud": "llm-engine", "act": str(uuid.uuid4()), "exp": 4102444800},
        SECRET,
        algorithm="HS256",
    )

    with pytest.raises(HTTPException) as refused:
        auth.require_service(f"Bearer {for_engine}")
    assert refused.value.status_code == 401


def test_require_owner_refuses_the_backend_acting_as_itself():
    with pytest.raises(HTTPException) as refused:
        auth.require_owner(auth.SYSTEM)
    assert refused.value.status_code == 403


def test_require_owner_returns_the_uuid_it_was_given():
    who = uuid.uuid4()

    assert auth.require_owner(str(who)) == who


def test_the_shared_contract_vector_verifies(monkeypatch):
    """The same fixture crates/auth verifies. Two implementations of one format
    drift silently — the symptom is a 401 in production, long after the change."""
    vector = json.loads((CONTRACTS / "data-engine-token.json").read_text())
    monkeypatch.setenv("DATA_ENGINE_SERVICE_SECRET", vector["secret"])

    assert auth.require_service(f"Bearer {vector['token']}") == vector["expected_act"]


def test_the_engines_vector_is_refused_here(monkeypatch):
    """Belt and braces on the audience split, using the two real fixtures rather
    than a token this test minted for itself."""
    vector = json.loads((CONTRACTS / "service-token.json").read_text())
    monkeypatch.setenv("DATA_ENGINE_SERVICE_SECRET", vector["secret"])

    with pytest.raises(HTTPException):
        auth.require_service(f"Bearer {vector['token']}")
