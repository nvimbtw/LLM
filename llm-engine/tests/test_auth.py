"""The service-token seam: who the LLM engine accepts."""

import asyncio
from datetime import UTC, datetime, timedelta

import jwt
import pytest
from httpx import ASGITransport, AsyncClient

from api import auth

SECRET = "service-secret-for-tests"


@pytest.fixture
def enforced(monkeypatch):
    monkeypatch.setenv("LLM_ENGINE_SERVICE_SECRET", SECRET)
    return SECRET


@pytest.fixture
async def client(engine_client):
    return engine_client


def token(
    *,
    secret: str = SECRET,
    audience: str = auth.AUDIENCE,
    issuer: str = auth.ISSUER,
    act: str = "system",
    age: timedelta = timedelta(seconds=60),
) -> str:
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "act": act,
            "scope": [],
            "exp": datetime.now(UTC) + age,
        },
        secret,
        algorithm="HS256",
    )


def bearer(value: str) -> dict:
    return {"Authorization": f"Bearer {value}"}


async def test_an_unset_secret_leaves_the_engine_open(client, monkeypatch):
    """The development default, and it announces itself at startup."""
    monkeypatch.delenv("LLM_ENGINE_SERVICE_SECRET", raising=False)
    assert auth.enforced() is False
    assert (await client.get("/")).status_code == 200


async def test_an_open_engine_still_reads_who_a_call_acts_for(client, monkeypatch):
    """Without a secret the LLM engine trusts the caller either way, and dropping
    `act` would make every owner-scoped route unusable in development."""
    import uuid

    monkeypatch.delenv("LLM_ENGINE_SERVICE_SECRET", raising=False)
    who = str(uuid.uuid4())

    assert auth.require_service(f"Bearer {token(act=who)}") == who
    assert auth.require_service(None) == auth.SYSTEM
    assert auth.require_service("Bearer not-a-token") == auth.SYSTEM
    assert auth.require_service("Basic whatever") == auth.SYSTEM


async def test_an_owner_scoped_route_needs_someone_to_act_for(client, monkeypatch):
    import uuid

    monkeypatch.delenv("LLM_ENGINE_SERVICE_SECRET", raising=False)
    acting = bearer(token(act=str(uuid.uuid4())))

    assert (await client.get("/chats")).status_code == 403
    assert (await client.get("/chats", headers=acting)).status_code == 200


async def test_a_configured_secret_refuses_an_unsigned_request(client, enforced):
    response = await client.get("/")
    assert response.status_code == 401
    assert "missing service token" in response.json()["detail"]


async def test_a_valid_token_is_accepted(client, enforced):
    assert (await client.get("/", headers=bearer(token()))).status_code == 200


async def test_the_acting_user_is_carried_through(enforced):
    """The LLM engine learns who the call is for without ever seeing their token."""
    import uuid

    who = str(uuid.uuid4())
    assert auth.require_service(f"Bearer {token(act=who)}") == who


@pytest.mark.parametrize(
    "bad,why",
    [
        (dict(secret="wrong-secret"), "signed with another key"),
        (dict(audience="scraper"), "addressed to a different service"),
        (dict(issuer="somebody-else"), "issued by someone else"),
        (dict(age=timedelta(seconds=-auth.LEEWAY_SECS - 1)), "expired"),
    ],
)
async def test_a_token_is_refused_when(client, enforced, bad, why):
    response = await client.get("/", headers=bearer(token(**bad)))
    assert response.status_code == 401, why


async def test_a_malformed_header_is_refused(client, enforced):
    for value in ["", "Bearer", "Basic abc", token()]:
        response = await client.get("/", headers={"Authorization": value})
        assert response.status_code == 401, value


async def test_every_http_route_is_covered(client, enforced):
    """Applied to the app, so a new route is authenticated because it exists."""
    import main

    unguarded = []
    for route in main.app.routes:
        methods = getattr(route, "methods", None)
        if not methods or "{" in route.path:
            continue
        method = "GET" if "GET" in methods else "POST"
        if (await client.request(method, route.path)).status_code != 401:
            unguarded.append(f"{method} {route.path}")
    assert not unguarded


async def test_the_schema_is_not_served(client, enforced):
    """FastAPI's own routes bypass app dependencies, so /openapi.json would
    hand the whole surface to an unauthenticated caller."""
    for path in ["/openapi.json", "/docs", "/redoc"]:
        assert (await client.get(path)).status_code == 404, path


def test_the_shared_contract_vector_verifies(monkeypatch):
    """The same fixture crates/auth verifies. Two implementations of one format
    drift silently — the symptom is a 401 in production, long after the change."""
    import json
    from pathlib import Path

    vector = json.loads(
        (Path(__file__).resolve().parents[2] / "contracts/service-token.json").read_text()
    )
    monkeypatch.setenv("LLM_ENGINE_SERVICE_SECRET", vector["secret"])
    assert auth.require_service(f"Bearer {vector['token']}") == vector["expected_act"]
