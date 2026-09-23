"""Who is allowed to call the data engine.

The same seam the LLM engine uses, addressed to a different name. Every request
carries a short-lived token minted by the backend; a token minted for `llm-engine` is
refused here and vice versa, which is the point of the audience claim.

Unset `DATA_ENGINE_SERVICE_SECRET` leaves the service open and says so at
startup. That is the development default, not a fallback: a production process
that loses its secret should refuse callers rather than quietly accept everyone.
"""

import os
import uuid

import jwt
from fastapi import Depends, Header, HTTPException

AUDIENCE = "data-engine"
ISSUER = "backend"

# Matches the minting side; both processes share a clock, so this only absorbs
# scheduling jitter rather than real skew.
LEEWAY_SECS = 5

SYSTEM = "system"


def secret() -> str | None:
    return os.environ.get("DATA_ENGINE_SERVICE_SECRET") or None


def enforced() -> bool:
    return secret() is not None


def require_service(authorization: str | None = Header(default=None)) -> str:
    """The acting caller, or 401. `"system"` is the backend acting as itself."""
    configured = secret()
    if configured is None:
        # Still read the claim when one is offered: without a secret the service
        # trusts the caller either way, and dropping `act` would make every
        # owner-scoped route unusable in development.
        return _unverified_act(authorization)

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing service token")

    try:
        claims = jwt.decode(
            authorization.split(" ", 1)[1],
            configured,
            algorithms=["HS256"],
            audience=AUDIENCE,
            issuer=ISSUER,
            leeway=LEEWAY_SECS,
        )
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"invalid service token: {e}") from e

    return claims.get("act", SYSTEM)


def _unverified_act(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return SYSTEM
    try:
        claims = jwt.decode(
            authorization.split(" ", 1)[1], options={"verify_signature": False}
        )
    except jwt.PyJWTError:
        return SYSTEM
    return claims.get("act", SYSTEM)


def require_owner(caller: str = Depends(require_service)) -> uuid.UUID:
    """Whose data this request touches.

    The backend acting as itself owns no datasets, so a route that needs an owner
    refuses it rather than falling back to an unscoped read.
    """
    try:
        return uuid.UUID(caller)
    except ValueError:
        raise HTTPException(status_code=403, detail="this route needs a user to act for")
