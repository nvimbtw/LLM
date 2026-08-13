"""Who is allowed to call the engine.

The engine binds to loopback, which is not the same as being unreachable — and
stops being any protection at all the moment it moves to its own host. Every
request carries a short-lived token minted by the backend, addressed to this
service by name.

Unset `ENGINE_SERVICE_SECRET` leaves the engine open and says so at startup.
That is the development default, not a fallback: a production process that
loses its secret should refuse callers rather than quietly accept everyone.
"""

import os

import jwt
from fastapi import Header, HTTPException

AUDIENCE = "engine"
ISSUER = "backend"

# Matches the minting side; both processes share a clock, so this only absorbs
# scheduling jitter rather than real skew.
LEEWAY_SECS = 5


def secret() -> str | None:
    return os.environ.get("ENGINE_SERVICE_SECRET") or None


def enforced() -> bool:
    return secret() is not None


def require_service(authorization: str | None = Header(default=None)) -> str:
    """The acting caller, or 401. Returns `"system"` for the backend itself."""
    configured = secret()
    if configured is None:
        return "system"

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

    return claims.get("act", "system")
