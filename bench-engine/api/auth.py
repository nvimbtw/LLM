import os
import jwt
from fastapi import Depends, Header, HTTPException
AUDIENCE = "bench-engine"
ISSUER = "backend"
LEEWAY_SECS = 5
SYSTEM = "system"
def secret() -> str | None:
    return os.environ.get("BENCH_ENGINE_SERVICE_SECRET") or None
def enforced() -> bool:
    return secret() is not None
def require_service(authorization: str | None = Header(default=None)) -> str:
    configured = secret()
    if configured is None:
        return _unverified_act(authorization)
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="missing service token")
    try:
        claims = jwt.decode(authorization.split(" ",1)[1], configured, algorithms=["HS256"], audience=AUDIENCE, issuer=ISSUER, leeway=LEEWAY_SECS)
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=f"invalid service token: {e}") from e
    return claims.get("act", SYSTEM)
def _unverified_act(authorization: str | None) -> str:
    if not authorization or not authorization.lower().startswith("bearer "):
        return SYSTEM
    try:
        claims = jwt.decode(authorization.split(" ",1)[1], options={"verify_signature": False})
    except jwt.PyJWTError:
        return SYSTEM
    return claims.get("act", SYSTEM)
