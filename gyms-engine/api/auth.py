import os

import jwt
from fastapi import Depends, HTTPException, Request

AUDIENCE = "gyms-engine"


def enforced() -> bool:
    return bool(os.environ.get("GYMS_ENGINE_SERVICE_SECRET"))


def require_service(request: Request) -> None:
    secret = os.environ.get("GYMS_ENGINE_SERVICE_SECRET")
    if not secret:
        return
    header = request.headers.get("authorization") or request.headers.get("Authorization")
    if not header or not header.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing bearer token")
    token = header.removeprefix("Bearer ").strip()
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], audience=AUDIENCE)
    except jwt.PyJWTError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e
    if payload.get("aud") != AUDIENCE:
        raise HTTPException(status_code=401, detail="wrong audience")
