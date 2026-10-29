"""Run the common heartbeat/authentication contract from each engine's working directory."""

import os
import socket
import subprocess
import sys
import time

import jwt
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect
from websockets.sync.client import connect

import main


@pytest.mark.parametrize("authorized", [False, True])
def test_heartbeat_service_authentication(monkeypatch, authorized):
    """A signed system caller receives pong; an unsigned caller cannot upgrade."""
    audience = os.environ["HEALTH_TEST_AUDIENCE"]
    monkeypatch.setenv(audience.upper().replace("-", "_") + "_SERVICE_SECRET", "test-secret")
    token = jwt.encode({"iss": "backend", "aud": audience, "act": "system", "exp": int(time.time()) + 60}, "test-secret", algorithm="HS256")
    headers = {"Authorization": f"Bearer {token}"} if authorized else {}
    client = TestClient(main.app)
    if not authorized:
        with pytest.raises((WebSocketDisconnect, RuntimeError)):
            with client.websocket_connect("/health/socket", headers=headers):
                pass
        return
    with client.websocket_connect("/health/socket", headers=headers) as socket:
        socket.send_text("ping")
        assert socket.receive_text() == "pong"


def test_real_uvicorn_heartbeat(monkeypatch):
    """A real Uvicorn socket accepts signed heartbeats without starting database workers."""
    audience = os.environ["HEALTH_TEST_AUDIENCE"]
    monkeypatch.setenv(audience.upper().replace("-", "_") + "_SERVICE_SECRET", "test-secret")
    token = jwt.encode({"iss": "backend", "aud": audience, "act": "system", "exp": int(time.time()) + 60}, "test-secret", algorithm="HS256")
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(16)
        port = listener.getsockname()[1]
        process = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--lifespan", "off", "--fd", str(listener.fileno())], pass_fds=(listener.fileno(),), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            deadline = time.monotonic() + 20
            while True:
                try:
                    with connect(f"ws://127.0.0.1:{port}/health/socket", additional_headers={"Authorization": f"Bearer {token}"}, open_timeout=1) as connection:
                        connection.send("ping")
                        assert connection.recv(timeout=2) == "pong"
                    break
                except (TimeoutError, ConnectionError, OSError):
                    if time.monotonic() >= deadline or process.poll() is not None:
                        raise
        finally:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
