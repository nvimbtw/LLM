"""Trusted launch records with no actors, credentials, service objects or custom instances."""

from pathlib import Path
from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field

from llm.architecture.spec import Contract
from llm.runtime.protocol import RuntimePolicy


class WorkerBootstrap(Contract):
    """Pass parent-approved worker policy and a trusted engine backend factory by name."""

    schema_version: Literal[1] = 1
    worker_id: UUID
    lease_id: UUID
    startup_request_id: UUID
    backend: Annotated[str, Field(pattern=r"^(?:llm|tests)\.[a-zA-Z0-9_.]+:[a-zA-Z][a-zA-Z0-9_]*$", max_length=256)]
    metadata_only: bool = False
    policy: RuntimePolicy


def read_bootstrap(directory: Path) -> WorkerBootstrap:
    """Load only bounded engine-owned JSON before backend import; never deserialize Python objects."""
    path = directory / "bootstrap.json"
    if path.stat().st_size > 16384:
        raise ValueError("runtime_bootstrap_budget")
    return WorkerBootstrap.model_validate_json(path.read_bytes())
