"""Bounded project facts and retrieved message provenance supplied by Rust."""

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, StrictInt

from llm.corpus.special import MEMORY_CLOSE as MEMORY_CLOSE, MEMORY_OPEN as MEMORY_OPEN

MAX_MEMORIES = 8
MAX_CONTEXT_SOURCE_CHARS = 65000
PROJECT_CONTEXT_MAX_TOKENS = 512
MEMORY_CONTEXT_MAX_TOKENS = 512
MEMORY_MAX_TOKENS = 256
CONTEXT_ALLOWANCE_DIVISOR = 3


class ProjectContext(BaseModel):
    """Authorized project identity and instructions; lengths count Unicode characters."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    project_id: UUID
    name: str = Field(min_length=1, max_length=120)
    description: str = Field(max_length=2000)
    notes: str = Field(max_length=8000)


class MemoryContext(BaseModel):
    """A source message, its speaker and elapsed age in whole nonnegative seconds."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    memory: str = Field(min_length=1, max_length=2000)
    thread: UUID
    message: UUID
    role: Literal["user", "assistant"]
    age_seconds: StrictInt = Field(ge=0)
