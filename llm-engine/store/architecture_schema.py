"""Engine catalog/evidence storage; execution_jobs remains the only lifecycle owner."""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import Boolean, CheckConstraint, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column

from store.schema import Base, Json, Key, _now


class CatalogFields:
    """Common mutable catalog identity/version ownership shared by the two resource families."""

    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(120))
    description: Mapped[str] = mapped_column(Text, default="")
    origin: Mapped[str] = mapped_column(String(12), default="custom")
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    draft_version: Mapped[int] = mapped_column(Integer, default=1)
    draft_hash: Mapped[str] = mapped_column(String(64))
    metadata_version: Mapped[int] = mapped_column(Integer, default=1)
    creator_id: Mapped[uuid.UUID] = mapped_column(Key)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ModelBlock(CatalogFields, Base):
    """Own editable Python bytes; reads/saves do not import or execute them."""

    __tablename__ = "model_blocks"
    __table_args__ = (
        CheckConstraint("origin IN ('builtin','custom')", name="model_blocks_origin_check"),
        CheckConstraint("draft_version >= 1 AND metadata_version >= 1", name="model_blocks_versions_check"),
    )
    draft_source: Mapped[str] = mapped_column(Text)


class Architecture(CatalogFields, Base):
    """Own editable YAML bytes independently of accepted immutable recipes."""

    __tablename__ = "architectures"
    __table_args__ = (
        CheckConstraint("origin IN ('builtin','custom')", name="architectures_origin_check"),
        CheckConstraint("draft_version >= 1 AND metadata_version >= 1", name="architectures_versions_check"),
    )
    draft_yaml: Mapped[str] = mapped_column(Text)


class ValidationEvidence(Base):
    """Retain immutable receipt bytes independently of prunable validation execution history."""

    __tablename__ = "validation_evidence"
    __table_args__ = (UniqueConstraint("source_run_id", name="uq_validation_evidence_source_run"),)
    receipt_id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True)
    identity_hash: Mapped[str] = mapped_column(String(64), index=True)
    receipt: Mapped[dict] = mapped_column(Json)
    source_run_id: Mapped[uuid.UUID] = mapped_column(Key)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class BlockRevision(Base):
    """Freeze source, SDK declarations and concrete acceptance evidence for one block revision."""

    __tablename__ = "block_revisions"
    __table_args__ = (
        UniqueConstraint("block_id", "revision_number", name="uq_block_revision_number"),
        UniqueConstraint("block_id", "content_hash", name="uq_block_revision_content"),
        CheckConstraint("revision_number >= 1", name="block_revision_number_check"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    block_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("model_blocks.id", ondelete="RESTRICT"), index=True)
    revision_number: Mapped[int] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    declaration: Mapped[dict] = mapped_column(Json)
    compatibility: Mapped[dict] = mapped_column(Json)
    acceptance_receipt_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("validation_evidence.receipt_id", ondelete="RESTRICT"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ArchitectureRevision(Base):
    """Freeze canonical recipe/source and dependency digest with retained acceptance evidence."""

    __tablename__ = "architecture_revisions"
    __table_args__ = (
        UniqueConstraint("architecture_id", "revision_number", name="uq_architecture_revision_number"),
        UniqueConstraint("architecture_id", "content_hash", "dependency_digest", name="uq_architecture_revision_content"),
        CheckConstraint("revision_number >= 1", name="architecture_revision_number_check"),
    )
    id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True, default=uuid.uuid4)
    architecture_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("architectures.id", ondelete="RESTRICT"), index=True)
    revision_number: Mapped[int] = mapped_column(Integer)
    yaml: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64))
    recipe: Mapped[dict] = mapped_column(Json)
    dependency_digest: Mapped[str] = mapped_column(String(64))
    acceptance_receipt_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("validation_evidence.receipt_id", ondelete="RESTRICT"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class ArchitectureDependency(Base):
    """Give each pinned direct dependency exactly one block/subarchitecture revision foreign key."""

    __tablename__ = "architecture_dependencies"
    __table_args__ = (
        CheckConstraint("(block_revision_id IS NOT NULL AND child_architecture_revision_id IS NULL) OR (block_revision_id IS NULL AND child_architecture_revision_id IS NOT NULL)", name="architecture_dependency_exactly_one_type"),
    )
    architecture_revision_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("architecture_revisions.id", ondelete="CASCADE"), primary_key=True)
    instance_path: Mapped[str] = mapped_column(String(512), primary_key=True)
    block_revision_id: Mapped[uuid.UUID | None] = mapped_column(Key, ForeignKey("block_revisions.id", ondelete="RESTRICT"), index=True)
    child_architecture_revision_id: Mapped[uuid.UUID | None] = mapped_column(Key, ForeignKey("architecture_revisions.id", ondelete="RESTRICT"), index=True)


class ModelDefinition(Base):
    """Retain semantic model identity and source manifest independently of validation jobs."""

    __tablename__ = "model_definitions"
    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    architecture_revision_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("architecture_revisions.id", ondelete="RESTRICT"), index=True)
    definition: Mapped[dict] = mapped_column(Json)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class DefinitionEvidence(Base):
    """Retain profile-specific evidence referenced by immutable packages sharing one definition."""

    __tablename__ = "definition_evidence"
    definition_id: Mapped[str] = mapped_column(String(64), ForeignKey("model_definitions.id", ondelete="CASCADE"), primary_key=True)
    receipt_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("validation_evidence.receipt_id", ondelete="RESTRICT"), primary_key=True)


class ModelValidation(Base):
    """One validation domain row; job kind/status/rank/progress remain exclusively on Run."""

    __tablename__ = "model_validations"
    job_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("execution_jobs.id", ondelete="CASCADE"), primary_key=True)
    config: Mapped[dict] = mapped_column(Json, default=dict)
    result: Mapped[dict | None] = mapped_column(Json)
    evidence_id: Mapped[uuid.UUID | None] = mapped_column(Key, ForeignKey("validation_evidence.receipt_id", ondelete="RESTRICT"))


class RunArchitecturePin(Base):
    """Freeze enqueue/waiting-step evidence without duplicating final training definition state."""

    __tablename__ = "run_architecture_pins"
    job_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("execution_jobs.id", ondelete="CASCADE"), primary_key=True)
    architecture_revision_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("architecture_revisions.id", ondelete="RESTRICT"), index=True)
    acceptance_receipt_id: Mapped[uuid.UUID] = mapped_column(Key, ForeignKey("validation_evidence.receipt_id", ondelete="RESTRICT"))
    exact_receipt_id: Mapped[uuid.UUID | None] = mapped_column(Key, ForeignKey("validation_evidence.receipt_id", ondelete="RESTRICT"))
    evidence: Mapped[dict] = mapped_column(Json)
    parameters: Mapped[dict] = mapped_column(Json)
    snapshot: Mapped[dict] = mapped_column(Json)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)


class MutationRequest(Base):
    """Deduplicate actor/request operations and retain their stable outcome after job pruning."""

    __tablename__ = "architecture_mutation_requests"
    actor_id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True)
    request_id: Mapped[uuid.UUID] = mapped_column(Key, primary_key=True)
    operation: Mapped[str] = mapped_column(String(32))
    payload_hash: Mapped[str] = mapped_column(String(64))
    outcome: Mapped[dict | None] = mapped_column(Json)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
