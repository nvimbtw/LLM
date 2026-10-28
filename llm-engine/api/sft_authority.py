"""Parent-only durable SFT input authority, retained across interrupted package adoption without changing enqueue pins."""

from typing import Annotated, Literal
from uuid import UUID
from pathlib import Path
import logging
import shutil
import tempfile

from pydantic import Field, model_validator
from sqlalchemy.orm import Session
import store

from llm import storage
from llm.architecture import package
from llm.architecture.package_integrity import METADATA_BYTES, read_bytes, publish_directory, sync_directory, write_bytes
from llm.architecture.spec import Contract, FinalTokenizer, NonnegativeInt, Sha256, canonical_digest
from llm.architecture.tokenizer_binding import TOKENIZER_BYTES, binding_from_bytes
from llm.config import SftConfig
from llm.orchestrate.modular_sft_data import PreparedSftData
from llm.train.modular_records import SftDataset, TrainingPublication, WarmStartOutcome
from llm.train.sft_tokenizer import marker_history
from llm.train.warm_start import untrained_marker_ids
from store import queries as q
from store.schema import Run

LOG = logging.getLogger(__name__)


class PreparedSftAuthority(Contract):
    """Exact parent-prepared inputs and allowed publication provenance for one immutable queued SFT request."""

    schema_version: Literal[1] = 1
    job_id: UUID
    source_model_id: UUID
    source_definition_id: Sha256
    config_digest: Sha256
    data_fingerprint: Sha256
    tokenizer: FinalTokenizer
    dataset: SftDataset
    requested_rows: Annotated[tuple[NonnegativeInt, ...], Field(max_length=65536)]
    recovered_warm_start: WarmStartOutcome | None = None

    @model_validator(mode="after")
    def bounded_rows(self):
        """Keep prepared marker/row ownership unique and within the exact final vocabulary, including recovered outcomes."""
        rows = self.requested_rows
        recovered = self.recovered_warm_start
        reported = recovered.initialized_ids + recovered.skipped_ids if recovered is not None else ()
        if (len(set(rows)) != len(rows) or not set(rows).issubset(self.dataset.special_ids_seen)
                or any(value >= self.tokenizer.vocabulary_size for value in (*self.dataset.special_ids_seen, *rows, *reported))):
            raise ValueError("SFT preparation has inconsistent marker/row ownership")
        return self


def prepare(job_id: UUID, cfg: SftConfig, saved: package.VerifiedPackage, data: PreparedSftData) -> PreparedSftAuthority:
    """Freeze verified actual rendering/tokenizer facts and source marker authority before exposing a numerical command."""
    metadata = TrainingPublication.model_validate_json(read_bytes(saved.directory / "meta.json", METADATA_BYTES))
    if str(metadata.generation.model_id) != (cfg.resume_from or cfg.base_model):
        raise ValueError("SFT preparation source differs from its original request")
    seen = marker_history(metadata.model_dump(mode="json"), data.data.binding.vocabulary_size)
    used = list(data.dataset.special_ids_seen)
    dataset = data.dataset.model_copy(update={"special_ids_seen": tuple(sorted(set(used) | set(seen or []))),
                                               "marker_history_complete": seen is not None})
    recovered = WarmStartOutcome.model_validate(metadata.result.get("warm_start")) if cfg.resume_from is not None else None
    return PreparedSftAuthority(job_id=job_id, source_model_id=metadata.generation.model_id,
        source_definition_id=saved.manifest.definition_id, config_digest=canonical_digest(cfg.model_dump(mode="json")),
        data_fingerprint=data.data.fingerprint, tokenizer=data.data.binding, dataset=dataset,
        requested_rows=tuple(untrained_marker_ids(used, seen)), recovered_warm_start=recovered)


def read(job_id: UUID) -> PreparedSftAuthority:
    """Read finite parent-owned authority; worker-supplied metadata and derived_from claims cannot replace it."""
    authority = PreparedSftAuthority.model_validate_json(read_bytes(storage.training_preparation_dir(job_id) / "prepared.json", METADATA_BYTES))
    if authority.job_id != job_id:
        raise ValueError("SFT prepared authority belongs to another job")
    return authority


def publish(authority: PreparedSftAuthority) -> None:
    """Fsync and atomically publish one no-replace authority before worker launch; identical retries retain it."""
    raw = authority.model_dump_json().encode()
    if len(raw) > METADATA_BYTES:
        raise ValueError("SFT prepared-input authority exceeds metadata budget")
    destination = storage.training_preparation_dir(authority.job_id)
    if destination.exists():
        if read(authority.job_id) != authority:
            raise ValueError("SFT job already owns different prepared-input authority")
        return
    root = storage.package_staging_dir()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < len(raw) + (512 << 20):
        raise ValueError("SFT authority publication has insufficient staging capacity")
    temporary = Path(tempfile.mkdtemp(prefix="sft-authority-", dir=root))
    try:
        write_bytes(temporary / "prepared.json", raw)
        sync_directory(temporary)
        destination.parent.mkdir(parents=True, exist_ok=True)
        publish_directory(temporary, destination)
        sync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def validate(db: Session, checked: package.VerifiedPackage, value: TrainingPublication, cfg: SftConfig) -> PreparedSftAuthority:
    """Require prepared source/config/tokenizer/data and honest warm-start/rendering provenance before durable adoption."""
    authority = read(value.job_id)
    source = q.get_model(db, authority.source_model_id)
    expected_source = UUID(cfg.resume_from or cfg.base_model)
    if (source is None or source.definition_id != authority.source_definition_id
            or authority.source_model_id != expected_source or authority.config_digest != canonical_digest(cfg.model_dump(mode="json"))
            or checked.manifest.definition.tokenizer != authority.tokenizer
            or value.result.get("data_fingerprint") != authority.data_fingerprint
            or value.result.get("base_model") != cfg.base_model
            or SftDataset.model_validate(value.result.get("dataset")) != authority.dataset):
        raise ValueError("SFT publication differs from parent-prepared input authority")
    raw = read_bytes(checked.directory / "tokenizer.json", TOKENIZER_BYTES)
    actual = binding_from_bytes(authority.tokenizer.tokenizer_id, raw, tuple(authority.tokenizer.reserved_token_ids))
    if actual.model_copy(update={"derived_from": authority.tokenizer.derived_from}) != authority.tokenizer:
        raise ValueError("SFT publication tokenizer differs from its authorized actual bytes")
    rows = WarmStartOutcome.model_validate(value.result.get("warm_start"))
    if type(value.result.get("warm_started")) is not int or value.result["warm_started"] != len(rows.initialized_ids):
        raise ValueError("SFT warm-start count differs from reported row ownership")
    if authority.recovered_warm_start is not None:
        if rows != authority.recovered_warm_start:
            raise ValueError("SFT recovery changed its original warm-start provenance")
    elif set(rows.initialized_ids + rows.skipped_ids) != set(authority.requested_rows):
        raise ValueError("SFT warm-start did not preserve prepared marker authority")
    return authority


def discard(job_id: UUID) -> None:
    """Retire prepared authority only after the complete numerical session and all package adoption have finished."""
    directory = storage.training_preparation_dir(job_id)
    if directory.exists():
        shutil.rmtree(directory)
        sync_directory(directory.parent)


def pending_publications() -> set[UUID] | None:
    """Find unresolved complete-package jobs before retiring authority; unreadable metadata conservatively retains all proofs."""
    pending = set()
    for directory in storage.models_dir().glob("*"):
        if not (directory / "package.json").is_file():
            continue
        try:
            value = TrainingPublication.model_validate_json(read_bytes(directory / "meta.json", METADATA_BYTES))
            with store.session() as db:
                if q.get_model(db, value.generation.model_id) is None:
                    pending.add(value.job_id)
        except (OSError, ValueError) as error:
            LOG.warning("SFT preparation cleanup retained authority for unreadable package %s: %s", directory.name, error)
            return None
    return pending


def reconcile() -> None:
    """Retire terminal/pruned-job proofs after exclusive worker cleanup and package adoption, retaining unresolved or active jobs."""
    pending = pending_publications()
    if pending is None:
        return
    for directory in storage.training_preparations_dir().glob("*"):
        try:
            authority = read(UUID(directory.name))
            with store.session() as db:
                run = db.get(Run, authority.job_id)
                terminal = run is None or run.status.is_terminal
            if terminal and authority.job_id not in pending:
                discard(authority.job_id)
        except (OSError, ValueError) as error:
            LOG.warning("SFT preparation cleanup cannot reconcile %s: %s", directory.name, error)
