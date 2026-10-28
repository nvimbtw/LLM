"""Version-one resident worker IPC declarations, without execution or service access."""

from __future__ import annotations

import json
from typing import Annotated, Literal
from uuid import UUID

from pydantic import BaseModel, Field, field_validator, model_validator

from llm.architecture.spec import Contract, Sha256

MAX_RUNTIME_PACKET_BYTES = 4096
MAX_RUNTIME_PENDING_MESSAGES = 64
Counter = Annotated[int, Field(strict=True, ge=0, le=(1 << 53) - 1)]
PositiveCounter = Annotated[int, Field(strict=True, ge=1, le=(1 << 53) - 1)]
Seconds = Annotated[float, Field(strict=True, gt=0, le=604800, allow_inf_nan=False)]
Probability = Annotated[float, Field(strict=True, le=0, allow_inf_nan=False)]
Phase = Literal[
    "inspection", "import", "construction", "load", "validation", "capture",
    "forward", "backward", "checkpoint", "generation", "measurement", "cleanup",
]


class RuntimePins(Contract):
    """Exact semantic definition, binding, execution-profile and valid-receipt identity."""

    definition_id: Sha256
    bindings_digest: Sha256
    profile_digest: Sha256
    receipt_id: UUID

    def require_current(self, expected: RuntimePins) -> None:
        """Refuse stale identities before invoking any model or optimizer behavior."""
        if self != expected:
            raise ValueError("stale_runtime_identity")


class RuntimeDeadlines(Contract):
    """Distinct absolute startup/request limits and completed-work training watchdogs."""

    import_seconds: Seconds = 30
    construction_seconds: Seconds = 30
    validation_seconds: Seconds = 120
    training_progress_seconds: Seconds = 300
    inference_seconds: Seconds = 120
    measurement_seconds: Seconds = 120
    effect_seconds: Seconds = 30
    artifact_adoption_seconds: Seconds = 30
    cancel_grace_seconds: Annotated[float, Field(strict=True, gt=0, le=10)] = 0.15
    reap_seconds: Annotated[float, Field(strict=True, gt=0, le=30)] = 2
    serving_idle_seconds: Annotated[float, Field(strict=True, gt=0, le=86400)] = 300


class RuntimeResources(Contract):
    """Parent admission budgets; tree RSS supervision is distinct from per-process AS."""

    process_address_bytes: Annotated[int, Field(strict=True, ge=1 << 20, le=1 << 40)]
    tree_resident_bytes: Annotated[int, Field(strict=True, ge=1 << 20, le=1 << 40)]
    tree_memory_enforcement: Literal["cgroup_v2", "preflight_and_monitor"]
    monitor_interval_seconds: Annotated[float, Field(strict=True, gt=0, le=0.1)] = 0.05
    device: Literal["cpu", "cuda"]
    device_allocator_bytes: Annotated[int, Field(strict=True, ge=0, le=1 << 40)]
    owned_processes: Annotated[int, Field(strict=True, ge=1, le=8)] = 1
    open_files_per_process: Annotated[int, Field(strict=True, ge=8, le=256)] = 64
    artifact_bytes: Annotated[int, Field(strict=True, ge=1, le=1 << 40)]
    total_artifact_bytes: Annotated[int, Field(strict=True, ge=1, le=1 << 40)]

    @model_validator(mode="after")
    def coherent_budgets(self) -> RuntimeResources:
        """Refuse contradictory device and artifact budgets before process admission."""
        if self.device == "cpu" and self.device_allocator_bytes != 0:
            raise ValueError("cpu_has_device_budget")
        if self.device == "cuda" and self.device_allocator_bytes == 0:
            raise ValueError("cuda_requires_device_budget")
        if self.artifact_bytes > self.total_artifact_bytes:
            raise ValueError("artifact_exceeds_total_budget")
        return self


class RuntimePolicy(Contract):
    """Parent-owned bounded transport and resident-worker resource/deadline policy."""

    schema_version: Literal[1] = 1
    packet_bytes: Literal[4096] = MAX_RUNTIME_PACKET_BYTES
    pending_messages: Annotated[int, Field(strict=True, ge=1, le=64)] = MAX_RUNTIME_PENDING_MESSAGES
    retained_progress: Annotated[int, Field(strict=True, ge=1, le=64)] = 64
    progress_interval_seconds: Annotated[float, Field(strict=True, ge=0.1, le=30)] = 0.25
    pending_effects: Literal[1] = 1
    pending_publications: Literal[1] = 1
    deadlines: RuntimeDeadlines = Field(default_factory=RuntimeDeadlines)
    resources: RuntimeResources

    @field_validator("packet_bytes", "pending_effects", "pending_publications", mode="before")
    @classmethod
    def integer_literal_policy(cls, value: object) -> object:
        """Refuse booleans that compare equal to fixed integer transport policy values."""
        if type(value) is not int:
            raise ValueError("runtime_policy_requires_integer")
        return value


class RuntimeArtifact(Contract):
    """Immutable bounded materialization grant; the parent resolves IDs, never worker paths."""

    artifact_id: UUID
    sha256: Sha256
    size_bytes: PositiveCounter
    format: Literal[
        "source_snapshot_v1", "validation_snapshot_v1", "definition_v1", "model_package_v1",
        "training_config_v1", "corpus_v1", "tokenizer_v1", "span_table_v1", "loss_mask_v1", "sft_probe_v1", "prompt_v1", "measurement_v1", "effect_data_v1",
        "validation_result_v1", "training_result_v1", "training_progress_v1", "generation_result_v1", "measurement_result_v1", "checkpoint_v1",
    ]

    @model_validator(mode="after")
    def source_byte_budget(self) -> RuntimeArtifact:
        """Apply the version-one source-snapshot ceiling before materializing any code."""
        if self.format == "source_snapshot_v1" and self.size_bytes > 262144:
            raise ValueError("runtime_source_budget")
        return self


class Inspect(Contract):
    """Inspect bounded source declarations without constructing a custom model."""

    type: Literal["inspect"]
    snapshot: RuntimeArtifact

    @model_validator(mode="after")
    def source_snapshot(self) -> Inspect:
        """Require source metadata inspection input rather than an executable result blob."""
        require_format(self.snapshot, "source_snapshot_v1")
        return self


class Validate(Contract):
    """Perform all source import, construction, capture and numerical checks in a worker."""

    type: Literal["validate"]
    snapshot: RuntimeArtifact

    @model_validator(mode="after")
    def validation_snapshot(self) -> Validate:
        """Require a fully captured immutable validation request snapshot."""
        require_format(self.snapshot, "validation_snapshot_v1")
        return self


class Load(Contract):
    """Load one pinned definition/package into the reusable serving worker."""

    type: Literal["load"]
    pins: RuntimePins
    definition: RuntimeArtifact
    package: RuntimeArtifact | None = None

    @model_validator(mode="after")
    def model_inputs(self) -> Load:
        """Require typed definition and optional self-contained saved model materializations."""
        require_format(self.definition, "definition_v1")
        if self.package is not None:
            require_format(self.package, "model_package_v1")
        return self


class Train(Contract):
    """Run the complete pretrain/SFT/RL lifecycle with resident model and optimizer state."""

    type: Literal["train"]
    pins: RuntimePins | None = None
    job_id: UUID
    training_kind: Literal["pretrain", "sft", "rl"]
    config: RuntimeArtifact
    corpus: RuntimeArtifact | None = None
    recovery: RuntimeArtifact | None = None

    @model_validator(mode="after")
    def training_inputs(self) -> Train:
        """Allow prepared pretraining or inherited SFT/RL to establish final pins inside the admitted job.

        Resident work retains exact loaded pins and its separate checkpoint
        format. Package and checkpoint grants cannot substitute for one another.
        """
        require_format(self.config, "training_config_v1")
        if self.corpus is not None:
            require_format(self.corpus, "corpus_v1")
        if self.recovery is not None:
            require_format(self.recovery, "model_package_v1" if self.pins is None else "checkpoint_v1")
        if self.pins is None and self.training_kind == "rl":
            if self.recovery is None or self.corpus is not None:
                raise ValueError("runtime_rl_bootstrap_requires_saved_package_without_corpus")
        elif self.pins is None and (self.corpus is None or self.training_kind == "sft" and self.recovery is None):
            raise ValueError("runtime_training_bootstrap_requires_corpus_and_inherited_sft_package")
        return self


class Generate(Contract):
    """Generate a bounded request using worker-owned cache, reset at request completion."""

    type: Literal["generate"]
    pins: RuntimePins
    prompt: RuntimeArtifact
    max_tokens: Annotated[int, Field(strict=True, ge=1, le=65536)]
    seed: Counter
    temperature: Annotated[float, Field(strict=True, ge=0, le=10)]

    @model_validator(mode="after")
    def prompt_input(self) -> Generate:
        """Require a worker-validated tokenized prompt rather than an arbitrary materialization."""
        require_format(self.prompt, "prompt_v1")
        return self


class Measure(Contract):
    """Run one bounded measurement in the model worker under the same admission lease."""

    type: Literal["measure"]
    pins: RuntimePins
    measurement: RuntimeArtifact

    @model_validator(mode="after")
    def measurement_input(self) -> Measure:
        """Require a bounded typed measurement request interpreted inside the worker."""
        require_format(self.measurement, "measurement_v1")
        return self


class Cancel(Contract):
    """Cooperatively cancel a request; this acknowledgement cannot release admission."""

    type: Literal["cancel"]
    target_request_id: UUID
    reason: Literal["user", "disconnect", "deadline", "shutdown", "resource_limit"]


class Shutdown(Contract):
    """Evict resident model state and close the worker after guardian-verified cleanup."""

    type: Literal["shutdown"]
    reason: Literal["eviction", "shutdown", "idle", "quarantine"]


class EffectData(RuntimeArtifact):
    """A versioned effect payload interpreted by its specific actor-bound parent adapter."""

    format: Literal["effect_data_v1"]


class DatasetReadEffect(Contract):
    """Read dataset metadata/pages through Rust using a dataset UUID and bounded query data."""

    operation: Literal["dataset_read"]
    dataset_id: UUID
    payload: EffectData | None = None


class GymCatalogueEffect(Contract):
    """List authorized gym metadata before resolving a named training lane."""

    operation: Literal["gym_catalogue"]


class GymCreateEffect(Contract):
    """Create an RL run for an opaque gym ID, preserving identifiers such as math/general."""

    operation: Literal["gym_create"]
    gym_id: Annotated[str, Field(min_length=1, max_length=128)]
    payload: EffectData


class GymReadEffect(Contract):
    """Read an authorized gym run by its UUID without exposing service URLs."""

    operation: Literal["gym_read"]
    run_id: UUID
    payload: EffectData | None = None


class GymStepEffect(Contract):
    """Issue one episode tool step; the operation identity prevents replay after uncertainty."""

    operation: Literal["gym_step"]
    episode_id: UUID
    payload: EffectData


class GymFinishEffect(Contract):
    """Finish and grade an episode exactly once with an explicit final-reply payload."""

    operation: Literal["gym_finish"]
    episode_id: UUID
    payload: EffectData


class GymAbandonEffect(Contract):
    """Close an unavailable episode without grading it or inventing a zero reward."""

    operation: Literal["gym_abandon"]
    episode_id: UUID


class GymStatusEffect(Contract):
    """Update the authorized run status using its UUID and typed status payload."""

    operation: Literal["gym_status"]
    run_id: UUID
    payload: EffectData


EffectOperation = Annotated[
    DatasetReadEffect | GymCatalogueEffect | GymCreateEffect | GymReadEffect
    | GymStepEffect | GymFinishEffect | GymAbandonEffect | GymStatusEffect,
    Field(discriminator="operation"),
]


class EffectRequest(Contract):
    """Request a parent-mediated Rust effect with exact operation-specific resource identity."""

    type: Literal["effect_request"]
    operation_id: UUID
    effect: EffectOperation


class EffectReply(Contract):
    """Return an issued operation's outcome; uncertain writes are terminal and never replayed."""

    type: Literal["effect_reply"]
    operation_id: UUID
    outcome: Literal["completed", "failed", "uncertain"]
    result: RuntimeArtifact | None = None
    error_code: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")] | None = None

    @model_validator(mode="after")
    def outcome_payload(self) -> EffectReply:
        """Refuse fabricated success payloads for failed or uncertain remote effects."""
        if self.outcome != "completed" and (self.result is not None or self.error_code is None):
            raise ValueError("invalid_effect_outcome")
        if self.outcome == "completed" and self.error_code is not None:
            raise ValueError("success_has_error")
        if self.result is not None:
            require_format(self.result, "effect_data_v1")
        return self


class Progress(Contract):
    """Bounded completed work; actual batch/iteration advancement refreshes training watchdogs."""

    type: Literal["progress"]
    phase: Phase
    completed: Counter
    total: PositiveCounter | None = None
    unit: Literal["bytes", "nodes", "batches", "steps", "iterations", "tokens", "phases"]
    loss: Annotated[float, Field(strict=True, allow_inf_nan=False)] | None = None

    @model_validator(mode="after")
    def bounded_completion(self) -> Progress:
        """Reject reports whose completed work exceeds the stated finite total."""
        if self.total is not None and self.completed > self.total:
            raise ValueError("progress_exceeds_total")
        return self


class Tokens(Contract):
    """A bounded streaming chunk; transcript storage remains in the Rust backend."""

    type: Literal["tokens"]
    token_ids: Annotated[tuple[Counter, ...], Field(min_length=1, max_length=32)]
    log_probabilities: Annotated[tuple[Probability, ...], Field(min_length=1, max_length=32)] | None = None

    @model_validator(mode="after")
    def matching_probabilities(self) -> Tokens:
        """Require one finite nonpositive policy log probability for each reported token."""
        if self.log_probabilities is not None:
            if len(self.log_probabilities) != len(self.token_ids):
                raise ValueError("invalid_token_probabilities")
        return self


class ArtifactReady(Contract):
    """Announce a complete staged artifact; availability waits for parent adoption acknowledgement."""

    type: Literal["artifact_ready"]
    publication_id: UUID
    artifact: RuntimeArtifact
    completed_step: Counter
    final: Annotated[bool, Field(strict=True)]

    @model_validator(mode="after")
    def publishable_artifact(self) -> ArtifactReady:
        """Refuse source/input grants presented as complete publishable worker output."""
        if self.artifact.format not in {
            "checkpoint_v1", "model_package_v1", "validation_result_v1",
            "training_result_v1", "training_progress_v1", "measurement_result_v1",
        }:
            raise ValueError("runtime_output_format")
        return self


class ArtifactAdopted(Contract):
    """Acknowledge parent verification, publication and durable registration of a staged artifact."""

    type: Literal["artifact_adopted"]
    publication_id: UUID
    artifact_id: UUID
    outcome: Literal["adopted", "rejected"]


class Completed(Contract):
    """Finish one operation, retaining any reusable serving model until explicit eviction."""

    type: Literal["completed"]
    outcome: Literal["ok", "cancelled"]
    result: RuntimeArtifact | None = None

    @model_validator(mode="after")
    def result_artifact(self) -> Completed:
        """Return only typed operation results; model/checkpoint publication uses adoption."""
        if self.result is not None and self.result.format not in {
            "validation_result_v1", "training_result_v1", "generation_result_v1", "measurement_result_v1",
        }:
            raise ValueError("runtime_result_format")
        return self


class Failed(Contract):
    """Report a bounded classified failure; source diagnostics remain administrator-only."""

    type: Literal["failed"]
    phase: Phase
    code: Annotated[str, Field(pattern=r"^[a-z][a-z0-9_]{0,63}$")]
    message: Annotated[str, Field(max_length=1024)]


class CleanupVerified(Contract):
    """Guardian-only tree cleanup certificate, bound to the engine's current lease identity."""

    type: Literal["cleanup_verified"]
    lease_id: UUID
    outcome: Literal["reaped", "uncertain"]
    owned_processes_remaining: Annotated[int, Field(strict=True, ge=0, le=8)]

    @model_validator(mode="after")
    def no_remaining_processes(self) -> CleanupVerified:
        """Refuse successful cleanup while any owned process may still hold resources."""
        if self.outcome == "reaped" and self.owned_processes_remaining != 0:
            raise ValueError("cleanup_has_live_processes")
        return self

    def require_lease(self, expected: UUID) -> None:
        """Reject cleanup evidence for another admission lease, including a prior worker lifetime."""
        if self.lease_id != expected:
            raise ValueError("stale_runtime_lease")


RuntimePayload = Annotated[
    Inspect | Validate | Load | Train | Generate | Measure | Cancel | Shutdown | EffectRequest
    | EffectReply | Progress | Tokens | ArtifactReady | ArtifactAdopted | Completed | Failed | CleanupVerified,
    Field(discriminator="type"),
]


class RuntimeEnvelope(Contract):
    """One ordered versioned message, bound to a fresh parent-created worker/session identity."""

    schema_version: Literal[1] = 1
    worker_id: UUID
    request_id: UUID
    sequence: Counter
    payload: RuntimePayload

    def require_context(self, *, worker_id: UUID, request_id: UUID, next_sequence: int) -> None:
        """Reject another worker/request or a replay before the owning supervisor advances state."""
        if self.worker_id != worker_id or self.request_id != request_id or self.sequence != next_sequence:
            raise ValueError("stale_runtime_message")

    def require_sender(self, sender: Literal["parent", "worker", "guardian"]) -> None:
        """Reject messages forbidden on the authenticated sender's separate transport channel."""
        allowed = {
            "parent": {"inspect", "validate", "load", "train", "generate", "measure", "cancel",
                       "shutdown", "effect_reply", "artifact_adopted"},
            "worker": {"effect_request", "progress", "tokens", "artifact_ready", "completed", "failed"},
            "guardian": {"cleanup_verified"},
        }
        if self.payload.type not in allowed[sender]:
            raise ValueError("runtime_sender_mismatch")

    def to_packet(self) -> bytes:
        """Encode one JSON packet and reject effective UTF-8 bytes beyond the transport budget."""
        packet = self.model_dump_json().encode("utf-8")
        if len(packet) > MAX_RUNTIME_PACKET_BYTES:
            raise ValueError("runtime_packet_budget")
        return packet

    @classmethod
    def from_packet(cls, packet: bytes) -> RuntimeEnvelope:
        """Decode bounded UTF-8 JSON, refusing duplicate fields before schema interpretation."""
        if not packet or len(packet) > MAX_RUNTIME_PACKET_BYTES:
            raise ValueError("runtime_packet_budget")
        json.loads(packet, object_pairs_hook=unique_fields)
        return cls.model_validate_json(packet)


def unique_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject duplicate JSON object fields instead of silently selecting one value."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_runtime_field")
        result[key] = value
    return result


def require_format(artifact: RuntimeArtifact, expected: str) -> None:
    """Reject artifact-role confusion before any blob is loaded or source is imported."""
    if artifact.format != expected:
        raise ValueError("runtime_artifact_format")


def check_runtime_identity(value: BaseModel, context: dict[str, object]) -> None:
    """Apply shared-fixture stale-identity expectations against the parent-pinned identity."""
    if not isinstance(value, RuntimePins):
        raise ValueError("runtime_identity_requires_pins")
    value.require_current(RuntimePins.model_validate(context["expected"]))


def check_runtime_packet(value: BaseModel, context: dict[str, object]) -> None:
    """Apply the effective transport-byte ceiling in the shared contract fixture harness."""
    if not isinstance(value, RuntimeEnvelope):
        raise ValueError("runtime_packet_requires_envelope")
    value.to_packet()


CONTRACT_MODELS: dict[str, type[BaseModel]] = {
    "RuntimePins": RuntimePins,
    "RuntimePolicy": RuntimePolicy,
    "RuntimeDeadlines": RuntimeDeadlines,
    "RuntimeResources": RuntimeResources,
    "RuntimeArtifact": RuntimeArtifact,
    "RuntimeEnvelope": RuntimeEnvelope,
}
CONTRACT_CHECKS = {
    "runtime_identity": check_runtime_identity,
    "runtime_packet": check_runtime_packet,
}
