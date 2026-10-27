"""Versioned data-only preparation records crossing the parent/training-worker boundary."""

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, JsonValue, StrictBool, field_serializer, model_validator

from llm.architecture.spec import Contract, NonnegativeInt, PositiveInt, Sha256, canonical_digest
from llm.architecture.parse import parse_recipe
from llm.architecture.package_integrity import PackageBudget
from llm.architecture.resolve import parameter_values
from llm.architecture.validation_records import ValidationSnapshot
from llm.config import RlConfig, SftConfig, TrainConfig
from llm.runtime.protocol import RuntimeArtifact, require_format
from llm.train.model_preflight import sft_loop_config, training_profile
from llm.train.progress import Checkpoint

MAX_TRAINING_DOCUMENTS = 1 << 24


class DataStage(Contract):
    """Name and epoch count for one stage whose membership lives in the packed span table."""

    name: Annotated[str, Field(max_length=256)] | None = None
    epochs: PositiveInt


class CorpusPlan(Contract):
    """Exact packed-data facts and private span grant; no parent storage paths cross IPC."""

    tokens: PositiveInt
    spans: RuntimeArtifact
    names: Annotated[tuple[Annotated[str, Field(max_length=256)], ...], Field(max_length=4096)]
    stages: Annotated[tuple[DataStage, ...], Field(max_length=128)]
    eot_id: NonnegativeInt
    corpus_key: Annotated[str, Field(min_length=1, max_length=255)]
    fingerprint: Sha256
    shuffle: StrictBool
    seed: NonnegativeInt
    token_counts: Annotated[tuple[NonnegativeInt, ...], Field(max_length=65536)] | None = None
    special_ids_seen: Annotated[tuple[NonnegativeInt, ...], Field(max_length=65536)] | None = None
    dtype: Literal["uint16"] = "uint16"

    @model_validator(mode="after")
    def span_format(self):
        """Reject unrelated grants before loading the structured, non-pickled span table."""
        require_format(self.spans, "span_table_v1")
        return self


class TrainingStorage(Contract):
    """Parent-owned per-generation serialized package ceilings, separate from runtime aggregate disk admission."""

    package_bytes: Annotated[PositiveInt, Field(le=32 << 30)] = 32 << 30
    file_bytes: Annotated[PositiveInt, Field(le=16 << 30)] = 16 << 30

    @model_validator(mode="after")
    def coherent_limits(self):
        """Reject a per-file ceiling larger than the complete package allowance."""
        if self.file_bytes > self.package_bytes:
            raise ValueError("training package file budget exceeds aggregate budget")
        return self

    def package_budget(self) -> PackageBudget:
        """Translate exact typed limits to the existing bounded package serializer/verifier."""
        return PackageBudget(total_bytes=self.package_bytes, file_bytes=self.file_bytes)


class SftDataset(Contract):
    """Bounded measured rendering provenance; dataset records are passed through for presentation."""

    sets: Annotated[tuple[dict[str, JsonValue], ...], Field(max_length=4096)]
    reasoning: Literal["keep", "strip"]
    conversations: PositiveInt
    total_tokens: PositiveInt
    trainable_tokens: NonnegativeInt
    skipped: NonnegativeInt
    special_ids_seen: Annotated[tuple[NonnegativeInt, ...], Field(max_length=65536)]
    marker_history_complete: StrictBool | None = None

    @model_validator(mode="after")
    def aligned_counts(self):
        """Refuse impossible target counts and ambiguous duplicate marker provenance."""
        if self.trainable_tokens > self.total_tokens or len(set(self.special_ids_seen)) != len(self.special_ids_seen):
            raise ValueError("SFT rendering counts or marker identities are inconsistent")
        return self


class SftInputs(Contract):
    """Original inherited request, exact aligned mask and optional bounded reference-probe grants."""

    request: SftConfig
    mask: RuntimeArtifact
    dataset: SftDataset
    pretrain_probe: RuntimeArtifact | None = None
    chat_probe: RuntimeArtifact | None = None
    reference: RuntimeArtifact | None = None

    @model_validator(mode="after")
    def formats(self):
        """Keep mask, numeric probes and model packages distinct before any worker reads them."""
        require_format(self.mask, "loss_mask_v1")
        for probe in (self.pretrain_probe, self.chat_probe):
            if probe is not None:
                require_format(probe, "sft_probe_v1")
        if self.reference is not None:
            require_format(self.reference, "model_package_v1")
        if (self.chat_probe is not None or self.reference is not None) and not self.request.measure_base_kl:
            raise ValueError("SFT reference grants require the requested KL measurement")
        if self.request.resume_from is not None and self.chat_probe is not None and self.reference is None:
            raise ValueError("SFT recovery KL requires the original frozen base package")
        UUID(self.request.base_model)
        if self.request.resume_from is not None:
            UUID(self.request.resume_from)
        return self


class WarmStartOutcome(Contract):
    """Published token-row ownership with bounded unique IDs, independent of arbitrary model attribute names."""

    status: Literal["initialized", "skipped_unsupported"]
    initialized_ids: Annotated[tuple[NonnegativeInt, ...], Field(max_length=65536)] = ()
    skipped_ids: Annotated[tuple[NonnegativeInt, ...], Field(max_length=65536)] = ()

    @model_validator(mode="after")
    def unique_rows(self):
        """Require disjoint row ownership and preserve unsupported as an unchanged-weight outcome."""
        rows = self.initialized_ids + self.skipped_ids
        if len(rows) != len(set(rows)) or self.status == "skipped_unsupported" and self.initialized_ids:
            raise ValueError("SFT warm-start row ownership is inconsistent")
        return self


class TrainingPreparation(Contract):
    """Frozen source/final bindings and authorized packed inputs for one complete pretrain request."""

    schema_version: Literal[1] = 1
    config: TrainConfig
    validation: ValidationSnapshot
    tokenizer: RuntimeArtifact
    data: CorpusPlan
    storage: TrainingStorage = TrainingStorage()
    sft: SftInputs | None = None

    @property
    def training_kind(self) -> Literal["pretrain", "sft"]:
        """Identify the typed full-lifecycle variation without maintaining a second writable mode field."""
        return "sft" if self.sft is not None else "pretrain"

    @property
    def request_config(self) -> SftConfig | TrainConfig:
        """Return the original public request for immutable metadata, separately from effective loop options."""
        return self.sft.request if self.sft is not None else self.config

    @field_serializer("config")
    def stated_config(self, value: TrainConfig) -> dict:
        """Keep legacy defaults out of selected-recipe serialization so roundtrips retain one structural authority."""
        return value.model_dump(mode="json", exclude_unset=True)

    @model_validator(mode="after")
    def final_bindings(self):
        """Require a causal revision/final tokenizer preflight with matching data and request identities."""
        require_format(self.tokenizer, "tokenizer_v1")
        request = self.validation.request
        if (self.validation.recipe is None or request.mode != "full" or request.subject.kind != "revision"
                or request.subject.reference.kind != "architecture" or request.profile is None or request.context_window is None
                or request.tokenizer is None or request.tokenizer.kind != "final"):
            raise ValueError("training requires a final causal revision snapshot")
        if self.tokenizer.sha256 != request.tokenizer.fingerprint or self.data.eot_id >= request.tokenizer.vocabulary_size:
            raise ValueError("training tokenizer bytes or separator differ from finalized bindings")
        if self.data.token_counts is not None and len(self.data.token_counts) != request.tokenizer.vocabulary_size:
            raise ValueError("training token counts differ from the finalized vocabulary")
        if self.data.special_ids_seen is not None and (len(set(self.data.special_ids_seen)) != len(self.data.special_ids_seen)
                or any(value >= request.tokenizer.vocabulary_size for value in self.data.special_ids_seen)):
            raise ValueError("training marker history differs from the finalized vocabulary")
        if request.profile != training_profile(self.config, request.context_window, request.profile.device):
            raise ValueError("training preflight differs from the effective configuration/input profile")
        if self.sft is not None:
            if self.config.model_dump() != sft_loop_config(self.sft.request, request.context_window).model_dump():
                raise ValueError("SFT effective loop options differ from the original inherited request")
            if (self.sft.mask.size_bytes != self.data.tokens or self.sft.dataset.total_tokens != self.data.tokens
                    or any(value >= request.tokenizer.vocabulary_size for value in self.sft.dataset.special_ids_seen)):
                raise ValueError("SFT mask or marker facts differ from the prepared corpus")
        selection = self.config.architecture
        if selection is not None:
            values = parameter_values(parse_recipe(self.validation.recipe).parameters, selection.parameters)
            if selection.revision_id != request.subject.reference.revision_id or canonical_digest(values) != canonical_digest(request.parameters):
                raise ValueError("training source/parameters differ from the submitted selection")
        return self


class TrainingGeneration(Contract):
    """Bound the durable checkpoint frame before interpreting any worker-supplied model identity or position."""

    model_id: UUID
    index: NonnegativeInt
    step: PositiveInt
    total_steps: PositiveInt
    epoch: PositiveInt
    stage: NonnegativeInt
    stage_name: Annotated[str, Field(max_length=256)] | None
    path: Literal["model.pt", "train_state.pt"]
    loss: Annotated[float, Field(allow_inf_nan=False)] | None = None
    architecture: Annotated[dict[Literal["ctx_window"], PositiveInt], Field(min_length=1, max_length=1)]
    final: StrictBool
    skipped_batches: NonnegativeInt = 0

    @model_validator(mode="after")
    def complete_position(self):
        """Require a real completed boundary and keep recovery/final file and ordinal meanings consistent."""
        if (self.step > self.total_steps or self.skipped_batches > self.total_steps - self.step
                or self.final and self.step + self.skipped_batches != self.total_steps
                or not self.final and self.skipped_batches != 0):
            raise ValueError("generation position differs from its completed schedule")
        if self.path != ("model.pt" if self.final else "train_state.pt") or (self.index == 0) != self.final:
            raise ValueError("generation final/recovery identity differs from its file and ordinal")
        return self

    def checkpoint(self) -> Checkpoint:
        """Translate a validated durable position to the existing live checkpoint event without another schema owner."""
        return Checkpoint(**{**self.model_dump(exclude={"skipped_batches"}), "model_id": str(self.model_id)})


class TrainingPublication(Contract):
    """Versioned parent-readable package metadata binding the generation to its original job and sparse request."""

    schema_version: Literal[1] = 1
    job_id: UUID
    training_kind: Literal["pretrain", "sft", "rl"] = "pretrain"
    config: TrainConfig | SftConfig | RlConfig
    result: dict[str, JsonValue]
    generation: TrainingGeneration

    @model_validator(mode="before")
    @classmethod
    def request_kind(cls, value):
        """Decode interpreted config using its explicit lifecycle kind rather than union-field guessing."""
        if isinstance(value, dict) and "config" in value:
            kind = value.get("training_kind", "pretrain")
            if kind not in {"pretrain", "sft", "rl"}:
                raise ValueError("unsupported training publication kind")
            config_type = {"pretrain": TrainConfig, "sft": SftConfig, "rl": RlConfig}[kind]
            if kind == "pretrain" and isinstance(value["config"], dict) and "base_model" in value["config"]:
                raise ValueError("SFT publication requires its explicit lifecycle kind")
            return {**value, "config": config_type.model_validate(value["config"])}
        return value

    @field_serializer("config")
    def stated_config(self, value: TrainConfig | SftConfig | RlConfig) -> dict:
        """Retain only explicitly stated training fields so selected-recipe metadata roundtrips unambiguously."""
        return value.model_dump(mode="json", exclude_unset=True)
