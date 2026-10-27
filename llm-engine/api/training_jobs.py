"""Parent-only recipe selection and retained training pins; source construction remains worker-owned."""

from uuid import UUID
from dataclasses import dataclass

from sqlalchemy.orm import Session
import torch

from api.validation_jobs import read_tokenizer_binding
from llm.architecture.errors import ArchitectureError
from llm.architecture.resolve import parameter_values, resolve_value
from llm.architecture.parse import parse_recipe
from llm.architecture.spec import FinalTokenizer, PendingTokenizer, Recipe, RevisionRef, RevisionSubject, Scalar, TokenizerBinding, ValidationReceipt, canonical_digest
from llm.architecture.builtins.catalog import builtin_definitions, compatibility_manifest, pinned_builtins
from llm.architecture.source import imports_public_builtins
from llm.architecture.validation_records import PinnedSource, ValidationSnapshot
from llm.architecture.tokenizer_binding import reserved_names
from llm.architecture.wire import ArchitectureSelection, EnqueueEvidence, TrainingArchitecturePreview, TrainingArchitectureRequest, ValidationRequest
from llm.config import RlConfig, TrainConfig
from llm.train.model_preflight import rl_profile, training_profile
from store import architecture_queries as catalog
from store.schema import ArchitectureRevision, Model, RunArchitecturePin, ValidationEvidence


@dataclass(frozen=True)
class SelectedConfiguration:
    """Keep an accepted recipe's database row, parsed declarations and resolved public bindings in one caller-owned read transaction."""

    revision: ArchitectureRevision
    recipe: Recipe
    parameters: dict[str, Scalar]
    context_window: int


def selected_configuration(db: Session, selection: ArchitectureSelection) -> SelectedConfiguration:
    """Read one immutable accepted causal recipe and resolve its actual public parameters/context without constructing a model."""
    revision = db.get(ArchitectureRevision, selection.revision_id)
    if revision is None:
        raise catalog.CatalogError("not_found", "accepted architecture revision not found")
    recipe = Recipe.model_validate(revision.recipe)
    if recipe.interface != "causal_lm":
        raise catalog.CatalogError("invalid_selection", "training requires a causal language-model recipe")
    try:
        parameters = parameter_values(recipe.parameters, selection.parameters)
        context = resolve_value(recipe.context_window, parameters, None)
    except ArchitectureError as error:
        raise catalog.CatalogError(error.code, str(error)) from error
    if type(context) is not int or context < 1:
        raise catalog.CatalogError("context_binding", "recipe context must resolve to a positive integer")
    return SelectedConfiguration(revision, recipe, parameters, context)


def training_preview(db: Session, request: TrainingArchitectureRequest) -> TrainingArchitecturePreview:
    """Resolve prospective selected bindings, actual runtime profile and optional final tokenizer facts; issue no queue identity, authorization receipt or numerical work."""
    selected = selected_configuration(db, request.selection)
    resource = catalog.get_resource(db, "architecture", selected.revision.architecture_id)
    if resource.archived:
        raise catalog.CatalogError("archived", "archived architectures cannot be selected for new training")
    closure = catalog.dependency_closure(db, selected.revision.id)
    recipes = (selected.recipe, *(Recipe.model_validate(item.metadata) for item in closure if item.reference.kind == "architecture"))
    names = reserved_names(recipes)
    try:
        profile = training_profile(request, selected.context_window, "cuda" if torch.cuda.is_available() else "cpu")
    except ValueError as error:
        raise catalog.CatalogError("runtime_profile", str(error)) from error
    tokenizer = read_tokenizer_binding(request.tokenizer_id, names) if request.tokenizer_id is not None else None
    return TrainingArchitecturePreview(architecture=catalog.revision_reference(selected.revision), parameters=selected.parameters,
        parameter_digest=canonical_digest(selected.parameters), dependency_digest=selected.revision.dependency_digest,
        context_window=selected.context_window, profile=profile, tokenizer=tokenizer, reserved_token_names=names)


def prepare_selection(db: Session, config: dict, job_id: UUID) -> catalog.PreparedTrainingPin:
    """Resolve accepted structural values and freeze data-only source/evidence before inserting a training job.

    The job owns pending tokenization. Every selected configuration requires
    final in-job preflight under the actual worker admission, even when retaining
    an explicitly supplied matching prior receipt. No model/source is imported.
    """
    cfg = TrainConfig.model_validate(config)
    if cfg.architecture is None:
        raise catalog.CatalogError("invalid_selection", "training requires an explicit accepted recipe selection")
    selected = selected_configuration(db, cfg.architecture)
    revision, recipe, parameters, context = selected.revision, selected.recipe, selected.parameters, selected.context_window
    try:
        profile = training_profile(cfg, context, "cuda" if torch.cuda.is_available() else "cpu")
    except ValueError as error:
        raise catalog.CatalogError("runtime_profile", str(error)) from error
    closure = catalog.dependency_closure(db, revision.id)
    recipes = (recipe, *(Recipe.model_validate(item.metadata) for item in closure if item.reference.kind == "architecture"))
    tokenizer = _tokenizer(cfg, job_id, recipes)
    evidence = EnqueueEvidence(architecture=catalog.revision_reference(revision),
        dependency_digest=revision.dependency_digest, parameter_digest=canonical_digest(parameters),
        acceptance_receipt_id=revision.acceptance_receipt_id, tokenizer=tokenizer,
        exact_receipt_id=_selected_receipt(db, cfg.architecture.validation_run_id), final_preflight_required=True)
    prepared = catalog.prepare_training_pin(db, evidence, parameters)
    if evidence.exact_receipt_id is not None:
        retained = ValidationReceipt.model_validate(db.get(ValidationEvidence, evidence.exact_receipt_id).receipt)
        expected_inputs = canonical_digest({"tensor_inputs": {}, "domains": [value.model_dump(mode="json") for value in profile.domains]})
        if (retained.identity.context_window != context or retained.identity.input_profile_digest != expected_inputs
                or retained.identity.profile.model_copy(update={"export_guards": ()}) != profile):
            raise catalog.CatalogError("stale_evidence", "selected receipt differs from the effective training input/runtime profile")
    return prepared


def _tokenizer(cfg: TrainConfig, job_id: UUID, recipes: tuple[Recipe, ...]) -> TokenizerBinding:
    """Read actual owned final tokenizer facts or pin this training job's future tokenization identity."""
    if cfg.tokenizer_id is None:
        return PendingTokenizer(kind="pending", build_request_id=job_id)
    try:
        tokenizer_id = UUID(cfg.tokenizer_id)
    except ValueError as error:
        raise catalog.CatalogError("tokenizer_binding", "recipe training requires an owned tokenizer UUID") from error
    return read_tokenizer_binding(tokenizer_id, reserved_names(recipes))


def _selected_receipt(db: Session, run_id: UUID | None) -> UUID | None:
    """Find retained prior evidence independently of prunable validation history; pin validation checks its identity."""
    if run_id is None:
        return None
    evidence = catalog.evidence_for_job(db, run_id)
    if evidence is None:
        raise catalog.CatalogError("missing_evidence", "selected validation has no retained valid receipt")
    return evidence.receipt_id


def inherited_pin(db: Session, job_id: UUID) -> catalog.PreparedTrainingPin | None:
    """Read an existing immutable job selection for retry/recovery without mutable catalog eligibility checks."""
    pin = db.get(RunArchitecturePin, job_id)
    if pin is None:
        return None
    return catalog.PreparedTrainingPin(EnqueueEvidence.model_validate(pin.evidence),
                                      dict(pin.parameters), dict(pin.snapshot))


def saved_definition(db: Session, model_id: str | None) -> str | None:
    """Read a saved input's semantic identity without importing source, loading tensors or requiring old job history."""
    if model_id is None:
        return None
    try:
        identity = UUID(model_id)
    except ValueError:
        return None
    model = db.get(Model, identity)
    return model.definition_id if model is not None else None


def frozen_recipe(pin: catalog.PreparedTrainingPin) -> tuple[Recipe, int, tuple[str, ...]]:
    """Read authoritative context/reserved names from immutable queued data, never current catalog defaults."""
    recipe = Recipe.model_validate(pin.snapshot["recipe"])
    if parse_recipe(pin.snapshot["yaml"]) != recipe:
        raise ValueError("queued recipe metadata differs from its exact source")
    context = resolve_value(recipe.context_window, pin.parameters, None)
    if type(context) is not int or context < 1:
        raise ValueError("queued recipe has invalid context authority")
    recipes = (recipe, *(Recipe.model_validate(item["metadata"]) for item in pin.snapshot["dependencies"]
                         if item["reference"]["kind"] == "architecture"))
    return recipe, context, reserved_names(recipes)


def frozen_validation(pin: catalog.PreparedTrainingPin, cfg: TrainConfig | RlConfig, binding: FinalTokenizer,
                      job_id: UUID, device: str, *, prepared_tokenizer: FinalTokenizer | None = None) -> ValidationSnapshot:
    """Create a final snapshot from frozen source, checking installed builtin compatibility before import.

    SFT may supply a parent-verified prepared tokenizer after durable authority
    publication, including a waiting chain's pending inherited pin. Registered
    saved-model bytes supply that final authority; the enqueue pin is unchanged.
    """
    recipe, context, _names = frozen_recipe(pin)
    if prepared_tokenizer is not None and prepared_tokenizer != binding:
        raise ValueError("prepared tokenizer differs from parent-authorized final bindings")
    expected = prepared_tokenizer or pin.evidence.tokenizer
    if expected.kind == "final" and expected != binding:
        raise ValueError("final tokenizer differs from the queued immutable binding")
    compatibility = canonical_digest(compatibility_manifest())
    builtins, sources = {}, []
    for item in pin.snapshot["dependencies"]:
        reference = RevisionRef.model_validate(item["reference"])
        if item.get("origin") == "builtin":
            pinned_builtins((reference,))
            builtins[reference.revision_id] = reference
        else:
            if reference.kind == "block" and imports_public_builtins(item["source"]):
                if (item.get("compatibility") or {}).get("builtin_compatibility") != compatibility:
                    raise ValueError("queued custom builtin implementation is unavailable")
                builtins.update({value.reference.revision_id: value.reference for value in builtin_definitions().values()})
            sources.append(PinnedSource(reference=reference, source=item["source"]))
    reference = RevisionRef.model_validate(pin.snapshot["architecture"])
    request = ValidationRequest(schema_version=1, request_id=job_id, mode="full",
        subject=RevisionSubject(kind="revision", reference=reference), parameters=pin.parameters,
        tokenizer=binding, context_window=context,
        profile=rl_profile(cfg, context, device) if isinstance(cfg, RlConfig) else training_profile(cfg, context, device))
    return ValidationSnapshot(run_id=job_id, request=request, recipe=pin.snapshot["yaml"],
                              dependencies=tuple(sources), builtins=tuple(builtins.values()))
