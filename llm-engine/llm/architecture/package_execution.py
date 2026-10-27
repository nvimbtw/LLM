"""Admitted-worker construction from verified self-contained packages; never used by parent readers."""

from dataclasses import dataclass
from collections.abc import Callable
import os
from pathlib import Path
import shutil

import torch

from llm.architecture import package
from llm.architecture import package_transfer
from llm.architecture.build import RecipeModule, build_model
from llm.architecture.builtins.catalog import builtin_definitions, compatibility_manifest
from llm.architecture.package_integrity import decode_json, encode_json
from llm.architecture.plan import ArchitectureDefinition
from llm.architecture.plan import ResolvedPlan
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader, SourceRevision, imports_public_builtins
from llm.architecture.spec import ResolvedDefinition, canonical_digest
from llm.model import checkpoint
from llm.runtime.worker import WorkerContext
from llm.runtime.protocol import Load, RuntimePins


@dataclass
class LoadedPackage:
    """Own custom import registrations for exactly the lifetime of one resident model."""

    _model: RecipeModule | None
    sources: SourceLoader
    state: checkpoint.TrainState
    checked: package.VerifiedPackage

    @property
    def model(self) -> RecipeModule:
        """Return the active resident model, refusing use after its owning package lifetime closes."""
        if self._model is None:
            raise ValueError("loaded model package is closed")
        return self._model

    def close(self) -> None:
        """Drop resident model/state tensors before releasing revision registrations; external borrowed references must already be retired."""
        self._model = None
        self.state.state_dict.clear()
        self.state.optimizer = None
        self.state.rng.clear()
        self.sources.close()

    def __enter__(self) -> "LoadedPackage":
        """Expose a scoped resident model/import lifetime to the owning worker backend."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release imports after normal execution or a failed worker operation."""
        self.close()


def load_dependencies(checked: package.VerifiedPackage, loader: SourceLoader) -> dict:
    """Resolve immutable packaged sources and installed builtins after compatibility verification."""
    from llm.architecture.parse import parse_recipe

    installed = builtin_definitions()
    manifest = compatibility_manifest()
    compatibility = canonical_digest(manifest)
    result = {}
    for entry in checked.manifest.definition.dependencies:
        source = checked.sources[entry.reference.revision_id]
        if entry.builtin_compatibility is not None and entry.builtin_compatibility != compatibility:
            raise ValueError("incompatible builtin implementation")
        if entry.reference.kind == "architecture":
            definition = ArchitectureDefinition(entry.reference, parse_recipe(source.decode("utf-8")))
        elif entry.path.endswith(".builtin.json"):
            descriptor = decode_json(source)
            definition = next((value for value in installed.values() if value.reference == entry.reference), None)
            expected = {"manifest": manifest, "declaration": definition.declaration.model_dump(mode="json")} if definition else None
            if entry.builtin_compatibility != compatibility or descriptor != expected or source != encode_json(expected):
                raise ValueError("incompatible builtin revision/declaration")
        else:
            if imports_public_builtins(source.decode("utf-8")) and entry.builtin_compatibility != compatibility:
                raise ValueError("custom builtin dependency is missing compatibility pin")
            definition = loader.load(SourceRevision(entry.reference, source.decode("utf-8")))
        result[entry.reference.revision_id] = definition
    return result


def validate_shared_tensors(model: RecipeModule, state_dict: dict) -> None:
    """Reject conflicting serialized aliases before load order can overwrite a shared Parameter."""
    aliases = {}
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if name not in state_dict:
            raise ValueError("packaged state is missing a Parameter")
        if id(parameter) in aliases:
            owner = aliases[id(parameter)]
            if not torch.equal(state_dict[owner], state_dict[name]):
                raise ValueError("packaged aliases contain conflicting tensors")
        else:
            aliases[id(parameter)] = name


def validate_saved_tensors(model: RecipeModule, state_dict: dict) -> None:
    """Reject missing, reshaped or silently cast saved tensors before assigning model weights/buffers."""
    expected = model.state_dict()
    if expected.keys() != state_dict.keys():
        raise ValueError("packaged tensor names differ from the saved model definition")
    for name, tensor in state_dict.items():
        target = expected[name]
        if not isinstance(tensor, torch.Tensor) or tensor.shape != target.shape or tensor.dtype != target.dtype:
            raise ValueError("packaged tensor shape/precision differs from the validated profile")
    validate_shared_tensors(model, state_dict)


def load_in_worker(directory: Path, context: WorkerContext, *, pins: RuntimePins,
                   restore_training_state: bool = True,
                   preflight: Callable[[ResolvedPlan, package.VerifiedPackage], None] | None = None) -> LoadedPackage:
    """Verify before source execution, rebuild aliases before tensors and retain revision-local imports.

    Only a real numerical WorkerContext may enter this API; ordinary checkpoint
    loading refuses modular construction. The runtime owner supplies and reaps
    this child and bounds import/construction time and host/device allocations.
    """
    if os.environ.get("LLM_RUNTIME_CHILD") != "1" or not isinstance(context, WorkerContext) or context.bootstrap.metadata_only:
        raise ValueError("modular model construction requires an admitted numerical worker")
    checked = package.verify(directory)
    return construct_package(checked, context, pins=pins, restore_training_state=restore_training_state, preflight=preflight)


def construct_package(checked: package.VerifiedPackage, context: WorkerContext, *, pins: RuntimePins,
                      restore_training_state: bool,
                      preflight: Callable[[ResolvedPlan, package.VerifiedPackage], None] | None = None) -> LoadedPackage:
    """Construct an already verified package inside its admitted numerical owner, rebuilding aliases before assigning safe tensor state.

    Production enters through the guarded load_in_worker boundary. Separating
    construction permits actual numerical tests after verifying parent refusal;
    it does not certify process admission or cleanup.
    """
    receipt = next((value for value in checked.receipts if value.receipt_id == pins.receipt_id), None)
    if receipt is None:
        raise ValueError("worker receipt is not retained by package")
    pins.require_current(RuntimePins(definition_id=checked.manifest.definition_id,
        bindings_digest=receipt.identity.parameter_digest,
        profile_digest=canonical_digest(receipt.identity.profile), receipt_id=receipt.receipt_id))
    profile = receipt.identity.profile
    if profile.device != context.device:
        raise ValueError("package profile differs from admitted worker device")
    precision = checked.manifest.parameter_precision or profile.precision
    dtype = {"fp32":torch.float32,"fp16":torch.float16,"bf16":torch.bfloat16}[precision]
    context.check_cancelled()
    sources = SourceLoader()
    try:
        dependencies = load_dependencies(checked, sources)
        definition = checked.manifest.definition
        plan = resolve_recipe(definition.recipe, definition.parameters, dependencies, definition.tokenizer)
        if preflight is not None:
            preflight(plan, checked)
        model = build_model(plan, device=context.device, dtype=dtype)
        context.check_cancelled()
        path = checked.directory / ((checked.manifest.training_state_path if restore_training_state else None) or checked.manifest.weights_path)
        state = checkpoint.load_state(path, device=context.device)
        validate_saved_tensors(model, state.state_dict)
        model.load_state_dict(state.state_dict, strict=True)
        model.eval()
        return LoadedPackage(model, sources, state, checked)
    except BaseException:
        sources.close()
        raise


def load_granted_package(command: Load, context: WorkerContext, *,
                         preflight: Callable[[ResolvedPlan, package.VerifiedPackage], None] | None = None) -> LoadedPackage:
    """Materialize verified private Load grants and require exact definition/pins before source execution.

    The worker owns the extracted directory until its session is removed. Failed
    loads remove partial materialization; browser paths never enter this boundary.
    """
    if os.environ.get("LLM_RUNTIME_CHILD") != "1" or not isinstance(context, WorkerContext) or context.bootstrap.metadata_only:
        raise ValueError("package grants require an admitted numerical worker")
    if command.package is None:
        raise ValueError("saved model loading requires a package grant")
    definition = ResolvedDefinition.model_validate_json(context.artifacts.read_bytes(command.definition))
    if canonical_digest(definition) != command.pins.definition_id:
        raise ValueError("load definition differs from pinned package")
    archive = context.artifacts.verify(command.package)
    directory = context.directory / f"package-{command.package.artifact_id}"
    if directory.exists() or directory.is_symlink():
        raise FileExistsError("package grant is already materialized")
    try:
        checked = package_transfer.materialize(archive, directory)
        if checked.manifest.definition != definition:
            raise ValueError("load definition differs from package bytes")
        return load_in_worker(directory, context, pins=command.pins, restore_training_state=False, preflight=preflight)
    except BaseException:
        if directory.exists():
            shutil.rmtree(directory)
        raise


def restore_optimizer(loaded: LoadedPackage, optimizer: torch.optim.Optimizer) -> None:
    """Restore moments only after exact saved parameter group/order and definition checks."""
    state = loaded.state
    if state.optimizer is None:
        raise ValueError("package has no optimizer history")
    if checkpoint.optimizer_layout(loaded.model, optimizer) != state.optimizer_parameter_groups:
        raise ValueError("optimizer group ownership/order differs from saved package")
    optimizer.load_state_dict(state.optimizer)
