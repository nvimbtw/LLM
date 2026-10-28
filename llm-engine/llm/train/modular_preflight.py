"""Admitted-worker final recipe construction, exact numerical evidence and memory planning."""

from dataclasses import dataclass
from collections.abc import Callable
from uuid import UUID, uuid4

import torch

from llm.architecture.build import RecipeModule, build_model
from llm.architecture.builtins.catalog import builtin_definitions, compatibility_manifest
from llm.architecture.capture import CausalCapture, capture_complete
from llm.architecture.dependencies import recipe_dependencies
from llm.architecture.graphs import GraphArtifacts, architecture_scopes
from llm.architecture.numerical import equivalence, preserved_model
from llm.architecture.package_integrity import encode_json
from llm.architecture.parse import parse_recipe
from llm.architecture.plan import ResolvedPlan
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader, imports_public_builtins
from llm.architecture.spec import ResolvedDefinition, RuntimeProfile, SourceManifestEntry, ValidationReceipt, canonical_digest
from llm.architecture.validation import capture_causal, captured_profile, load_dependencies, make_inputs, runtime_profile, validation_identity
from llm.architecture.validation_records import ValidationOutput, ValidationSnapshot
from llm.architecture.wire import StageOutcome, ValidationReport
from llm.train.loop import autocast_dtype
from llm.train.model_memory import ModelMemoryPlan, estimate_model
from llm.train.recompute_capture import capture_replay_storage
from llm.train.recompute_plan import TrainingRecomputation, captured_recomputation
from llm.model.checkpoint import TrainState


@dataclass
class ReadyModel:
    """Own the live model/imports, exact validation evidence and its admitted memory plan through training."""

    _model: RecipeModule | None
    loader: SourceLoader
    definition: ResolvedDefinition
    sources: dict[UUID, bytes]
    validation: ValidationOutput
    memory: ModelMemoryPlan
    recomputation: TrainingRecomputation
    state: TrainState | None = None
    reference: RecipeModule | None = None

    @property
    def model(self) -> RecipeModule:
        """Return the live admitted model while this owner retains its revision imports."""
        if self._model is None:
            raise RuntimeError("preflight model lifetime is closed")
        return self._model

    def close(self) -> None:
        """Drop model references before releasing revision-local import registrations."""
        self._model = None
        self.state = None
        self.reference = None
        self.loader.close()


def definition_sources(snapshot: ValidationSnapshot) -> tuple[tuple[SourceManifestEntry, ...], dict[UUID, bytes]]:
    """Build the exact transitive package closure; builtin implementation descriptors never become Python source."""
    dependencies = {value.reference.revision_id: value for value in snapshot.dependencies}
    builtins = {value.reference.revision_id: value for value in builtin_definitions().values()}
    compatibility = compatibility_manifest()
    digest = canonical_digest(compatibility)
    pending = list(recipe_dependencies(parse_recipe(snapshot.recipe)).values())
    entries, sources = {}, {}
    while pending:
        reference = pending.pop()
        if reference.revision_id in entries:
            continue
        if reference in snapshot.builtins:
            builtin = builtins.get(reference.revision_id)
            if builtin is None or builtin.reference != reference:
                raise ValueError("frozen builtin implementation is unavailable")
            raw = encode_json({"manifest": compatibility, "declaration": builtin.declaration.model_dump(mode="json")})
            suffix, builtin_digest = "builtin.json", digest
        else:
            dependency = dependencies[reference.revision_id]
            if dependency.reference != reference:
                raise ValueError("frozen dependency differs from recipe pin")
            raw = dependency.source.encode("utf-8")
            suffix = "py" if reference.kind == "block" else "yaml"
            builtin_digest = digest if reference.kind == "block" and imports_public_builtins(dependency.source) else None
            if reference.kind == "architecture":
                pending.extend(recipe_dependencies(parse_recipe(dependency.source)).values())
        entries[reference.revision_id] = SourceManifestEntry(reference=reference,
            path=f"sources/{reference.revision_id}.{suffix}", bytes=len(raw), sdk_version=1,
            builtin_compatibility=builtin_digest)
        sources[reference.revision_id] = raw
    return tuple(entries[key] for key in sorted(entries, key=str)), sources


def training_memory(model: RecipeModule, exported: torch.export.ExportedProgram, args: tuple, *,
                    allocator_bytes: int, profile_digest: str, recompute: bool | int | None,
                    retained_policy: dict | None = None, compiled: bool = False,
                    reference_model: RecipeModule | None = None,
                    reference_export: torch.export.ExportedProgram | None = None,
                    ) -> tuple[ModelMemoryPlan, TrainingRecomputation]:
    """Admit exact captured storage with the least fitting replay, retaining an exact recovery policy when supplied.

    The caller preserves RNG/mode/state around real-model probes. Meta execution
    observes only storage shapes before allocating real weights. Compiled replay
    remains unsupported and refuses before optimization instead of changing policy.
    """
    memory = estimate_model(model, exported, allocator_bytes=allocator_bytes,
                            reference_model=reference_model, reference_export=reference_export)
    storage = capture_replay_storage(model, exported, args)
    memory, replay = captured_recomputation(model.recomputation_targets(), memory, storage,
        profile_digest=profile_digest, requested=recompute, retained_policy=retained_policy)
    if compiled and replay.selected.occurrences:
        raise ValueError("recipe replay requires compile_model=false until checkpoint compilation is validated")
    return memory, replay


def frozen_reference(plan: ResolvedPlan, definition: ResolvedDefinition, device: str,
                     restore: Callable[[RecipeModule, ResolvedDefinition], None] | None) -> RecipeModule | None:
    """Construct an optional frozen reference under caller admission without consuming training RNG draws."""
    if restore is None:
        return None
    devices = [torch.cuda.current_device()] if device == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        reference = build_model(plan, device=device, dtype=torch.float32)
        reference.eval().requires_grad_(False)
        if device != "meta":
            restore(reference, definition)
    return reference


def reference_capture(reference: RecipeModule | None, args: tuple, profile: RuntimeProfile,
                      progress: Callable[[str], None]) -> torch.export.ExportedProgram | None:
    """Validate frozen evaluation domains and bound the exact full-batch reference workspace without retaining probe graphs."""
    if reference is None:
        return None
    if args[0].device.type == "meta":
        return capture_complete(CausalCapture(reference), args).exported
    with preserved_model(reference), torch.no_grad():
        reference.eval()
        graphs = GraphArtifacts(canonical_digest(profile))
        capture_causal(reference, profile.model_copy(update={"modes": ("eval",)}), graphs, progress)
        return capture_complete(CausalCapture(reference), args).exported


def preflight(snapshot: ValidationSnapshot, *, allocator_bytes: int,
              progress: Callable[[str], None],
              restore: Callable[[RecipeModule, ResolvedDefinition], TrainState | None] | None = None,
              recompute: bool | int | None = None, recover_exact: bool = False,
              compiled: bool = False,
              initialize: Callable[[RecipeModule], None] | None = None,
              restore_reference: Callable[[RecipeModule, ResolvedDefinition], None] | None = None) -> ReadyModel:
    """Construct fp32 master weights only after meta admission, then validate exact execution before optimizer work.

    The caller owns an admitted numerical worker and its host/device budgets.
    Numerical/capture probes preserve the post-initialization RNG, gradients and
    modes; selected occurrence replay additionally proves full-batch output,
    gradient and RNG equivalence to ordinary strict capture. The SourceLoader
    remains alive with the returned model. Exact recovery retains its saved
    profile/storage selection even if the current allocator would choose less replay.
    """
    if (snapshot.recipe is None or snapshot.request.subject.kind != "revision"
            or snapshot.request.tokenizer is None or snapshot.request.tokenizer.kind != "final"
            or snapshot.request.profile is None):
        raise ValueError("training preflight requires a final accepted causal revision")
    profile = snapshot.request.profile
    runtime_profile(profile, profile.device)
    recipe = parse_recipe(snapshot.recipe)
    loader = SourceLoader()
    try:
        progress("construction")
        dependencies = load_dependencies(snapshot, loader)
        plan = resolve_recipe(recipe, snapshot.request.parameters, dependencies, snapshot.request.tokenizer)
        if plan.context_window != snapshot.request.context_window:
            raise ValueError("final preflight context differs from frozen recipe authority")
        entries, sources = definition_sources(snapshot)
        definition = ResolvedDefinition(schema_version=1, architecture=snapshot.request.subject.reference,
            recipe=recipe, dependencies=entries, parameters=dict(plan.parameters),
            tokenizer=snapshot.request.tokenizer, context_window=plan.context_window,
            aliases=recipe.aliases, sdk_version=1)
        meta = build_model(plan, device="meta", dtype=torch.float32)
        domain = profile.domains[0]
        args = make_inputs(domain.batch_max, plan.context_window, plan.vocabulary_size, "meta")
        progress("capture")
        capture = capture_complete(CausalCapture(meta), args)
        meta_reference = frozen_reference(plan, definition, "meta", restore_reference)
        reference_export = reference_capture(meta_reference, args, profile, progress)
        training_memory(meta, capture.exported, args, allocator_bytes=allocator_bytes,
                        profile_digest=canonical_digest(profile), recompute=recompute, compiled=compiled,
                        reference_model=meta_reference, reference_export=reference_export)
        del meta, capture, args, meta_reference, reference_export
        torch.manual_seed(profile.numerical_seed)
        progress("construction")
        model = build_model(plan, device=profile.device, dtype=torch.float32)
        state = restore(model, definition) if restore is not None else None
        if initialize is not None:
            initialize(model)
        reference = frozen_reference(plan, definition, profile.device, restore_reference)
        identity = validation_identity(snapshot, model)
        graphs = GraphArtifacts(canonical_digest(identity))
        dtype = autocast_dtype(profile.precision, profile.device)
        with preserved_model(model), torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            capture_causal(model, profile, graphs, progress)
            model.train()
            args = make_inputs(domain.batch_max, plan.context_window, plan.vocabulary_size, profile.device)
            capture = capture_complete(CausalCapture(model), args)
            reference_export = reference_capture(reference, args, profile, progress)
            scopes = architecture_scopes(model.plan, model)
            for mode in profile.modes:
                for scope, (nodes, edges) in scopes.items():
                    graphs.add(view="architecture", mode=mode, signature="full_prefix", scope=scope, nodes=nodes, edges=edges)
        identity = identity.model_copy(update={"profile": captured_profile(profile, graphs)})
        retained_policy = state.extra.get("recomputation_policy") if recover_exact and state is not None else None
        if recover_exact and retained_policy is None:
            raise ValueError("exact recovery requires a retained replay selection")
        with preserved_model(model), torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
            model.train()
            memory, recomputation = training_memory(model, capture.exported, args,
                allocator_bytes=allocator_bytes, profile_digest=canonical_digest(identity.profile),
                recompute=recompute, retained_policy=retained_policy, compiled=compiled,
                reference_model=reference, reference_export=reference_export)
        model.configure_recomputation(recomputation.selected.occurrences)
        if recomputation.selected.occurrences:
            progress("numerical")
            with preserved_model(model), torch.autocast(device_type="cuda", dtype=dtype, enabled=dtype is not None):
                model.train()
                equivalence(CausalCapture(model), capture.exported.module(), args, profile.precision,
                            tokens=args[0], check_rng=True)
        graphs.identity_hash = canonical_digest(identity)
        manifest_hash, blobs = graphs.finish()
        receipt = ValidationReceipt(schema_version=1, receipt_id=uuid4(), identity=identity,
            identity_hash=canonical_digest(identity), verdict="valid", graph_manifest_hash=manifest_hash)
        report = ValidationReport(schema_version=1, run_id=snapshot.run_id, verdict="valid", receipt=receipt,
            stages=tuple(StageOutcome(stage=value, outcome="passed") for value in
                         ("inspection", "resolution", "construction", "capture", "numerical")),
            diagnostics=(), unique_parameter_count=sum(value.numel() for value in model.parameters()))
        return ReadyModel(model, loader, definition, sources, ValidationOutput(report=report, graph_blobs=blobs),
                          memory, recomputation, state, reference)
    except BaseException:
        loader.close()
        raise
