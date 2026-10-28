"""Child-callable bounded inspection, strict capture and numerical validation orchestration."""

import math
import re
from itertools import product
from collections.abc import Callable
from uuid import uuid4

import torch

from llm.architecture.build import build_model, place_module
from llm.architecture.builtins.catalog import pinned_builtins, verified_builtin
from llm.architecture.capture import CausalCapture, NamedCapture, capture_complete
from llm.architecture.dependencies import dependency_digest, recipe_dependencies
from llm.architecture.errors import ArchitectureError
from llm.architecture.graphs import GraphArtifacts, architecture_scopes, export_guards
from llm.architecture.numerical import causal_probes, equivalence, output_tensors, preserved_model
from llm.architecture.parse import parse_recipe
from llm.architecture.plan import ArchitectureDefinition, BlockDefinition
from llm.architecture.resolve import bound_port, parameter_values, resolve_recipe
from llm.architecture.source import SourceLoader, SourceRevision
from llm.architecture.spec import (
    FinalTokenizer, PendingTokenizer, ReceiptIdentity, RuntimeProfile, ValidationReceipt, canonical_digest,
)
from llm.architecture.validation_records import ValidationOutput, ValidationSnapshot, snapshot_dependencies, subject_reference
from llm.architecture.wire import Diagnostic, SourceLocation, StageOutcome, ValidationReport

MAX_PREFLIGHT_TENSOR_BYTES = 64 << 20
PRECISIONS = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def load_dependencies(snapshot: ValidationSnapshot, loader: SourceLoader) -> dict:
    """Resolve only supplied exact dependency bytes/installed manifest pins, never current catalog names."""
    definitions = pinned_builtins(snapshot.builtins)
    for dependency in snapshot.dependencies:
        reference = dependency.reference
        definition = (loader.load(SourceRevision(reference, dependency.source)) if reference.kind == "block"
                      else ArchitectureDefinition(reference, parse_recipe(dependency.source)))
        definitions[reference.revision_id] = definition
    return definitions


def root_block(snapshot: ValidationSnapshot, loader: SourceLoader) -> BlockDefinition:
    """Resolve a verified installed builtin or revision-local custom block; builtin descriptors never enter SourceLoader."""
    if snapshot.builtin_root is not None:
        return verified_builtin(snapshot.builtin_root,snapshot.source)
    return loader.load(SourceRevision(subject_reference(snapshot),snapshot.source))


def runtime_profile(profile: RuntimeProfile, device: str) -> None:
    """Refuse runtime/precision/signature mismatches before allocation or authorizing any evidence."""
    if profile.torch_version != torch.__version__ or profile.device != device:
        raise ArchitectureError("runtime_profile", "requested runtime/device differs from the admitted worker")
    if profile.token_dtype != "int64" or set(profile.modes) != {"train", "eval"}:
        raise ArchitectureError("runtime_profile", "full validation requires canonical int64 and both train/eval modes")
    if any(domain.signatures != ("full_prefix",) or domain.past_min or domain.past_max for domain in profile.domains):
        raise ArchitectureError("signature_unsupported", "this model adapter supports full-prefix signatures only")
    if profile.allow_tf32 != (torch.backends.cuda.matmul.allow_tf32 if device == "cuda" else False):
        raise ArchitectureError("runtime_profile", "TF32 policy differs from the admitted runtime")
    if device == "cuda" and not torch.cuda.is_available():
        raise ArchitectureError("device_unavailable", "requested CUDA device is unavailable")


def make_inputs(batch: int, sequence: int, vocabulary: int, device: str) -> tuple:
    """Create deterministic bounded token/position/mask examples without uploaded tensors."""
    if batch * sequence * (vocabulary * 4 + 17) > MAX_PREFLIGHT_TENSOR_BYTES:
        raise ArchitectureError("tensor_byte_limit", "input/logit profile exceeds numerical probe byte budget")
    tokens = torch.arange(batch * sequence, device=device).reshape(batch, sequence) % vocabulary
    positions = torch.arange(sequence, device=device).expand(batch, -1)
    return tokens, positions, torch.ones_like(tokens, dtype=torch.bool)


def sequence_parts(minimum: int, maximum: int) -> tuple[tuple[int, int], ...]:
    """Keep singleton specialization explicit while covering every requested sequence length."""
    parts = ((1, 1),) if minimum == 1 else ()
    return parts + (((max(2, minimum), maximum),) if maximum >= 2 else ())


def dimension(name: str, minimum: int, maximum: int):
    """Return a bounded symbolic axis or static shape when its declared range is singleton."""
    return minimum if minimum == maximum else torch.export.Dim(name, min=minimum, max=maximum)


def probe_shapes(minimum: int, maximum: int) -> tuple[int, ...]:
    """Probe endpoints and one interior point without claiming exhaustive numerical proof."""
    return tuple(sorted({minimum, maximum, (minimum + maximum) // 2}))


def capture_causal(model, profile: RuntimeProfile, graphs: GraphArtifacts, progress: Callable) -> None:
    """Capture all mode/domain partitions symbolically and compare eager/exported endpoint arithmetic."""
    for domain_index, domain in enumerate(profile.domains):
        if domain.sequence_max > model.facts.context_window:
            raise ArchitectureError("context_binding", "validation domain exceeds authoritative trained context")
        partitions = product(sequence_parts(domain.batch_min, domain.batch_max), sequence_parts(domain.sequence_min, domain.sequence_max))
        for part_index, ((batch_min, batch_max), (sequence_min, sequence_max)) in enumerate(partitions):
            batch = min(batch_max, max(2, batch_min))
            sequence = min(sequence_max, max(3, sequence_min))
            args = make_inputs(batch, sequence, model.facts.vocabulary_size, profile.device)
            axes = {0: dimension("batch", batch_min, batch_max),
                    1: dimension("sequence", sequence_min, sequence_max)}
            for mode in profile.modes:
                model.train(mode == "train")
                wrapper = CausalCapture(model)
                progress("numerical")
                output_tensors(wrapper(*args))
                progress("capture")
                captured = capture_complete(wrapper, args, dynamic_shapes=(axes, axes, axes))
                scope = f"domain-{domain_index}/part-{part_index}"
                graphs.add(view="operations", mode=mode, signature="full_prefix", scope=scope,
                           nodes=captured.nodes, edges=captured.edges, guards=captured.guards,
                           domain={"batch_min": batch_min, "batch_max": batch_max,
                                   "sequence_min": sequence_min, "sequence_max": sequence_max}, strict=True)
                progress("numerical")
                for actual_batch in probe_shapes(batch_min, batch_max):
                    for actual_sequence in probe_shapes(sequence_min, sequence_max):
                        example = make_inputs(actual_batch, actual_sequence, model.facts.vocabulary_size, profile.device)
                        equivalence(wrapper, captured.exported.module(), example, profile.precision, tokens=example[0])
                if mode == "eval":
                    causal_probes(model, make_inputs(batch_max, sequence_max, model.facts.vocabulary_size, profile.device)[0], profile.precision)
    for mode in profile.modes:
        children = [{"id": entry["scope"], "kind": "module", "label": entry["scope"],
                     "occurrence_path": entry["scope"], "child_scope": entry["scope"]}
                    for entry in graphs.entries if entry["view"] == "operations" and entry["mode"] == mode]
        graphs.add(view="operations", mode=mode, signature="full_prefix", scope="root", nodes=children, edges=())


def constructed_model(snapshot: ValidationSnapshot, dependencies: dict, loader: SourceLoader, *, meta: bool):
    """Construct one disposable exact profile on meta or its admitted real device after resolution."""
    request = snapshot.request
    profile = request.profile
    device = "meta" if meta else profile.device
    dtype = PRECISIONS[profile.precision]
    if snapshot.recipe is not None:
        recipe = parse_recipe(snapshot.recipe)
        if recipe.interface == "causal_lm" and not isinstance(request.tokenizer, FinalTokenizer):
            raise ArchitectureError("pending_tokenizer", "full causal validation requires finalized tokenizer bindings")
        plan = resolve_recipe(recipe, request.parameters, dependencies, request.tokenizer)
        if request.context_window is not None and request.context_window != plan.context_window:
            raise ArchitectureError("context_binding", "preview cannot override recipe context authority")
        model = build_model(plan, device=device, dtype=dtype)
        model.__slate_resolved_inputs__ = plan.inputs
        model.__slate_resolved_outputs__ = plan.outputs
        return model
    definition = root_block(snapshot,loader)
    values = parameter_values(definition.declaration.parameters, request.parameters)
    with torch.device(device):
        model = definition.module_type(**values)
        place_module(model, device, dtype)
    model.__slate_resolved_inputs__ = {name: bound_port(port, values) for name, port in definition.declaration.inputs.items()}
    model.__slate_resolved_outputs__ = {name: bound_port(port, values) for name, port in definition.declaration.outputs.items()}
    model.__slate_constructor_values__ = values
    return model


def check_allocation(model) -> None:
    """Bound declared persistent tensor bytes before allocating probe weights on the real device."""
    size = sum(tensor.numel() * tensor.element_size() for tensor in (*model.parameters(), *model.buffers()))
    if size > MAX_PREFLIGHT_TENSOR_BYTES:
        raise ArchitectureError("model_byte_limit", "model profile exceeds validation allocation budget")


def capture_tensors(snapshot: ValidationSnapshot, model, declaration, graphs: GraphArtifacts, progress: Callable) -> None:
    """Validate exact named-tensor profiles for blocks and tensor recipes without implicit LM assumptions."""
    request, profile = snapshot.request, snapshot.request.profile
    if set(request.tensor_inputs) != set(declaration.inputs):
        raise ArchitectureError("tensor_inputs", "full tensor validation requires every declared synthetic tensor input")
    inputs = {}
    for name, value in request.tensor_inputs.items():
        dtype = PRECISIONS[profile.precision] if value.dtype == "floating" else getattr(torch, value.dtype)
        if torch.tensor([], dtype=dtype).element_size() * math.prod(value.shape) > MAX_PREFLIGHT_TENSOR_BYTES:
            raise ArchitectureError("tensor_byte_limit", "synthetic input exceeds validation byte budget")
        inputs[name] = torch.ones(value.shape, dtype=dtype, device=profile.device)
    for domain in profile.domains:
        if domain.batch_min != domain.batch_max or domain.sequence_min != domain.sequence_max:
            raise ArchitectureError("block_shape_domain", "tensor validation requires an exact synthetic input domain")
        for value in inputs.values():
            if value.dim() >= 2 and value.shape[:2] != (domain.batch_min, domain.sequence_min):
                raise ArchitectureError("block_shape_domain", "synthetic input shape differs from the declared exact domain")
    wrapper = NamedCapture(model)
    for mode in profile.modes:
        model.train(mode == "train")
        progress("numerical")
        output_tensors(wrapper(inputs))
        progress("capture")
        captured = capture_complete(wrapper, (inputs,))
        graphs.add(view="operations", mode=mode, signature="full_prefix", scope="root",
                   nodes=captured.nodes, edges=captured.edges, guards=captured.guards, strict=True)
        progress("numerical")
        equivalence(wrapper, captured.exported.module(), (inputs,), profile.precision)


def validation_identity(snapshot: ValidationSnapshot, model) -> ReceiptIdentity:
    """Bind exact closure, resolved values, inputs, tokenizer/context and declared runtime profile."""
    request = snapshot.request
    dependencies = snapshot_dependencies(snapshot)
    parameters = dict(model.plan.parameters) if snapshot.recipe is not None else model.__slate_constructor_values__
    context = model.plan.context_window if snapshot.recipe is not None else request.context_window
    return ReceiptIdentity(subject=request.subject, dependency_digest=dependency_digest(dependencies),
        parameter_digest=canonical_digest(parameters),
        input_profile_digest=canonical_digest({"tensor_inputs": {name: value.model_dump(mode="json") for name, value in request.tensor_inputs.items()},
                                               "domains": [domain.model_dump(mode="json") for domain in request.profile.domains]}),
        tokenizer=request.tokenizer, context_window=context, profile=request.profile)


def provisional_validation(snapshot: ValidationSnapshot, declaration) -> ValidationOutput:
    """Report checked source/parameter metadata without inventing pending vocabulary or claiming numerical execution."""
    request = snapshot.request
    recipe = parse_recipe(snapshot.recipe) if snapshot.recipe is not None else None
    parameters = parameter_values(recipe.parameters if recipe else declaration.parameters,request.parameters)
    context = recipe.context_window if recipe else request.context_window
    if isinstance(context,str):
        context = parameters[context[2:-1]]
    if request.context_window is not None and request.context_window != context:
        raise ArchitectureError("context_binding","preview cannot override recipe context authority")
    inputs = {"tensor_inputs":{name:value.model_dump(mode="json") for name,value in request.tensor_inputs.items()},
              "domains":[domain.model_dump(mode="json") for domain in request.profile.domains]}
    identity = ReceiptIdentity(subject=request.subject,dependency_digest=dependency_digest(snapshot_dependencies(snapshot)),
        parameter_digest=canonical_digest(parameters),input_profile_digest=canonical_digest(inputs),
        tokenizer=request.tokenizer,context_window=context,profile=request.profile)
    receipt = ValidationReceipt(schema_version=1,receipt_id=uuid4(),identity=identity,
        identity_hash=canonical_digest(identity),verdict="provisional")
    report = ValidationReport(schema_version=1,run_id=snapshot.run_id,verdict="provisional",declaration=declaration,receipt=receipt,
        stages=(StageOutcome(stage="inspection",outcome="passed"),
                *(StageOutcome(stage=stage,outcome="skipped") for stage in ("resolution","construction","capture","numerical"))),
        diagnostics=(Diagnostic(code="pending_tokenizer",phase="resolution",
            message="Final tokenizer binding is required before resolution, construction, capture or numerical authorization."),))
    return ValidationOutput(report=report)


def captured_profile(requested: RuntimeProfile, graphs: GraphArtifacts) -> RuntimeProfile:
    """Bind receipts to actual labeled strict exports and reject stale supplied guard claims.

    Empty requested guards ask capture to discover them. Each bounded record
    identifies a mode/signature/shape partition, including static specializations.
    """
    guards = export_guards(graphs.entries)
    if requested.export_guards and requested.export_guards != guards:
        raise ArchitectureError("capture_guard_mismatch", "requested export guards differ from actual captured partitions")
    return requested.model_copy(update={"export_guards": guards})


def diagnostic(error: Exception, phase: str, snapshot: ValidationSnapshot) -> Diagnostic:
    """Expose bounded errors with immutable source locations, retaining positive parser positions without private filenames or tracebacks."""
    reference = subject_reference(snapshot)
    location = SourceLocation(revision_id=reference.revision_id, file=f"blocks/{reference.revision_id}.py") if snapshot.source is not None else None
    cause = error
    while cause is not None:
        if isinstance(cause, SyntaxError) and location is not None:
            location = location.model_copy(update={
                "line": cause.lineno if type(cause.lineno) is int and cause.lineno > 0 else None,
                "column": cause.offset if type(cause.offset) is int and cause.offset > 0 else None,
            })
        traceback = cause.__traceback__
        while traceback is not None:
            match = re.search(r"blocks/([0-9a-f-]{36})\.py$", traceback.tb_frame.f_code.co_filename)
            if match:
                location = SourceLocation(revision_id=match.group(1), file=f"blocks/{match.group(1)}.py", line=traceback.tb_lineno)
            traceback = traceback.tb_next
        cause = cause.__cause__
    return Diagnostic(code=error.code if isinstance(error, ArchitectureError) else "validation_failed",
        message=str(error)[:2000] if isinstance(error, ArchitectureError) else f"{phase} failed: {type(error).__name__}",
        phase=phase, resource_id=reference.resource_id, revision_id=reference.revision_id, source=location)


def validate_snapshot(snapshot: ValidationSnapshot, *, device: str, progress: Callable[[str], None] = lambda phase: None) -> ValidationOutput:
    """Preserve host/device RNG across all inspection, fake construction and disposable numerical work."""
    devices = [torch.cuda.current_device()] if device == "cuda" and torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        return execute_validation(snapshot, device=device, progress=progress)


def execute_validation(snapshot: ValidationSnapshot, *, device: str, progress: Callable[[str], None]) -> ValidationOutput:
    """Inspect or fully validate exact captured bytes inside a supervised admitted worker.

    Constructors/probe weights are disposable. Output receipts become acceptance
    evidence only after parent verifies cleanup and atomically publishes artifacts.
    """
    stages, phase = [], "inspection"
    progress_error = False

    def report_progress(value: str) -> None:
        """Track the actual stage for bounded errors while forwarding cancellation/progress effects."""
        nonlocal phase, progress_error
        phase = value
        try:
            progress(value)
        except Exception:
            progress_error = True
            raise

    with SourceLoader() as loader:
        try:
            report_progress(phase)
            pinned_builtins(snapshot.builtins)
            declaration = root_block(snapshot,loader).declaration if snapshot.source is not None else None
            if snapshot.request.mode == "inspect":
                if declaration is None:
                    raise ArchitectureError("inspection_subject", "declaration inspection requires block source")
                report = ValidationReport(schema_version=1, run_id=snapshot.run_id, verdict="inspection_only",
                    declaration=declaration, stages=(StageOutcome(stage="inspection", outcome="passed"),), diagnostics=())
                return ValidationOutput(report=report)
            stages.append(StageOutcome(stage=phase, outcome="passed"))
            phase = "resolution"
            runtime_profile(snapshot.request.profile, device)
            dependencies = load_dependencies(snapshot, loader)
            if isinstance(snapshot.request.tokenizer,PendingTokenizer):
                return provisional_validation(snapshot,declaration)
            report_progress(phase)
            phase = "construction"
            meta = constructed_model(snapshot, dependencies, loader, meta=True)
            check_allocation(meta)
            del meta
            stages.extend((StageOutcome(stage="resolution", outcome="passed"),))
            report_progress(phase)
            devices = [torch.cuda.current_device()] if device == "cuda" else []
            with torch.random.fork_rng(devices=devices):
                torch.manual_seed(snapshot.request.profile.numerical_seed)
                model = constructed_model(snapshot, dependencies, loader, meta=False)
                identity = validation_identity(snapshot, model)
                graphs = GraphArtifacts(canonical_digest(identity))
                stages.append(StageOutcome(stage="construction", outcome="passed"))
                phase = "capture"
                with preserved_model(model):
                    if snapshot.recipe is not None and model.plan.interface == "causal_lm":
                        capture_causal(model, snapshot.request.profile, graphs, report_progress)
                    else:
                        capture_tensors(snapshot, model, model.plan if snapshot.recipe is not None else declaration,
                                        graphs, report_progress)
                    if snapshot.recipe is not None:
                        scopes = architecture_scopes(model.plan, model)
                        for mode in snapshot.request.profile.modes:
                            for scope, (nodes, edges) in scopes.items():
                                graphs.add(view="architecture", mode=mode, signature="full_prefix", scope=scope, nodes=nodes, edges=edges)
                identity = identity.model_copy(update={"profile": captured_profile(snapshot.request.profile, graphs)})
                graphs.identity_hash = canonical_digest(identity)
                manifest, blobs = graphs.finish()
                count = sum(parameter.numel() for parameter in model.parameters())
                del model
            receipt = ValidationReceipt(schema_version=1, receipt_id=uuid4(), identity=identity,
                identity_hash=canonical_digest(identity), verdict="valid", graph_manifest_hash=manifest)
            stages.extend((StageOutcome(stage="capture", outcome="passed"), StageOutcome(stage="numerical", outcome="passed")))
            report = ValidationReport(schema_version=1, run_id=snapshot.run_id, verdict="valid", declaration=declaration,
                receipt=receipt, stages=tuple(stages), diagnostics=(), unique_parameter_count=count)
            return ValidationOutput(report=report, graph_blobs=blobs)
        except Exception as error:
            if progress_error:
                raise
            stages.append(StageOutcome(stage=phase, outcome="failed"))
            report = ValidationReport(schema_version=1, run_id=snapshot.run_id, verdict="invalid",
                stages=tuple(stages), diagnostics=(diagnostic(error, phase, snapshot),))
            return ValidationOutput(report=report)
