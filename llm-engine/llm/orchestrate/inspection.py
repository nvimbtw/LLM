"""Worker-owned evidence for bounded attention measurements, separate from training or cached-decoding authorization."""

from dataclasses import dataclass
import base64

import torch
from torch import Tensor

from llm.architecture.build import RecipeModule, build_model
from llm.architecture.builtins.transformer import StandardTransformer
from llm.architecture.capture import capture_complete
from llm.architecture.errors import ArchitectureError
from llm.architecture.inspection import InspectionBinding, inspection_bindings, PreparedInspection
from llm.architecture.numerical import close, preserved_model
from llm.architecture.spec import RuntimeProfile, canonical_digest, MAX_ATTENTION_HEADS, MAX_ATTENTION_SEQUENCE
from llm.architecture.plan import PlanCall
from llm.architecture.wire import AttentionMatrix, AttentionSelection, InspectionCatalog, InspectionPointView
from llm.architecture.validation import dimension, probe_shapes, sequence_parts
from llm.architecture.validation import PRECISIONS
from llm.runtime.protocol import Progress, RuntimePins
from llm.runtime.worker import WorkerContext
from llm.train.memory import planning_target
from llm.train.model_memory import activation_bytes, owned_tensors

HOST_RUNTIME_BYTES = 768 << 20


@dataclass(frozen=True)
class AttentionEvidence:
    """Record exact resident definition/profile/view and validated single-row sequence bounds without retaining graphs or tensors."""

    definition_id: str
    profile_digest: str
    point_id: str
    sequence_max: int
    heads: int
    capture_hashes: tuple[str, ...]
    evidence_hash: str


@dataclass(frozen=True)
class AttentionMeasurement:
    """Own a single independent fp32 CPU matrix and its proved view identity until result serialization finishes."""

    weights: Tensor
    evidence: AttentionEvidence


def tensor_signature(model: RecipeModule) -> tuple:
    """Identify resident tensor objects/versions/layout so changed weights or placement cannot reuse an earlier measurement proof."""
    return tuple((id(value), value._version, value.dtype, value.device, tuple(value.shape)) for value in owned_tensors(model).values())


def rng_signature(model: RecipeModule) -> tuple:
    """Snapshot controlled CPU/device RNG for observation determinism checks without advancing generators."""
    devices = sorted({value.device.index for value in owned_tensors(model).values() if value.device.type == "cuda"})
    return (torch.get_rng_state().clone(), *(torch.cuda.get_rng_state(device).clone() for device in devices))


def require_same_rng(before: tuple, model: RecipeModule) -> None:
    """Reject evaluation observations or metadata that draw random values instead of preserving the caller's sampling state."""
    if any(not torch.equal(left, right) for left, right in zip(before, rng_signature(model), strict=True)):
        raise ArchitectureError("inspection_randomness", "attention inspection must preserve evaluation RNG state")


def inputs(sequence: int, vocabulary: int, device: str) -> tuple[Tensor, Tensor, Tensor]:
    """Construct one deterministic canonical visible-prefix probe inside an already admitted byte/sequence domain."""
    tokens = (torch.arange(sequence, device=device) % vocabulary)[None]
    return tokens, torch.arange(sequence, device=device)[None], torch.ones_like(tokens, dtype=torch.bool)


def require_attention(values: Tensor, *, sequence: int, heads: int, precision: str) -> None:
    """Require actual finite nonnegative causal probabilities with the declared axes and precision-appropriate row normalization."""
    if values.shape != (1, heads, sequence, sequence) or not values.is_floating_point():
        raise ArchitectureError("inspection_output", "attention axes differ from the proved head/sequence identity")
    if not bool(torch.isfinite(values).all()) or bool((values < 0).any()) or bool(values.triu(1).count_nonzero()):
        raise ArchitectureError("inspection_output", "attention probabilities must be finite, nonnegative and causal")
    close(values.float().sum(-1), torch.ones((1, heads, sequence), device=values.device), precision)


class InspectionRuntime:
    """Own bounded worker-lifetime observation proofs; request tensors/graphs never survive one measurement."""

    def __init__(self, model: RecipeModule, context: WorkerContext, *, pins: RuntimePins, profile: RuntimeProfile):
        """Borrow the immutable resident model and exact runtime identity, with no independent admission gate or service clients."""
        if canonical_digest(profile) != pins.profile_digest or profile.device != context.device:
            raise ArchitectureError("inspection_profile", "inspection runtime differs from the loaded model profile")
        self._model, self.context, self.pins, self.profile = model, context, pins, profile
        self.proven: dict[str, AttentionEvidence] = {}
        self.signature = tensor_signature(model)

    @property
    def model(self) -> RecipeModule:
        """Return the borrowed resident model while its inspection owner is active, refusing use after closure."""
        if self._model is None:
            raise ValueError("inspection runtime is closed")
        return self._model

    def bindings(self) -> tuple[InspectionBinding, ...]:
        """Read bounded declared metadata with state/mode/RNG preservation and retire proofs after resident tensors change."""
        model = self.model
        self.context.check_cancelled()
        signature = tensor_signature(model)
        if signature != self.signature:
            self.proven.clear()
            self.signature = signature
        self.snapshot_budget()
        with preserved_model(model), torch.no_grad():
            model.eval()
            rng = rng_signature(model)
            result = inspection_bindings(model)
            require_same_rng(rng, model)
        return result

    def snapshot_budget(self) -> None:
        """Refuse an impossible resident/snapshot allowance before cloning tensors for metadata or numerical preservation."""
        resident = self.snapshot_bytes(self.model)
        resources = self.context.bootstrap.policy.resources
        ceiling = resources.device_allocator_bytes if self.context.device == "cuda" else resources.tree_resident_bytes
        if (resident + HOST_RUNTIME_BYTES > planning_target(resources.tree_resident_bytes)
                or resident + (HOST_RUNTIME_BYTES if self.context.device == "cpu" else 0) > planning_target(ceiling)):
            raise ArchitectureError("inspection_budget", "attention model snapshots exceed admitted memory")

    def snapshot_bytes(self, model: RecipeModule) -> int:
        """Charge unique resident tensors, mutation-detection snapshots and conservative lower-precision autocast copies once per owner."""
        tensors = tuple(owned_tensors(model).values())
        resident = 2 * sum(value.numel() * value.element_size() for value in tensors)
        if self.profile.precision != "fp32":
            dtype = PRECISIONS[self.profile.precision]
            resident += sum(value.numel() * (torch.finfo(dtype).bits // 8) for value in tensors
                if value.is_floating_point() and value.dtype != dtype)
        return resident

    def meta_budget(self, binding: InspectionBinding, sequence: int) -> int:
        """Strictly capture the selected view on meta and charge resident/snapshot/workspace bytes before real observation allocation."""
        self.context.check_cancelled()
        self.snapshot_budget()
        dtype = next((value.dtype for value in owned_tensors(self.model).values() if value.is_floating_point()), torch.float32)
        with torch.random.fork_rng(devices=[]), torch.device("meta"):
            meta = build_model(self.model.plan, device="meta", dtype=dtype).eval()
            target = next((value for value in inspection_bindings(meta) if value.point == binding.point), None)
            if target is None:
                raise ArchitectureError("inspection_metadata", "meta construction changed declared attention identity")
            captured = capture_complete(PreparedInspection(meta, target), inputs(sequence, meta.facts.vocabulary_size, "meta"))
            output = list(captured.exported.graph.nodes)[-1].args[0][0].meta.get("val")
            if not isinstance(output, Tensor) or output.ndim != 4:
                raise ArchitectureError("inspection_output", "captured observation has no bounded attention tensor")
            shape = tuple(int(value) for value in output.shape)
            heads = shape[1]
            if not 1 <= heads <= MAX_ATTENTION_HEADS or shape != (1, heads, sequence, sequence) or binding.point.heads not in (None, heads):
                raise ArchitectureError("inspection_output", "captured attention axes differ from declared metadata")
            resident = self.snapshot_bytes(meta)
            estimate = resident + activation_bytes(captured.exported) + 2 * heads * sequence * sequence * 4
            resources = self.context.bootstrap.policy.resources
            if resident + HOST_RUNTIME_BYTES > planning_target(resources.tree_resident_bytes):
                raise ArchitectureError("inspection_budget", "attention model snapshots exceed admitted host memory")
            ceiling = resources.device_allocator_bytes if self.context.device == "cuda" else resources.tree_resident_bytes
            if estimate + (HOST_RUNTIME_BYTES if self.context.device == "cpu" else 0) > planning_target(ceiling):
                raise ArchitectureError("inspection_budget", "captured attention workspace exceeds admitted memory; use fewer tokens")
            return heads

    def validate(self, binding: InspectionBinding, sequence: int) -> AttentionEvidence:
        """Prove complete fixed-view capture/eager equivalence and deterministic causal probabilities over the requested bounded sequence domain."""
        previous = self.proven.get(binding.point.id)
        if previous is not None and previous.sequence_max >= sequence:
            return previous
        heads = self.meta_budget(binding, sequence)
        hashes = []
        with preserved_model(self.model), torch.no_grad():
            self.model.eval()
            inspector = PreparedInspection(self.model, binding)
            for minimum, maximum in sequence_parts(1, sequence):
                self.context.check_cancelled()
                example = inputs(min(maximum, max(3, minimum)), self.model.facts.vocabulary_size, self.context.device)
                axis = {1: dimension("sequence", minimum, maximum)}
                captured = capture_complete(inspector, example, dynamic_shapes=(axis, axis, axis))
                hashes.append(canonical_digest({"minimum": minimum, "maximum": maximum,
                    "nodes": captured.nodes, "edges": captured.edges, "guards": captured.guards}))
                exported = captured.exported.module()
                for size in probe_shapes(minimum, maximum):
                    self.context.check_cancelled()
                    arguments = inputs(size, self.model.facts.vocabulary_size, self.context.device)
                    rng = rng_signature(self.model)
                    actual, normal = inspector(*arguments)
                    retained, retained_normal = exported(*arguments)
                    require_same_rng(rng, self.model)
                    require_attention(actual, sequence=size, heads=heads, precision=self.profile.precision)
                    close(actual, retained, self.profile.precision)
                    close(normal, retained_normal, self.profile.precision)
                    close(normal, self.model(arguments[0], positions=arguments[1], key_mask=arguments[2]), self.profile.precision)
                    close(actual, inspector(*arguments)[0], self.profile.precision)
                    require_same_rng(rng, self.model)
        facts = {"definition_id": self.pins.definition_id, "profile_digest": self.pins.profile_digest,
            "point_id": binding.point.id, "sequence_max": sequence, "batch": 1, "heads": heads,
            "mode": "eval", "signature": "attention", "capture_hashes": tuple(hashes)}
        evidence = AttentionEvidence(self.pins.definition_id, self.pins.profile_digest, binding.point.id,
            sequence, heads, tuple(hashes), canonical_digest(facts))
        self.proven[binding.point.id] = evidence
        return evidence

    def attention(self, tokens: Tensor, *, point_id: str, head: int) -> AttentionMeasurement:
        """Validate/measure one requested view, returning only an independent fp32 CPU head matrix and scoped evidence."""
        facts = self.model.facts
        self.context.check_cancelled()
        if (tokens.ndim != 2 or tokens.shape[0] != 1 or tokens.dtype != torch.int64
                or tokens.device.type != self.context.device
                or not 1 <= tokens.shape[1] <= min(MAX_ATTENTION_SEQUENCE, facts.context_window)
                or bool((tokens < 0).any()) or bool((tokens >= facts.vocabulary_size).any())
                or type(head) is not int or not 0 <= head < MAX_ATTENTION_HEADS):
            raise ArchitectureError("inspection_input", "attention requires one bounded int64 prefix and a valid head")
        binding = next((value for value in self.bindings() if value.point.id == point_id), None)
        if binding is None:
            raise ArchitectureError("inspection_unsupported", "model has no declared matching attention inspection point")
        if binding.point.heads is not None and head >= binding.point.heads:
            raise ArchitectureError("inspection_head", "head is outside this attention point's declared range")
        self.context.progress(Progress(type="progress", phase="measurement", completed=0, unit="phases"), force=True)
        evidence = self.validate(binding, tokens.shape[1])
        if head >= evidence.heads:
            raise ArchitectureError("inspection_head", "head is outside the validated attention range")
        with preserved_model(self.model), torch.no_grad():
            self.model.eval()
            rng = rng_signature(self.model)
            arguments = (tokens, torch.arange(tokens.shape[1], device=tokens.device)[None], torch.ones_like(tokens, dtype=torch.bool))
            values, _ = PreparedInspection(self.model, binding)(*arguments)
            require_same_rng(rng, self.model)
            require_attention(values, sequence=tokens.shape[1], heads=evidence.heads, precision=self.profile.precision)
            result = values[0, head].to(device="cpu", dtype=torch.float32, copy=True).contiguous()
        return AttentionMeasurement(result, evidence)

    def legacy_dimensions(self) -> tuple[int, int] | None:
        """Retain numeric layer/head selection only for a direct standard composite with the complete forwarded LM interface."""
        operations = self.model.plan.operations
        if len(operations) != 1 or not isinstance(operations[0], PlanCall):
            return None
        call = operations[0]
        module = self.model.instances[call.instance]
        if (type(module) is not StandardTransformer or self.model.plan.result != call.outputs
                or set(call.inputs) != set(self.model.plan.inputs)
                or any(port != value for port, value in call.inputs.items())):
            return None
        return module.model.layers, module.model.heads

    def catalog(self, model_name: str) -> InspectionCatalog:
        """Return declared candidates and only existing worker-lifetime proofs, without pretending metadata discovery ran numerical validation."""
        bindings = self.bindings()
        legacy = self.legacy_dimensions()
        points = []
        for binding in bindings:
            point = binding.point
            evidence = self.proven.get(point.id)
            points.append(InspectionPointView(id=point.id, title=point.title,
                instance_path=point.occurrence.instance_path, call_path=point.occurrence.call_path,
                heads=evidence.heads if evidence else point.heads,
                validated_sequence_max=evidence.sequence_max if evidence else 0,
                evidence_hash=evidence.evidence_hash if evidence else None))
        return InspectionCatalog(model_name=model_name, definition_id=self.pins.definition_id,
            profile_digest=self.pins.profile_digest, runtime_id=self.context.bootstrap.worker_id,
            context_window=self.model.facts.context_window,
            max_tokens=min(MAX_ATTENTION_SEQUENCE, self.model.facts.context_window),
            legacy_layers=legacy[0] if legacy else None, legacy_heads=legacy[1] if legacy else None,
            points=tuple(points), unavailable_reason=None if points else "This model has no declared supported attention inspection method")

    def selected_binding(self, selection: AttentionSelection) -> tuple[InspectionBinding, int | None]:
        """Resolve an exact model-local point or the deliberate direct-standard legacy adapter before measuring any tensor."""
        bindings = self.bindings()
        if selection.point_id is not None:
            binding = next((value for value in bindings if value.point.id == selection.point_id), None)
            if binding is None:
                raise ArchitectureError("inspection_point", "selected attention point is unavailable for this model")
            return binding, None
        legacy = self.legacy_dimensions()
        layer = selection.layer if selection.layer is not None else 0
        if legacy is None:
            raise ArchitectureError("inspection_selection", "this model requires a declared attention point instead of a legacy layer")
        if layer >= legacy[0]:
            raise ArchitectureError("inspection_layer", "layer is outside this standard model's actual range")
        return bindings[layer], layer

    def matrix(self, model_name: str, tokenizer, selection: AttentionSelection) -> AttentionMatrix:
        """Measure the requested real view over canonical tokenizer IDs and serialize one bounded independent fp32 CPU head matrix."""
        from llm.corpus import bpe

        binding, layer = self.selected_binding(selection)
        ids = bpe.encode(tokenizer, selection.text)
        limit = min(MAX_ATTENTION_SEQUENCE, self.model.facts.context_window)
        if not ids or len(ids) > limit:
            raise ArchitectureError("inspection_input", f"attention text requires 1 to {limit} tokens")
        measured = self.attention(torch.tensor([ids], dtype=torch.int64, device=self.context.device),
            point_id=binding.point.id, head=selection.head)
        legacy = self.legacy_dimensions()
        evidence = measured.evidence
        return AttentionMatrix(model_name=model_name, layer=layer, head=selection.head,
            layers=legacy[0] if layer is not None else None, heads=evidence.heads, max_tokens=limit,
            tokens=tuple(bpe.token_text(tokenizer, token) for token in ids), shape=tuple(measured.weights.shape),
            weights=base64.b64encode(measured.weights.numpy().astype("<f4", copy=False).tobytes()).decode("ascii"),
            point_id=binding.point.id, definition_id=self.pins.definition_id, profile_digest=self.pins.profile_digest,
            runtime_id=self.context.bootstrap.worker_id, validated_sequence_max=evidence.sequence_max, evidence_hash=evidence.evidence_hash)

    def close(self) -> None:
        """Drop the borrowed model/context and all proof metadata before the resident package/import owner closes."""
        self.proven.clear()
        self._model = self.context = None
