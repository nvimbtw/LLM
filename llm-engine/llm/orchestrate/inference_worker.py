"""Supervised serving orchestration; model loading, numerical admission and request caches remain child-owned."""

from contextlib import closing

import torch
from tokenizers import Tokenizer

from llm.architecture import package
from llm.architecture.build import build_model
from llm.architecture.capture import CausalCapture, capture_complete
from llm.architecture.errors import ArchitectureError
from llm.architecture.package_execution import load_granted_package
from llm.architecture.package_execution import LoadedPackage
from llm.architecture.package_integrity import encode_json, package_file, read_bytes
from llm.architecture.validation import PRECISIONS
from llm.architecture.plan import ResolvedPlan
from llm.architecture.spec import RuntimeProfile
from llm.architecture.wire import AttentionMatrix
from llm.model.generation import stream_model
from llm.corpus import bpe
from llm.model import continuation, likelihood
from llm.model.inference_records import (
    AttentionInput, InspectionInput, InspectionRefusal, MEASUREMENT_REQUEST,
    GenerationInput, GenerationResult, MeasurementInput, MeasurementResult, MAX_MEASUREMENT_TOKENS,
)
from llm.orchestrate.inspection import InspectionRuntime
from llm.model.language import model_facts
from llm.runtime.protocol import Completed, Generate, Load, Measure, Progress, Tokens
from llm.runtime.protocol import RuntimePayload
from llm.runtime.worker import WorkerContext
from llm.train.memory import planning_target
from llm.train.model_memory import activation_bytes, owned_tensors, require_capture


class CooperativeCancel:
    """Adapt cooperative scoring boundaries to supervised request cancellation without acquiring device ownership."""

    def __init__(self, context: WorkerContext):
        """Retain only the worker request's bounded control-channel owner."""
        self.context = context

    def is_set(self) -> bool:
        """Raise the runtime cancellation signal when observed; otherwise allow one bounded numerical operation."""
        self.context.check_cancelled()
        return False


class InferenceBackend:
    """Own one verified model/import namespace and no cache state between complete requests."""

    def __init__(self, context: WorkerContext):
        """Initialize trusted numerical runtime settings; loading occurs only after a typed Load command."""
        torch.set_num_threads(1)
        self.loaded: LoadedPackage | None = None
        self.tokenizer: Tokenizer | None = None
        self.profile: RuntimeProfile | None = None
        self.inspection: InspectionRuntime | None = None

    def handle(self, payload: RuntimePayload, context: WorkerContext) -> Completed:
        """Dispatch only immutable package load, generation or measurement under exact resident runtime pins."""
        if isinstance(payload, Load):
            return self.load(payload, context)
        if self.loaded is None or not isinstance(payload, (Generate, Measure)):
            raise ValueError("serving requires a loaded immutable model package")
        payload.pins.require_current(context.pins)
        with torch.autocast(context.device, dtype=PRECISIONS[self.profile.precision], enabled=self.profile.precision != "fp32"):
            if isinstance(payload, Generate):
                return self.generate(payload, context)
            if isinstance(payload, Measure):
                return self.measure(payload, context)
        raise ValueError("serving worker accepts only load/generate/measure")

    def load(self, command: Load, context: WorkerContext) -> Completed:
        """Verify saved evidence and preflight memory on meta before allocating real weights; serving never restores optimizer/RNG state."""
        if self.loaded is not None or command.package is None:
            raise ValueError("one serving worker owns one immutable saved model")
        context.progress(Progress(type="progress", phase="load", completed=0, unit="phases"), force=True)
        def preflight(plan: ResolvedPlan, checked: package.VerifiedPackage) -> None:
            """Charge unique model tensors, conservative captured workspace and loading copies before constructing a real model."""
            context.check_cancelled()
            receipt = next((item for item in checked.receipts if item.receipt_id == command.pins.receipt_id), None)
            if receipt is None:
                raise ValueError("serving requires a retained valid profile")
            self.profile = receipt.identity.profile
            window = checked.manifest.definition.context_window
            if (self.profile.device != context.device or "eval" not in self.profile.modes
                    or not any(domain.batch_min == 1 and domain.sequence_min == 1 and domain.sequence_max >= window
                        and "full_prefix" in domain.signatures for domain in self.profile.domains)):
                raise ValueError("saved receipt does not cover full-prefix serving over the trained context")
            torch.backends.cuda.matmul.allow_tf32 = self.profile.allow_tf32
            dtype = PRECISIONS[checked.manifest.parameter_precision or self.profile.precision]
            with torch.random.fork_rng(devices=[]), torch.device("meta"):
                model = build_model(plan, device="meta", dtype=dtype).eval()
                tokens = torch.zeros((1, window), dtype=torch.int64)
                positions = torch.arange(window)[None]
                exported = capture_complete(CausalCapture(model), (tokens, positions, torch.ones_like(tokens, dtype=torch.bool))).exported
                require_capture(model, exported)
                tensors = sum(value.numel() * value.element_size() for value in owned_tensors(model).values())
                estimate = 2 * tensors + activation_bytes(exported) + window * model.facts.vocabulary_size * 8
                resources = context.bootstrap.policy.resources
                ceiling = resources.device_allocator_bytes if context.device == "cuda" else resources.tree_resident_bytes
                host_load = 2 * tensors + (768 << 20)
                if host_load > planning_target(resources.tree_resident_bytes) or estimate + ((768 << 20) if context.device == "cpu" else 0) > planning_target(ceiling):
                    raise ValueError("saved model inference workspace exceeds admitted memory")

        self.loaded = load_granted_package(command, context, preflight=preflight)
        self.loaded.state.state_dict.clear()
        self.tokenizer = Tokenizer.from_str(read_bytes(package_file(self.loaded.checked.directory, "tokenizer.json"), 32 << 20).decode("utf-8"))
        if self.tokenizer.get_vocab_size() != self.loaded.model.facts.vocabulary_size:
            raise ValueError("serving tokenizer differs from saved vocabulary")
        self.inspection = InspectionRuntime(self.loaded.model, context, pins=command.pins, profile=self.profile)
        return Completed(type="completed", outcome="ok")

    def generate(self, command: Generate, context: WorkerContext) -> Completed:
        """Run complete canonical generation with bounded token events and request-owned cache cleanup on every exit."""
        prepared = GenerationInput.model_validate_json(context.artifacts.read_bytes(command.prompt))
        cfg = prepared.config
        if (command.max_tokens != cfg.max_tokens or command.temperature != cfg.temperature
                or command.seed != (cfg.seed or 0)):
            raise ValueError("generation command differs from prepared sampling options")
        context.progress(Progress(type="progress", phase="generation", completed=0, unit="tokens"), force=True)

        def emit_token(token_id):
            """Emit one vocabulary-validated ID through bounded IPC; the parent owns only decoding/presentation."""
            context.emit(Tokens(type="tokens", token_ids=(token_id,)))

        with closing(stream_model(cfg, self.loaded.model, self.tokenizer, context.device,
                prompt_ids=list(prepared.prompt_ids), check_cancelled=context.check_cancelled, on_token=emit_token)) as stream:
            while True:
                try:
                    next(stream)
                except StopIteration as done:
                    result = GenerationResult.model_validate(done.value)
                    break
        artifact = context.artifacts.write_bytes(encode_json(result), "generation_result_v1")
        return Completed(type="completed", outcome="ok", result=artifact)

    def measure(self, command: Measure, context: WorkerContext) -> Completed:
        """Dispatch bounded typed observation operations inside the single loaded numerical owner and publish only validated data."""
        prepared = MEASUREMENT_REQUEST.validate_json(context.artifacts.read_bytes(command.measurement))
        context.check_cancelled()
        if isinstance(prepared, AttentionInput):
            result = self.attention_measurement(prepared)
        elif isinstance(prepared, InspectionInput):
            result = self.inspection.catalog(prepared.model_name)
        else:
            result = self.language_measurement(prepared, context)
        context.check_cancelled()
        artifact = context.artifacts.write_bytes(encode_json(result), "measurement_result_v1")
        return Completed(type="completed", outcome="ok", result=artifact)

    def attention_measurement(self, prepared: AttentionInput) -> AttentionMatrix | InspectionRefusal:
        """Return one proved matrix or a static classified public refusal; custom exception details never cross this ordinary-user boundary."""
        try:
            return self.inspection.matrix(prepared.model_name, self.tokenizer, prepared.selection)
        except ArchitectureError as error:
            code = {
                "inspection_unsupported": "unsupported", "inspection_point": "selection",
                "inspection_head": "selection", "inspection_input": "input", "inspection_budget": "budget",
            }.get(error.code, "validation")
            return InspectionRefusal(code=code)

    def language_measurement(self, prepared: MeasurementInput, context: WorkerContext) -> MeasurementResult:
        """Compute bounded generic token likelihood and unconstrained seeded samples; unsupported fixed-layer ablation remains absent."""
        context.progress(Progress(type="progress", phase="measurement", completed=0, unit="tokens"), force=True)
        ids = bpe.encode(self.tokenizer, prepared.text)
        if len(ids) > MAX_MEASUREMENT_TOKENS:
            raise ValueError("measurement text exceeds token-work budget")
        cancel = CooperativeCancel(context)
        scores = likelihood.token_logprobs(self.loaded.model, ids, context.device, cancel=cancel)
        cfg = prepared.config
        sampling = cfg.model_dump(exclude={"model_name", "suite"})
        samples = []
        for index, prompt in enumerate(prepared.prompts):
            context.check_cancelled()
            options = continuation.SamplingConfig(**{**sampling, "seed": cfg.seed + index})
            samples.append({"prompt": prompt, "text": continuation.continue_tokens(
                self.loaded.model, self.tokenizer, prompt, options, context.device, cancel=cancel)})
        facts = model_facts(self.loaded.model)
        return MeasurementResult(model={"id": cfg.model_name, "name": cfg.model_name,
            "context": facts.context_window, "vocab": facts.vocabulary_size,
            "parameters": sum(value.numel() for value in self.loaded.model.parameters())},
            likelihood={"total": scores.total, "count": scores.count,
                "unscored_bytes": len(bpe.decode(self.tokenizer, ids[:1]).encode("utf-8"))}, samples=tuple(samples))

    def finish_request(self) -> None:
        """Keep only resident weights/imports; generation's scoped iterator owns and drops every request cache."""
        return None

    def close(self) -> None:
        """Drop resident tensors before removing revision-local import registrations and worker teardown."""
        loaded, self.loaded = self.loaded, None
        inspection, self.inspection = self.inspection, None
        if inspection is not None:
            inspection.close()
        self.tokenizer = self.profile = None
        if loaded is not None:
            loaded.close()
