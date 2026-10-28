"""Parent-owned reusable model worker, bounded streaming and interactive cancellation."""

from concurrent.futures import Future
from contextlib import contextmanager
from pathlib import Path
from queue import Empty, Full, Queue
import threading
from uuid import UUID

from tokenizers import Tokenizer
from pydantic import TypeAdapter

import store
from api.training_admission import DISK_RESERVE_BYTES, capacity
from api.model_metadata import read_model_meta, resolve_model_metadata
from llm import storage
from llm.architecture import package, package_transfer
from llm.architecture.package_integrity import PackageBudget, encode_json, read_bytes
from llm.architecture.spec import canonical_digest
from llm.architecture.spec import MAX_ATTENTION_SEQUENCE
from llm.architecture.wire import AttentionMatrix, AttentionSelection, InspectionCatalog
from llm.model.generation import TokenRenderer, prepare_prompt, history, RESERVE_FOR_REPLY
from llm.chat.template import ThinkBudget, context_floor
from llm.config import ChatConfig, EvalConfig
from llm.corpus import bpe
from llm.model.device import Busy
from llm.model.inference_records import (
    AttentionInput, InspectionInput, InspectionRefusal, MeasurementRequest,
    GenerationInput, GenerationResult, MeasurementInput, MeasurementResult, generation_bounds,
)
from llm.orchestrate import chat as legacy_chat
from llm.orchestrate import attention as legacy_attention
from llm.model import device
from llm.runtime.protocol import Completed, Generate, Load, Measure, RuntimePins, RuntimePolicy, RuntimeResources, Tokens
from llm.runtime.supervisor import CleanupUncertain, RequestCancelled, WorkerManager, WorkerSession, WorkerStopped
from llm.train import memory
from store import queries as q

SERVING_BACKEND = "llm.orchestrate.inference_worker:InferenceBackend"
REQUEST_BYTES = 8 << 20


def modular(model_name: str) -> bool:
    """Identify immutable modular artifacts without importing code or opening any tensor file."""
    return (storage.model_dir(model_name) / "package.json").exists()


def serving_policy(checked: package.VerifiedPackage, root: Path, device: str) -> tuple[RuntimePolicy, PackageBudget]:
    """Reserve private archive/grant/extraction copies and bounded request/results before starting a resident numerical worker."""
    available = capacity(root)
    payload = sum(value.size_bytes for value in checked.integrity.files.values()) + (4 << 20)
    budget = PackageBudget(total_bytes=payload, file_bytes=min(payload, 16 << 30))
    archive = package_transfer.archive_limit(budget)
    if 3 * archive + 4 * REQUEST_BYTES + DISK_RESERVE_BYTES > available.disk_bytes:
        raise ValueError("insufficient disk admission for serving package copies")
    resources = RuntimeResources(process_address_bytes=1 << 40 if device == "cuda" else 64 << 30,
        tree_resident_bytes=available.host_bytes, tree_memory_enforcement="preflight_and_monitor",
        device=device, device_allocator_bytes=memory.budget_bytes(device) if device == "cuda" else 0,
        owned_processes=2, artifact_bytes=max(archive, REQUEST_BYTES), total_artifact_bytes=archive + 4 * REQUEST_BYTES)
    return RuntimePolicy(resources=resources), budget


class ServingModels:
    """Own one resident package/worker and serialize complete interactive requests under the shared parent device gate."""

    def __init__(self):
        """Keep no processes before main attaches the already-started runtime manager."""
        self.workers: WorkerManager | None = None
        self.session: WorkerSession | None = None
        self.model_id: str | None = None
        self.pins: RuntimePins | None = None
        self.tokenizer: Tokenizer | None = None
        self.window: int | None = None
        self._request = threading.Lock()

    def attach_workers(self, workers: WorkerManager) -> None:
        """Attach the application's existing runtime owner; a live resident from another owner must be retired first."""
        with self.exclusive():
            self.retire()
            self.workers = workers

    @contextmanager
    def exclusive(self):
        """Refuse concurrent requests instead of creating another model or an unbounded admission queue."""
        if not self._request.acquire(blocking=False):
            raise Busy("serving is busy; retry after the active request finishes")
        try:
            yield
        finally:
            self._request.release()

    def retire(self) -> None:
        """Verify resident worker cleanup before dropping its private package/tokenizer identity; uncertainty preserves quarantine."""
        if self.session is not None:
            self.session.close()
        self.session = self.pins = self.tokenizer = self.model_id = self.window = None

    def evict(self) -> None:
        """Retire an idle resident under its single request owner before a legacy request acquires the device."""
        with self.exclusive():
            self.retire()

    def ensure_loaded(self, model_name: str, cancel: threading.Event) -> None:
        """Reuse one exact resident package or verify/admit/load a registered saved model without parent tensor/source execution."""
        if cancel.is_set():
            raise RequestCancelled("serving request cancelled before loading")
        if self.session is not None and not self.session.closed and self.model_id == model_name:
            return
        self.retire()
        if self.workers is None:
            raise WorkerStopped("supervised serving runtime is unavailable")

        def check_cancelled():
            """Stop bounded parent verification/archive/grant work when its interactive owner disconnects."""
            if cancel.is_set():
                raise RequestCancelled("serving input preparation cancelled")

        model_id = UUID(model_name)
        with store.session() as db:
            model = q.get_model(db, model_id)
            if model is None:
                raise FileNotFoundError("saved model is not registered")
            expected = model.definition_id
        checked = package.verify(storage.model_dir(model_name), check_cancelled=check_cancelled)
        if expected is not None and checked.manifest.definition_id != expected:
            raise ValueError("saved package differs from indexed model definition")
        receipt = next((item for item in checked.receipts if "eval" in item.identity.profile.modes
            and any(domain.batch_min == 1 and domain.sequence_min == 1
                and domain.sequence_max >= checked.manifest.definition.context_window
                and "full_prefix" in domain.signatures for domain in item.identity.profile.domains)), None)
        if receipt is None:
            raise ValueError("saved model has no full-prefix serving profile covering its trained context")
        profile = receipt.identity.profile
        policy, budget = serving_policy(checked, self.workers.root, profile.device)
        legacy_chat.evict()
        session = self.workers.resident_session(policy, SERVING_BACKEND)
        self.session = session
        pins = RuntimePins(definition_id=checked.manifest.definition_id,
            bindings_digest=receipt.identity.parameter_digest,
            profile_digest=canonical_digest(profile), receipt_id=receipt.receipt_id)
        archive = session.directory / "load-package.tar"
        grants = []

        try:
            package_transfer.write_archive(checked.directory, archive, budget=budget, check_cancelled=check_cancelled)
            definition = session.artifacts.write_bytes(encode_json(checked.manifest.definition), "definition_v1")
            grants.append(definition)
            grant = session.artifacts.stage_file(archive, "model_package_v1", check_cancelled=check_cancelled)
            grants.append(grant)
            result = session.run(Load(type="load", pins=pins, definition=definition, package=grant), cancel=cancel)
            if result.outcome != "ok" or session.closed:
                raise WorkerStopped("serving model load was cancelled")
            self.tokenizer = Tokenizer.from_str(read_bytes(checked.directory / "tokenizer.json", 32 << 20).decode("utf-8"))
            self.model_id, self.pins = model_name, pins
            self.window = checked.manifest.definition.context_window
        except BaseException:
            self.retire()
            raise
        finally:
            archive.unlink(missing_ok=True)
            if not session.closed:
                for grant in grants:
                    session.artifacts.remove(grant)

    def stream(self, cfg: ChatConfig, cancel: threading.Event):
        """Yield bounded canonical chat frames while a separate owned thread drains the supervised worker; disconnect cancels/reaps it."""
        generation_bounds(cfg)
        with self.exclusive():
            self.ensure_loaded(cfg.model_name, cancel)
            prepared = GenerationInput(config=cfg, prompt_ids=tuple(prepare_prompt(cfg, self.tokenizer, self.window)))
            grant = self.session.artifacts.write_bytes(encode_json(prepared), "prompt_v1")
            session = self.session
            command = Generate(type="generate", pins=self.pins, prompt=grant, max_tokens=cfg.max_tokens,
                temperature=cfg.temperature, seed=cfg.seed or 0)
            events: Queue[Tokens] = Queue(maxsize=64)
            outcome: Future = Future()
            renderer = TokenRenderer(cfg, self.tokenizer)
            thread = threading.Thread(target=self.generate_request,
                args=(session, command, cancel, events, outcome), name="llm-serving-request", daemon=True)
            started = False
            try:
                thread.start()
                started = True
                while not outcome.done() or not events.empty():
                    try:
                        event = events.get(timeout=0.05)
                    except Empty:
                        continue
                    for token_id in event.token_ids:
                        if token_id >= self.tokenizer.get_vocab_size() or len(renderer.ids) >= min(self.window - len(prepared.prompt_ids), 2 * cfg.max_tokens):
                            raise WorkerStopped("worker generated tokens outside its saved vocabulary/context")
                        yield from renderer.push(token_id)
                result = outcome.result()
                if (result.model_name != cfg.model_name or result.prompt_tokens != len(prepared.prompt_ids)
                        or result.generated_tokens != len(renderer.ids) or result.text != bpe.decode(renderer.tokenizer, renderer.ids)
                        or result.drained_blocks > result.generated_tokens - result.drained_blocks
                        or result.generated_tokens - result.drained_blocks > cfg.max_tokens
                        or result.model_dump(mode="json")["tool_calls"] != renderer.calls.made):
                    raise WorkerStopped("generation result differs from its actual token stream")
                yield from renderer.finish()
                return result.model_dump(mode="json")
            except BaseException:
                cancel.set()
                self.retire()
                raise
            finally:
                if not outcome.done():
                    cancel.set()
                    session.close()
                if started:
                    thread.join(timeout=session.policy.deadlines.reap_seconds + 1)
                if started and thread.is_alive():
                    self.workers.gate.quarantine("serving request drain did not stop")
                    raise CleanupUncertain("serving request drain did not stop")
                if not session.closed:
                    session.artifacts.remove(grant)

    def generate_request(self, session: WorkerSession, command: Generate, cancel: threading.Event,
                         events: Queue, outcome: Future) -> None:
        """Own one blocking worker drain; bounded backpressure remains cancellable and never retains unbounded token events."""
        result: GenerationResult | None = None

        def on_event(event):
            """Queue only bounded token chunks and stop when the consumer disconnects instead of blocking cleanup."""
            nonlocal result
            if isinstance(event, Completed):
                if event.outcome != "ok" or event.result is None or event.result.format != "generation_result_v1":
                    raise WorkerStopped("worker returned no complete generation result")
                result = GenerationResult.model_validate_json(session.artifacts.read_bytes(event.result))
                session.artifacts.remove(event.result)
                return
            if not isinstance(event, Tokens):
                return
            while not cancel.is_set():
                try:
                    events.put(event, timeout=0.05)
                    return
                except Full:
                    pass
            raise RequestCancelled("generation consumer disconnected")

        try:
            session.run(command, on_event=on_event, cancel=cancel)
            if result is None:
                raise WorkerStopped("worker completed without an observed generation result")
            outcome.set_result(result)
        except BaseException as error:
            outcome.set_exception(error)

    def measure(self, cfg: EvalConfig, text: str, prompts: list[str], cancel: threading.Event) -> dict:
        """Return bounded language measurements through the resident owner, preserving the existing benchmark result contract."""
        prepared = MeasurementInput(config=cfg, text=text, prompts=tuple(prompts))
        return self.observe(cfg.model_name, prepared, cancel).model_dump(mode="json")

    def inspection_points(self, model_name: str, cancel: threading.Event) -> InspectionCatalog:
        """Discover actual declared attention views and existing proof bounds in the resident worker."""
        result = self.observe(model_name, InspectionInput(model_name=model_name), cancel)
        if not isinstance(result, InspectionCatalog):
            raise WorkerStopped("worker returned a different observation operation")
        return result

    def attention(self, model_name: str, selection: AttentionSelection, cancel: threading.Event) -> AttentionMatrix:
        """Request one occurrence/head measurement and require matching resident definition/profile and real token axes."""
        result = self.observe(model_name, AttentionInput(model_name=model_name, selection=selection), cancel)
        if not isinstance(result, AttentionMatrix):
            raise WorkerStopped("worker returned a different observation operation")
        return result

    def observe(self, model_name: str, prepared: MeasurementRequest, cancel: threading.Event) -> MeasurementResult | AttentionMatrix | InspectionCatalog:
        """Own one typed observation exchange, consuming its bounded data before any queued job can evict private result grants."""
        result_type = {"language": MeasurementResult, "attention": AttentionMatrix, "inspection_points": InspectionCatalog}[prepared.operation]
        result_adapter = TypeAdapter(result_type | InspectionRefusal) if isinstance(prepared, AttentionInput) else TypeAdapter(result_type)
        with self.exclusive():
            self.ensure_loaded(model_name, cancel)
            session = self.session
            grant = session.artifacts.write_bytes(encode_json(prepared), "measurement_v1")
            result = None

            def on_event(event):
                """Consume result bytes before queue-priority eviction can retire their private session grants."""
                nonlocal result
                if isinstance(event, Completed):
                    if event.outcome != "ok" or event.result is None or event.result.format != "measurement_result_v1":
                        raise WorkerStopped("worker returned no complete measurement result")
                    result = result_adapter.validate_json(session.artifacts.read_bytes(event.result))
                    session.artifacts.remove(event.result)

            try:
                session.run(Measure(type="measure", pins=self.pins, measurement=grant), on_event=on_event, cancel=cancel)
                if result is None:
                    raise WorkerStopped("worker completed without an observed measurement result")
                if isinstance(result, InspectionRefusal):
                    raise ValueError(result.message)
                self.require_observation(result, prepared, model_name, session)
                return result
            except BaseException:
                self.retire()
                raise
            finally:
                if not session.closed:
                    session.artifacts.remove(grant)

    def require_observation(self, result: MeasurementResult | AttentionMatrix | InspectionCatalog,
                            prepared: MeasurementRequest, model_name: str, session: WorkerSession) -> None:
        """Reject stale identities, incorrect axes or mismatched selections before exposing numerical worker data to callers."""
        if isinstance(result, MeasurementResult):
            if result.model.id != model_name or result.model.context != self.window or result.model.vocab != self.tokenizer.get_vocab_size():
                raise WorkerStopped("measurement differs from its loaded model identity")
            return
        if (result.model_name != model_name or result.definition_id != self.pins.definition_id
                or result.profile_digest != self.pins.profile_digest or result.runtime_id != session.worker_id
                or result.max_tokens != min(MAX_ATTENTION_SEQUENCE, self.window)):
            raise WorkerStopped("observation differs from its loaded model runtime identity")
        if isinstance(result, InspectionCatalog):
            if result.context_window != self.window:
                raise WorkerStopped("inspection catalog differs from its trained context")
            return
        selection = prepared.selection
        labels = tuple(bpe.token_text(self.tokenizer, token) for token in bpe.encode(self.tokenizer, selection.text))
        if (result.tokens != labels or result.head != selection.head or result.point_id is None
                or selection.point_id is not None and result.point_id != selection.point_id
                or selection.point_id is None and result.layer != (selection.layer or 0)):
            raise WorkerStopped("attention result differs from its requested point/head/token axes")


SERVING = ServingModels()


def inspection_points(model_name: str, cancel: threading.Event) -> InspectionCatalog:
    """Discover saved-model points inside its worker, or report actual legacy dimensions from data-only metadata reads."""
    if modular(model_name):
        return SERVING.inspection_points(model_name, cancel)
    directory = storage.model_dir(model_name)
    if not (directory / "model.pt").is_file():
        raise FileNotFoundError("model weights are unavailable")
    with store.session() as db:
        try:
            model_id = UUID(model_name)
        except ValueError:
            model_id = None
        row = q.get_model(db, model_id) if model_id else None
        facts = resolve_model_metadata(db, row, read_model_meta(directory), directory)
    shape = facts.architecture
    return InspectionCatalog(model_name=model_name, context_window=facts.context,
        max_tokens=min(MAX_ATTENTION_SEQUENCE, facts.context) if facts.context else None,
        legacy_layers=shape["layers"] if shape else None, legacy_heads=shape["heads"] if shape else None,
        unavailable_reason=None if shape else "Saved attention dimensions are unavailable")


def attention(model_name: str, selection: AttentionSelection, cancel: threading.Event) -> AttentionMatrix:
    """Select supervised occurrence-aware inspection or the legacy layer adapter under the one existing device owner."""
    if modular(model_name):
        return SERVING.attention(model_name, selection, cancel)
    if selection.point_id is not None:
        raise ValueError("legacy models require layer/head attention selection")
    if cancel.is_set():
        raise RequestCancelled("attention request cancelled before model loading")
    SERVING.evict()
    with device.admit():
        return AttentionMatrix.model_validate(legacy_attention.matrix(model_name, selection.text, selection.layer or 0, selection.head))


def stream(cfg: ChatConfig, *, cancel: threading.Event | None = None):
    """Select the supervised saved-model path or the existing legacy adapter without acquiring a second device lease."""
    if modular(cfg.model_name):
        return (yield from SERVING.stream(cfg, cancel or threading.Event()))
    SERVING.evict()
    return (yield from legacy_chat.stream(cfg))


def run(cfg: ChatConfig, *, cancel: threading.Event | None = None) -> dict:
    """Drain the selected serving iterator and return the existing complete stateless chat summary."""
    generator = stream(cfg, cancel=cancel)
    try:
        while True:
            next(generator)
    except StopIteration as done:
        return done.value
    finally:
        generator.close()


def saved_history_floor(cfg: ChatConfig) -> int:
    """Fit history from immutable saved context/tokenizer bytes without loading custom models or consuming a device lease."""
    manifest = package.read_manifest(storage.model_dir(cfg.model_name))
    tokenizer = bpe.load(storage.model_dir(cfg.model_name))
    window = manifest.definition.context_window
    return context_floor(history(cfg.turns), tokenizer, window,
        min(RESERVE_FOR_REPLY, max(1, window // 2)), cfg.system,
        ThinkBudget.parse(cfg.budget) if cfg.budget else None, cfg.project, cfg.memories)
