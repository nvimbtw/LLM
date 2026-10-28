"""The training loop: forward, backward, clip, step.

Everything device- and gradient-related that the Rust implementation spends
~2,000 lines on (workspace allocation, the hand-written backward pass, the Adam
kernels, host/device synchronization) is autograd and torch.optim here.
"""

import gc
import math
import time
from threading import Event
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from llm.config import TrainConfig
from llm.model import checkpoint
from llm.model.checkpoint import TrainState
from llm.model.transformer import Transformer
from llm.model.language import logits, model_facts
from llm.train import memory
from llm.train.diagnostics_model import ModelCollector
from llm.train.model_memory import ModelMemoryPlan, owned_tensors
from llm.train.model_policy import optimizer_groups
from llm.train import plan as plan_mod
from llm.train.diagnostics import Collector, Probes, diagnostic_rows
from llm.train.plan import DataPlan
from llm.train.progress import (
    Checkpoint,
    Progress,
    ProgressCallback,
    checkpoint_every,
    diagnose_every,
    report_every,
)

#: What the loop asks of a checkpoint writer. It decides *when*; the writer
#: decides what a checkpoint directory is and answers with the frame describing
#: the one it just wrote — which is why it is told the run's shape as well as the
#: moment: `total_steps` and `stage_name` are fields of that frame and nothing
#: below the orchestrator knows them.
CheckpointWriter = Callable[..., Checkpoint]

# The floor of the cosine decay, as a fraction of the peak. Training all the way
# to zero wastes the tail of the run on steps that barely move the weights.
MIN_LR_RATIO = 0.1


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def release_device() -> None:
    """Hand the caching allocator's pool back to the driver.

    Torch never does this on its own: a freed block stays reserved so the next
    allocation of that size does not have to ask the driver again, which is the
    right trade inside a run and the wrong one between them. The LLM engine is a
    long-lived process — `api/runner.py` runs training on a thread of the same
    uvicorn that serves chat — so without this a run's high-water mark is what
    the GPU reports as occupied until the process exits, and the next run starts
    against a card that looks full.

    Cached blocks and live tensors need different cleanup: collect unreachable
    reference cycles first, then release unoccupied reservations. Callers must
    drop live owners too; the runner clears failed traceback frames before this
    call because empty_cache cannot free tensors a worker future still holds.

    It also lifts the allocation cap `memory.guard` puts on for the duration of a
    run. That cap is what turns an oversized step into an immediate
    `OutOfMemoryError` instead of a silent spill into host RAM, and it is sized
    against what was free when *that* run started — so leaving it in place would
    hold every later chat and every later run to one run's budget. Both halves
    belong at the same boundary, which is this one.

    Cheap (a few ms, once per run) and a no-op off CUDA, so callers do not have
    to know which device the run they just finished was on.
    """
    if torch.cuda.is_available():
        # Compiled graphs and failed forwards can leave tensors in reference
        # cycles. empty_cache cannot release storage while those tensors live.
        gc.collect()
        torch.cuda.set_per_process_memory_fraction(1.0)
        torch.cuda.empty_cache()


def autocast_dtype(precision: str, device: str) -> torch.dtype | None:
    """The autocast dtype for this run, or None to stay in fp32.

    bf16 is the default and matters more than it looks: `scaled_dot_product_
    attention` has no flash kernel for fp32 inputs, so an fp32 run silently falls
    back to the memory-efficient or math backend and pays for the attention matrix
    it did not need to materialize. Measured on an RTX 3060 at 256d/4h/6L/ctx512,
    bf16 is ~1.7x fp32 and torch.compile on top of it ~2.5x.

    bf16 over fp16 because it has fp32's exponent range: no GradScaler, no loss
    scaling to tune, and no silent overflow to inf in the backward pass. Master
    weights stay fp32 under autocast either way — only the ops are cast.
    """
    if device != "cuda" or precision == "fp32":
        return None
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        return torch.float16
    return torch.bfloat16 if precision == "bf16" else torch.float16


def compiled(fn, enabled: bool):
    """`torch.compile` a callable, falling back to eager if compilation fails.

    A failure here is a performance regression, not a broken run — the same
    weights train either way — so it must not take the run down with it. Inductor
    needs a working C compiler and writable cache, neither guaranteed.
    """
    if not enabled:
        return fn
    try:
        return torch.compile(fn)
    except Exception:
        return fn


def step_fn(model: torch.nn.Module, vocab: int):
    """The forward pass and its loss as one callable, so both compile together.

    Compiling the model alone leaves cross-entropy in eager, and on this model
    that is not a rounding error: the logits are [batch x ctx, vocab], autocast
    puts cross-entropy on its fp32 list, and eager therefore materializes the
    whole tensor a second time in fp32 before reducing it away. Measured at
    256d/8k-vocab/ctx512/bs8, that cast and the softmax around it were 14 of 33
    ms per step. Inside the graph, inductor fuses the cast into the softmax's
    reduction and the fp32 copy never exists — the same arithmetic, 3.6 ms/step
    cheaper, and peak memory drops from 724 to 430 MiB.
    """

    def forward_loss(x, y, w):
        """Compute next-token loss through the deliberate legacy or declared whole-model logits interface."""
        return _loss(logits(model,x), y, w, vocab)

    return forward_loss


_TORCH_DTYPE = {np.dtype(np.int64): torch.int64, np.dtype(np.float32): torch.float32}


def staging_allocator(device: str):
    """Allocate `iter_batches`' buffers in pinned memory, on CUDA.

    A copy out of ordinary pageable memory has to be staged through a pinned
    bounce buffer by the driver; allocating the batch there in the first place
    skips that. Measured at the batch shape this loop uses, 117 -> 72 us per
    copy, twice a step.

    The copy stays blocking. Overlapping it would need the buffer not to be
    rewritten while the transfer is in flight, and there is one buffer, reused
    every batch — the honest version of `non_blocking=True` here is a race, and
    the whole transfer is 0.15 ms of a 24 ms step.
    """
    if device != "cuda":
        return None

    def allocate(n: int, dtype) -> np.ndarray:
        return torch.empty(n, dtype=_TORCH_DTYPE[np.dtype(dtype)], pin_memory=True).numpy()

    return allocate


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to MIN_LR_RATIO of the peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)

    decay_steps = max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)
    scale = MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * scale


def step_plan(
    corpus: np.ndarray,
    data_plan: DataPlan,
    cfg: TrainConfig,
    mask: np.ndarray | None = None,
) -> tuple[list[int], int]:
    """Batches per epoch of each stage, and the run's total across all of them.

    Out here rather than buried in `run` because a run that names `resume_from`
    has to compare this number against the one its saved state was taken under
    before it decides whether it is continuing that run or starting a new one
    from its weights — and that question is answered before anything is trained.
    """
    stages = data_plan.ordered_stages(cfg.epochs)
    per_stage = [
        plan_mod.batch_count(
            len(plan_mod.EpochStream(corpus, data_plan, 0, mask, stage.documents)),
            cfg.ctx_window,
            cfg.batch_size,
        )
        for stage in stages
    ]
    return per_stage, sum(n * st.epochs for n, st in zip(per_stage, stages))


class _Checkpoints:
    """When a run takes one, and the bookkeeping that keeps it to once each.

    Three rules ask for a checkpoint and they overlap on purpose: the interval,
    the end of every epoch, and the last step of the run. `checkpoint_every`
    snaps the interval onto a step that is already reporting, so an epoch ending
    on one is the common case rather than the rare one — and written twice that
    would be the same weights under two ids, two directories and two rows.
    """

    def __init__(
        self,
        write: CheckpointWriter | None,
        publish: ProgressCallback | None,
        interval: int,
        total_steps: int,
    ):
        self.write = write
        self.publish = publish
        self.interval = interval
        self.total_steps = total_steps
        self.index = 0
        self.last = -1

    def _due(self, step: int, epoch_end: bool) -> bool:
        # A zero interval is `checkpoint_target=0`, which means none at all — the
        # epoch and final-step rules are what a cadence is *made of*, not
        # exceptions to it, so they go quiet with it. The run's own model is
        # still saved; that has never been this class's business.
        if self.write is None or not self.interval or step == self.last:
            return False
        return epoch_end or step >= self.total_steps or step % self.interval == 0

    def at(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        step: int,
        epoch: int,
        stage: int,
        stage_name: str | None,
        loss: Callable[[], float],
        epoch_end: bool = False,
        cursor: dict | None = None,
    ) -> str | None:
        """Write one if this step calls for it, and answer with its state file.

        `loss` is a callable and not a number because reading one drains the
        device-side loss buffer — a host sync, on the training path, which a run
        taking no checkpoints must not start paying for the ones it is not
        taking.

        The frame goes out on the progress callback because that is the seam a
        run already has for things it must say while it is still running. The
        path is returned as well, and the accompanying `Progress` carries it —
        but only the frame is a complete account of the run's checkpoints: a
        boundary save deduplicated against an interval one leaves its epoch-end
        frame carrying no path at all, so anything listing them reads the frames.
        """
        if not self._due(step, epoch_end):
            return None

        self.last = step
        self.index += 1
        frame = self.write(
            model,
            optimizer,
            step=step,
            total_steps=self.total_steps,
            epoch=epoch,
            stage=stage,
            stage_name=stage_name,
            loss=loss(),
            cursor=cursor,
        )
        if self.publish is not None:
            self.publish(frame)
        return frame.path


def _resume(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    state: TrainState | None,
) -> int:
    """Put the optimizer and the generators back, and answer with the step to
    re-enter at.

    Saved moments and RNG are restored, while the newly constructed optimizer's
    group options retain the requested decay and admitted device policy. The run
    fast-forwards to `state.step`. A continuation that means to keep the moments
    but not the position hands over a state whose step is 0 — which of the two a
    request meant is the orchestrator's to decide, because it is the only layer
    that knows whether the corpus and the schedule are the same ones.

    The weights are not loaded here. A resumed run's architecture comes from the
    state, so the model was built from it and already holds them by the time this
    runs; loading them again here would make that silently optional.
    """
    if state is None:
        return 0
    if state.optimizer:
        options = [{key: value for key, value in group.items() if key != "params"}
                   for group in optimizer.param_groups]
        optimizer.load_state_dict(state.optimizer)
        for group, requested in zip(optimizer.param_groups, options, strict=True):
            group.update(requested)
    checkpoint.restore_rng(state.rng)
    return state.step


def run(
    model: torch.nn.Module,
    corpus: np.ndarray,
    data_plan: DataPlan,
    cfg: TrainConfig,
    device: str,
    on_progress: ProgressCallback | None = None,
    cancel: Event | None = None,
    mask: np.ndarray | None = None,
    token_counts: list[int] | None = None,
    probes: Probes | None = None,
    on_checkpoint: CheckpointWriter | None = None,
    resume: TrainState | None = None,
    model_plan: ModelMemoryPlan | None = None,
    on_batch: Callable[[int], None] | None = None,
) -> dict:
    """Train `model` in place. Returns a summary; True in `cancelled` means the
    run stopped at a step boundary with its weights intact.

    `token_counts` is the histogram the corpus build recorded, forwarded to the
    diagnostics so they never have to derive it from the corpus.

    `probes` are the fixed batches a fine-tune is measured against — see
    `diagnostics.Probes`. This function does not build them: what a base model
    is, and where its corpus and the held-out chat live, is `orchestrate/sft.py`'s
    to know. Nothing here is retained past the return, so the reference model
    (when one was asked for) is collectable before `release_device` runs.

    `on_checkpoint` is called at the moments `_Checkpoints` decides. Without one
    a run takes none at all, which is what `bench.py` and the tests want.

    `resume` restores optimizer/RNG state and an explicit epoch/batch cursor.
    Masked batches can advance the cursor without an optimizer step. Legacy
    points without a cursor use the old step-based fallback; exact SFT recovery
    refuses those points in the orchestrator.

    A declared recipe model requires its actual captured/admitted model_plan
    from in-worker final preflight. Its facts own context/vocabulary; legacy
    shape defaults never configure it. Recomputation is already selected by that
    preflight, without changing structural loop counts here.

    Learning rate, reporting and checkpoint cadence share one optimizer-step
    counter. Epoch shuffling uses zero-based epoch identities; frames report
    one-based epochs. Cancellation lands before an update, gradients are dropped
    before each forward and saved cursors name the next unconsumed batch.
    on_batch reports consumed batches for worker liveness, including masked
    batches without an update; it never substitutes for optimizer-step telemetry.
    """
    facts = model_facts(model)
    if isinstance(model,Transformer) and model_plan is not None:
        raise ValueError("legacy Transformer training owns its measured legacy memory planner")
    if not isinstance(model,Transformer):
        cfg = cfg.model_copy(update={"ctx_window":facts.context_window})
        if model_plan is None:
            raise ValueError("recipe training requires captured final-profile memory admission")
        actual_bytes = sum(value.numel() * value.element_size() for value in owned_tensors(model).values())
        if actual_bytes != model_plan.parameter_bytes or model_plan.estimated_bytes > model_plan.target_bytes:
            raise ValueError("captured memory plan differs from the admitted model")
        if model_plan.input_shapes != ((cfg.batch_size,facts.context_window),)*3:
            raise ValueError("captured full-prefix memory profile differs from the training batch/context")
        if probes is not None and probes.reference is not None:
            reference = probes.reference
            reference_bytes = sum(value.numel() * value.element_size() for value in owned_tensors(reference).values())
            if any(value.requires_grad for value in reference.parameters()) or model_plan.reference_bytes < reference_bytes:
                raise ValueError("captured memory plan does not admit the frozen SFT reference")
    budget = model_plan.allocator_bytes if model_plan is not None else memory.budget_bytes(device)
    resident_bytes = memory.other_allocated_bytes(model, device)
    memory.guard(device, budget)
    model.to(device)
    model.train()
    eager_forward = step_fn(model, facts.vocabulary_size)
    forward = compiled(eager_forward, cfg.compile_model and device == "cuda")
    memory_plan = model_plan or memory.plan(
        model, batch_size=cfg.batch_size, device=device,
        requested=cfg.recompute_blocks, budget=budget,
        compiled=forward is not eager_forward,
        resident_bytes=resident_bytes,
    )
    if isinstance(model,Transformer):
        model.recompute_depth = memory_plan.recompute_depth
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    print(f"[memory] {memory_plan.reason}")

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = cfg.allow_tf32
        torch.backends.cudnn.allow_tf32 = cfg.allow_tf32

    amp_dtype = autocast_dtype(cfg.precision, device)
    groups = optimizer_groups(model,weight_decay=cfg.weight_decay)

    optimizer = torch.optim.AdamW(
        groups,
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
        fused=(device == "cuda"),
    )

    stages = data_plan.ordered_stages(cfg.epochs)
    stage_steps, total_steps = step_plan(corpus, data_plan, cfg, mask)
    if total_steps == 0:
        raise ValueError(
            f"Corpus of {len(corpus)} tokens is too small for ctx_window "
            f"{cfg.ctx_window} x batch_size {cfg.batch_size} — it yields no full "
            f"batches. Add data, or lower either setting."
        )

    collector = _LazyCollector(model, corpus, token_counts, probes)

    interval = report_every(total_steps)
    diag_interval = diagnose_every(max(1, total_steps // interval))
    total_epochs = sum(st.epochs for st in stages)
    checkpoints = _Checkpoints(
        on_checkpoint,
        on_progress,
        checkpoint_every(total_steps, total_epochs, cfg.checkpoint_target),
        total_steps,
    )

    started = time.time()
    resumed_at = _resume(model, optimizer, resume)
    step = resumed_at
    staging = staging_allocator(device)
    log = _LossLog(interval, device)
    grad = None

    def report(
        epoch: int,
        stage_index: int,
        stage,
        chunk: int,
        steps_this_epoch: int,
        lr: float,
        epoch_started: float,
        done: bool = False,
        checkpoint_path: str | None = None,
    ) -> None:
        """Publish measured windows with one-based epoch identity and optional bounded diagnostics."""
        log.flush()
        epoch_losses = log.epoch
        window = log.values[-interval:] or [0.0]
        elapsed = time.time() - started
        remaining = max(0, total_steps - step)
        taken = step - resumed_at
        eta = (elapsed / taken * remaining) if taken else 0.0
        diagnostics = collector().after_step(grad, step) if (grad and not done) else None

        on_progress(
            Progress(
                step=step,
                total_steps=total_steps,
                epoch=epoch + 1,
                total_epochs=total_epochs,
                chunk=chunk,
                total_chunks=steps_this_epoch,
                loss=float(np.mean(window)),
                lr=lr,
                elapsed_secs=elapsed,
                avg_loss=float(np.mean(epoch_losses)) if epoch_losses else 0.0,
                min_loss=float(np.min(epoch_losses)) if epoch_losses else 0.0,
                max_loss=float(np.max(epoch_losses)) if epoch_losses else 0.0,
                batch_min=float(np.min(window)),
                batch_avg=float(np.mean(window)),
                batch_max=float(np.max(window)),
                eta_secs=eta,
                epoch_done=done,
                epoch_duration_secs=time.time() - epoch_started,
                checkpoint_path=checkpoint_path,
                peak_mib=(
                    torch.cuda.max_memory_allocated() / 2**20
                    if device == "cuda" else None
                ),
                budget_mib=memory_plan.budget_mib if device == "cuda" else None,
                stage=stage_index,
                stage_name=stage.name,
                diag=diagnostics.as_dict() if diagnostics is not None else None,
            )
        )

    def current_loss() -> float:
        """The number the frame accompanying this checkpoint will carry."""
        log.flush()
        return float(np.mean(log.values[-interval:] or [0.0]))

    epochs_done = 0
    to_skip = resumed_at
    consumed_batches = 0
    cursor = resume.extra.get("cursor") if resume is not None else None

    for stage_index, stage in enumerate(stages):
      for stage_epoch in range(stage.epochs):
        epoch = epochs_done + stage_epoch
        steps_this_epoch = stage_steps[stage_index]
        if cursor is not None and epoch < cursor["epoch"]:
            continue
        if cursor is None and to_skip >= steps_this_epoch:
            to_skip -= steps_this_epoch
            continue

        start_batch = cursor["batch"] if cursor is not None and epoch == cursor["epoch"] else to_skip
        to_skip = 0
        epoch_started = time.time()
        log.start_epoch()
        chunk = start_batch
        stream = plan_mod.EpochStream(corpus, data_plan, epoch, mask, stage.documents)

        for batch_index, (inputs, targets, weights, offset) in enumerate(plan_mod.iter_batches(
            stream, cfg.ctx_window, cfg.batch_size, mask is not None, staging,
            start_batch=start_batch,
        ), start=start_batch):
            if cancel is not None and cancel.is_set():
                log.flush()
                return _summary(step, total_steps, log.values, started, cancelled=True,
                                plan=memory_plan)

            if weights is not None and not weights.any():
                consumed_batches += 1
                if on_batch is not None:
                    on_batch(consumed_batches)
                continue

            x = torch.from_numpy(inputs).to(device)
            y = torch.from_numpy(targets).to(device)
            w = torch.from_numpy(weights).to(device) if weights is not None else None

            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            first_step = step == resumed_at
            reporting = (step + 1) % interval == 0 or first_step
            measuring = on_progress is not None and reporting
            diagnosing = (
                measuring
                and not first_step
                and ((step + 1) // interval) % diag_interval == 0
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=amp_dtype is not None
            ):
                if diagnosing:
                    collector().observe(x, y, w, _diagnosed_documents(stream, offset, cfg, w))
                loss = forward(x, y, w)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            grad = collector().before_step() if diagnosing else None

            optimizer.step()
            consumed_batches += 1
            if on_batch is not None:
                on_batch(consumed_batches)

            log.record(loss)
            step += 1
            chunk += 1

            saved = (
                checkpoints.at(
                    model, optimizer,
                    step=step, epoch=epoch + 1, stage=stage_index,
                    stage_name=stage.name, loss=current_loss,
                    cursor={"epoch": epoch, "batch": batch_index + 1},
                )
                if reporting
                else None
            )

            if measuring:
                report(epoch, stage_index, stage, chunk, steps_this_epoch,
                       lr, epoch_started, checkpoint_path=saved)
                time.sleep(0.002)

        saved = checkpoints.at(
            model, optimizer,
            step=step, epoch=epoch + 1, stage=stage_index, stage_name=stage.name,
            loss=current_loss, epoch_end=True,
            cursor={"epoch": epoch + 1, "batch": 0},
        )
        if on_progress:
            grad = None
            report(epoch, stage_index, stage, chunk, steps_this_epoch,
                   lr, epoch_started, done=True, checkpoint_path=saved)
      epochs_done += stage.epochs

    log.flush()
    return _summary(step, total_steps, log.values, started, cancelled=False,
                    plan=memory_plan)


class _LazyCollector:
    """The diagnostics collector, built on the step that first needs it.

    Constructing one has to establish which tokens the embedding metrics track,
    and on a corpus whose build did not record its histogram that means reading
    the corpus. It used to happen at the top of every run, unconditionally:
    measured at 64 s and 8.3 GB of the mapping made resident before the first
    step — including on runs with `on_progress=None`, which never diagnose
    anything and never read the result. `bench.py` pays that on every timing.

    Diagnostics are sampled across the reports, so the first call comes at least
    `interval * diag_interval` steps in, by which point the run is warm and the
    cost is off the critical path either way. Declared recipe models instead use
    bounded semantic ModelCollector samples, without inferred embedding/layer fields.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        corpus: np.ndarray,
        counts: list[int] | None,
        probes: Probes | None = None,
    ):
        """Retain diagnostic inputs without allocating the optional collector before first use."""
        self._args = (model, corpus, counts, probes)
        self._collector: Collector | ModelCollector | None = None

    def __call__(self) -> Collector | ModelCollector:
        """Create the model-appropriate measured diagnostics only when the existing cadence requests them."""
        if self._collector is None:
            self._collector = (Collector(*self._args) if isinstance(self._args[0],Transformer)
                               else ModelCollector(self._args[0], self._args[3]))
        return self._collector


def _diagnosed_documents(stream, offset: int, cfg: TrainConfig, weights) -> np.ndarray | None:
    """Which conversation each position of the diagnosed rows came from.

    None without a mask: a pretrain corpus has documents too, but per-document
    loss is a question about a *curated* dataset, and walking the span table on
    a pretrain run would be cost for a chart that run does not draw.

    Only the rows `Collector.observe` actually runs over — the walk is over the
    window, not the batch, and the arrays have to be the same length.
    """
    if weights is None:
        return None
    rows = diagnostic_rows(cfg.batch_size)
    return stream.document_starts(offset, offset + rows * cfg.ctx_window)


class _LossLog:
    """Per-step losses, read back from the device one report window at a time.

    `loss.item()` after every step is a synchronization point: it blocks the host
    until that step's kernels have retired, so the host cannot queue the next
    step's work while the current one runs. On a step this small — a few hundred
    kernels the host is already only just keeping ahead of — that costs about
    1.2 ms of 29, for a number nothing reads until the next report frame.

    So the losses accumulate in a device buffer and come back in one transfer, at
    exactly the moment a consumer first needs them. Same values, one sync per
    ~150 steps instead of one per step.
    """

    def __init__(self, window: int, device: str):
        self.buffer = torch.zeros(window, device=device)
        self.pending = 0
        self.values: list[float] = []
        self.epoch: list[float] = []

    def record(self, loss) -> None:
        self.buffer[self.pending] = loss.detach()
        self.pending += 1
        # The window is the report interval, so this drains on the same step the
        # report does. It is here as well because a skipped SFT window makes the
        # two counts drift, and overrunning the buffer would be an IndexError
        # thousands of steps into a run.
        if self.pending == len(self.buffer):
            self.flush()

    def flush(self) -> None:
        if self.pending:
            drained = self.buffer[: self.pending].tolist()
            self.values.extend(drained)
            self.epoch.extend(drained)
            self.pending = 0

    def start_epoch(self) -> None:
        self.flush()
        self.epoch = []


def _loss(logits, targets, weights, vocab: int):
    """Cross-entropy, restricted to the masked positions when SFT supplies one.

    Averaged over the *trainable* tokens rather than all of them: dividing by the
    window would make a batch's loss depend on how much user context happened to
    sit beside the reply, so identical replies would report different losses.

    A window with nothing trainable in it never reaches here — the caller drops
    it while the mask is still a host array, so this can divide without first
    reading a device scalar back.
    """
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)

    if weights is None:
        return F.cross_entropy(flat_logits, flat_targets)

    w = weights.reshape(-1)
    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (per_token * w).sum() / w.sum()


def _summary(step, total_steps, losses, started, cancelled: bool, plan=None) -> dict:
    summary = {
        "steps": step,
        "total_steps": total_steps,
        "final_loss": float(np.mean(losses[-100:])) if losses else None,
        "elapsed_secs": round(time.time() - started, 1),
        "cancelled": cancelled,
    }
    if plan is not None:
        # What the run spent to fit, recorded beside what it produced. Two runs
        # of the same architecture with very different wall-clocks are otherwise
        # indistinguishable after the fact — and the depth is the likeliest
        # reason, since it is chosen from whatever the card happened to have free.
        summary["memory"] = plan.as_dict()
        if torch.cuda.is_available():
            summary["memory"]["peak_mib"] = round(
                torch.cuda.max_memory_allocated() / 2**20, 1
            )
    return summary
