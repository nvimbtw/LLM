"""The training loop: forward, backward, clip, step.

Everything device- and gradient-related that the Rust implementation spends
~2,000 lines on (workspace allocation, the hand-written backward pass, the Adam
kernels, host/device synchronization) is autograd and torch.optim here.
"""

import math
import time
from contextlib import contextmanager
from threading import Event

import numpy as np
import torch
import torch.nn.functional as F

from llm import storage
from llm.config import TrainConfig
from llm.model import checkpoint
from llm.model.transformer import Transformer
from llm.train import plan as plan_mod
from llm.train.diagnostics import Collector
from llm.train.plan import DataPlan
from llm.train.progress import Progress, ProgressCallback, diagnose_every, report_every

# The floor of the cosine decay, as a fraction of the peak. Training all the way
# to zero wastes the tail of the run on steps that barely move the weights.
MIN_LR_RATIO = 0.1


def pick_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def lr_at(step: int, total_steps: int, cfg: TrainConfig) -> float:
    """Linear warmup, then cosine decay to MIN_LR_RATIO of the peak."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / max(1, cfg.warmup_steps)

    decay_steps = max(1, total_steps - cfg.warmup_steps)
    progress = min(1.0, (step - cfg.warmup_steps) / decay_steps)
    scale = MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress))
    return cfg.lr * scale


def run(
    model: Transformer,
    corpus: np.ndarray,
    data_plan: DataPlan,
    cfg: TrainConfig,
    device: str,
    on_progress: ProgressCallback | None = None,
    cancel: Event | None = None,
    mask: np.ndarray | None = None,
) -> dict:
    """Train `model` in place. Returns a summary; True in `cancelled` means the
    run stopped at a step boundary with its weights intact."""
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        betas=(0.9, 0.95),
        weight_decay=cfg.weight_decay,
    )

    # Every epoch reorders the same documents, so the batch count is stable and
    # the schedule can be built from one measurement.
    stages = data_plan.ordered_stages(cfg.epochs)
    # Each stage holds different documents, so its step count is its own.
    stage_steps = []
    for stage in stages:
        stream, _ = plan_mod.epoch_stream(corpus, data_plan, 0, mask, stage.documents)
        stage_steps.append(plan_mod.batch_count(len(stream), cfg.ctx_window, cfg.batch_size))
    total_steps = sum(n * st.epochs for n, st in zip(stage_steps, stages))
    if total_steps == 0:
        raise ValueError(
            f"Corpus of {len(corpus)} tokens is too small for ctx_window "
            f"{cfg.ctx_window} x batch_size {cfg.batch_size} — it yields no full "
            f"batches. Add data, or lower either setting."
        )

    # Tracks the same frequent tokens for the whole run; see frequent_tokens.
    collector = Collector(model, corpus)

    # Cadence follows the run's length, so a 200-step run and a 200k-step run
    # both draw a curve rather than a line or a flood.
    interval = report_every(total_steps)
    diag_interval = diagnose_every(max(1, total_steps // interval))

    started = time.time()
    step = 0
    losses: list[float] = []
    grad = None

    def report(
        # 0-based, as the loop counts them; reported 1-based. The two differ on
        # purpose: the shuffle seed is derived from the 0-based index, while the
        # consumer's contract is `step = (epoch - 1) * total_chunks + chunk`, so
        # emitting 0-based here would both read as off-by-one and break the
        # (epoch, chunk) reconstruction a reattaching socket does.
        epoch: int,
        stage_index: int,
        stage,
        chunk: int,
        steps_this_epoch: int,
        epoch_losses: list[float],
        lr: float,
        epoch_started: float,
        done: bool = False,
        checkpoint_path: str | None = None,
    ) -> None:
        window = losses[-interval:] or [0.0]
        elapsed = time.time() - started
        # Projected from the rate achieved so far rather than from the last
        # window: a single slow batch should not swing the estimate a person is
        # reading off the screen.
        remaining = max(0, total_steps - step)
        eta = (elapsed / step * remaining) if step else 0.0

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
                stage=stage_index,
                stage_name=stage.name,
                diag=collector.after_step(grad, step).as_dict() if (grad and not done) else None,
            )
        )
    storage.ensure_dir(storage.checkpoints_dir(cfg.model_name))

    epochs_done = 0
    total_epochs = sum(st.epochs for st in stages)

    for stage_index, stage in enumerate(stages):
      for stage_epoch in range(stage.epochs):
        epoch = epochs_done + stage_epoch
        epoch_started = time.time()
        epoch_losses: list[float] = []
        chunk = 0
        steps_this_epoch = stage_steps[stage_index]
        # Seeded off the global epoch so no two epochs of a run — in any stage —
        # see the same document order.
        stream, stream_mask = plan_mod.epoch_stream(
            corpus, data_plan, epoch, mask, stage.documents
        )

        for inputs, targets, weights in plan_mod.iter_batches(
            stream, cfg.ctx_window, cfg.batch_size, stream_mask
        ):
            # Cooperative cancellation at a step boundary: the weights are
            # consistent here, mid-backward they are not.
            if cancel is not None and cancel.is_set():
                return _summary(step, total_steps, losses, started, cancelled=True)

            x = torch.from_numpy(inputs).to(device)
            y = torch.from_numpy(targets).to(device)

            lr = lr_at(step, total_steps, cfg)
            for group in optimizer.param_groups:
                group["lr"] = lr

            # Diagnostics change the attention path and retain activations, so
            # they are scoped to exactly the step that is measured.
            measuring = on_progress is not None and (step + 1) % interval == 0
            # Diagnostics are sampled across the reports, not taken on each one.
            diagnosing = measuring and ((step + 1) // interval) % diag_interval == 0
            context = collector if diagnosing else _nothing()

            with context:
                logits, _ = model(x)
                loss = _loss(logits, y, weights, model.vocab, device)

            # An SFT window can contain no assistant tokens at all, which makes
            # the loss exactly zero and its gradient nothing. Stepping on it is
            # harmless but pollutes the reported average with zeros that are not
            # measurements of anything, so the window is skipped outright.
            if loss is None:
                continue

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

            # Between backward and step: the gradients still exist, and the
            # weights have not moved yet. Both halves of the update ratio need
            # this exact window.
            grad = collector.before_step() if diagnosing else None

            optimizer.step()

            losses.append(loss.item())
            epoch_losses.append(loss.item())
            step += 1
            chunk += 1

            if measuring:
                report(epoch, stage_index, stage, chunk, steps_this_epoch,
                       epoch_losses, lr, epoch_started)
                # Hand the GIL to the event loop so the frame just published
                # actually goes out.
                #
                # Training runs on a worker thread and publishes through
                # `call_soon_threadsafe`; the loop needs the GIL to drain that
                # queue and write the SSE frame. A tight training loop holds it
                # almost continuously, so without this the frames accumulate and
                # flush only when the run ends — the whole run appears at once,
                # which is indistinguishable from "streaming is broken".
                #
                # `time.sleep` is the yield that actually releases the GIL;
                # `sleep(0)` does not. At ~150 reports this costs well under a
                # second across an entire run.
                time.sleep(0.002)

        saved = checkpoint.save(
            model,
            storage.checkpoints_dir(cfg.model_name) / f"epoch{epoch + 1:03d}.pt",
            epoch=epoch + 1,
        )
        # An epoch-boundary frame, so the chart gets a point at the end of every
        # epoch rather than only when a report interval happens to land there.
        if on_progress:
            grad = None
            report(epoch, stage_index, stage, chunk, steps_this_epoch,
                   epoch_losses, lr, epoch_started, done=True, checkpoint_path=str(saved))
      epochs_done += stage.epochs

    return _summary(step, total_steps, losses, started, cancelled=False)


@contextmanager
def _nothing():
    """A no-op stand-in for the collector on unmeasured steps."""
    yield


def _loss(logits, targets, weights, vocab: int, device: str):
    """Cross-entropy, restricted to the masked positions when SFT supplies one.

    Averaged over the *trainable* tokens rather than all of them: dividing by the
    window would make a batch's loss depend on how much user context happened to
    sit beside the reply, so identical replies would report different losses.
    """
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = targets.reshape(-1)

    if weights is None:
        return F.cross_entropy(flat_logits, flat_targets)

    w = torch.from_numpy(weights).to(device).reshape(-1)
    trainable = w.sum()
    if trainable.item() == 0:
        return None

    per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    return (per_token * w).sum() / trainable


def _summary(step, total_steps, losses, started, cancelled: bool) -> dict:
    return {
        "steps": step,
        "total_steps": total_steps,
        "final_loss": float(np.mean(losses[-100:])) if losses else None,
        "elapsed_secs": round(time.time() - started, 1),
        "cancelled": cancelled,
    }
