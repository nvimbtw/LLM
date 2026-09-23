"""The reinforcement-learning pipeline: roll out, score, update, repeat.

    1. load the base model and its tokenizer, and a frozen copy of it
    2. per iteration: create a gym run, roll its episodes out, score the groups
    3. take one optimizer step over the tokens the model itself drew
    4. checkpoint, which is a model, and measure it with an ordinary gym run
    5. save weights + tokenizer + meta.json

The architecture comes from the base checkpoint rather than from the request,
exactly as a fine-tune's does: a run cannot silently disagree with the weights it
started from.

**One iteration is one gym run.** Not a convenience — it is what makes an
iteration reproducible from a row, and it means the trajectory record, the step
rewards and the grading are the gym's exactly as they are for a run a person
watched. The LLM engine writes down no trajectory of its own.

**A run needs both models resident**, the policy and the frozen reference, plus a
batch of rollouts in flight. That is a real change in what a run needs to fit,
and it is not optional: the KL anchor is what stops a binary reward and a
forgiving grader converging on a model that has forgotten how to write a
sentence.
"""

import json
import random
import time
from threading import Event

import torch

from llm import storage
from llm.config import RlConfig
from llm.corpus import bpe
from llm.gym.client import Client
from llm.model import checkpoint
from llm.orchestrate import train as train_pipeline
from llm.orchestrate import practice
from llm.train import loop, policy, rollout
from llm.train.progress import Checkpoint, Progress, ProgressCallback, log_progress

#: Status the gyms engine is told a batch reached, so an abandoned run does not
#: sit at `running` forever.
DONE = "done"
FAILED = "failed"


def run(
    cfg: RlConfig,
    on_progress: ProgressCallback | None = log_progress,
    cancel: Event | None = None,
    owner: str | None = None,
) -> dict:
    started = time.time()
    with Client(owner) as client:
        return _learn(cfg, client, on_progress, cancel, started)


def _draw(cfg: RlConfig, rng: random.Random, task_types) -> list[str]:
    """This iteration's task types, one per group, drawn by weight.

    A list with repetition rather than a set: the gyms engine assigns group *g*
    the type at `g % len(task_types)`, so a list of exactly `tasks_per_batch`
    entries *is* the assignment, and the weights are honoured by how often an id
    appears in it.
    """
    ids = [entry.id for entry in task_types]
    weights = [entry.weight for entry in task_types]
    return rng.choices(ids, weights=weights, k=cfg.tasks_per_batch)


def _frozen(weights, device):
    """The reference the KL is measured against: the model this run started from,
    loaded a second time and never updated.

    A second resident copy of the weights, which is the cost `SftConfig.
    measure_base_kl` documents for the diagnostic version of the same idea —
    mandatory here rather than optional.
    """
    reference, _ = checkpoint.load(weights, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    return reference


def _rollouts(cfg, client, model, tokenizer, iteration, rng, window, rope, generator, lessons=None):
    """One iteration's batch: every lane's gym run, rolled out and pooled.

    Pooled without weighting or rescaling. `RlConfig.lanes` says why that is
    sound across gyms as well as within one.
    """
    pooled: list[rollout.Rollout] = []
    runs: list[str] = []
    for gym_id, task_types in cfg.lanes():
        sampling = {}
        if gym_id == "general":
            # The service draws sources first; pre-drawing here would bias the mixture.
            drawn = [entry.id for entry in task_types]
            sampling = {"gym_weights": cfg.gym_weights,
                        "task_weights": {entry.id: entry.weight for entry in task_types}}
        else:
            drawn = lessons.draw(rng, cfg.tasks_per_batch) if lessons else _draw(cfg, rng, task_types)
        run_id, episodes = client.create_run(
            gym_id=gym_id,
            model_name=cfg.base_model,
            task_types=drawn,
            count=cfg.tasks_per_batch * cfg.group_size,
            seed_base=cfg.seed_base + (lessons.training_groups if lessons else iteration * cfg.tasks_per_batch),
            group_size=cfg.group_size,
            **sampling,
        )
        runs.append(run_id)
        rolled = rollout.run(
            model,
            tokenizer,
            client,
            episodes,
            cfg,
            window=window,
            rope=rope,
            max_rounds=cfg.max_rounds,
            generator=generator,
        )
        for roll in rolled:
            # The gym's own arithmetic: episode *i* is generated from
            # `seed_base + i // group_size`, so siblings are consecutive
            # episodes of one run. The run id is in the key because a pooled
            # batch holds several runs and every one of them counts from zero.
            roll.group = f"{run_id}:{roll.episode.index // cfg.group_size}"
        pooled.extend(rolled)
    return pooled, runs


def _optimizer(model, cfg: RlConfig):
    """AdamW, with the learned per-head temperatures excluded from weight decay
    for the same reason `loop.run` excludes them: they are a scale, and decaying
    a scale toward zero means decaying it toward temperature 1."""
    decayed, plain = [], []
    for name, parameter in model.named_parameters():
        (plain if name.endswith("log_temperature") else decayed).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": decayed, "weight_decay": cfg.weight_decay},
            {"params": plain, "weight_decay": 0.0},
        ],
        lr=cfg.lr,
        betas=(0.9, 0.95),
    )


def _source_scores(rollouts) -> dict:
    scores = {}
    for source in sorted({roll.episode.gym_id for roll in rollouts}):
        episodes = [roll for roll in rollouts if roll.episode.gym_id == source]
        graded = [roll for roll in episodes if policy.counts(roll)]
        scores[source] = {
            "episode_count": len(episodes), "graded": len(graded),
            "correct": sum(roll.score >= 1 for roll in graded),
            "score": sum(roll.score for roll in graded) / len(graded) if graded else None,
        }
    return scores


def _task_sequence(rollouts) -> list[str]:
    """Source gym for each task group, in the order the gym assigned them."""
    seen = set()
    sources = []
    for roll in sorted(rollouts, key=lambda item: item.episode.index):
        if roll.group not in seen:
            seen.add(roll.group)
            sources.append(roll.episode.gym_id)
    return sources


def _frame(cfg, iteration, batch, update, rollouts, started, lr) -> Progress:
    """One iteration's report.

    An RL run has no loss curve worth reading — the policy-gradient loss is an
    advantage-weighted sum whose sign and size say nothing on their own — so
    `loss` carries it for the chart's sake and everything a person actually reads
    is in `diag`. `signal` is the number to watch: the fraction of groups that
    had any variance in their returns at all. A run whose signal is collapsing is
    a run that is spending without learning, and no pass rate shows that.
    """
    live = [roll for roll in rollouts if policy.counts(roll)]
    rounds = sum(roll.rounds for roll in live) / len(live) if live else 0.0
    return Progress(
        step=iteration + 1,
        total_steps=cfg.iterations,
        epoch=1,
        total_epochs=1,
        chunk=iteration + 1,
        total_chunks=cfg.iterations,
        loss=update.loss,
        lr=lr,
        elapsed_secs=round(time.time() - started, 2),
        diag={
            # The x position of every point on the chart. A diag dict without it
            # is a sample the panel cannot place, so the stats render off the
            # latest one while the curves draw nothing — see `Diagnostics.step`,
            # whose whole job on the training path is this same field.
            "step": iteration + 1,
            "pass_rate": batch.pass_rate,
            "source_scores": _source_scores(rollouts),
            "task_sequence": _task_sequence(rollouts),
            "mean_return": batch.mean_return,
            "signal": batch.signal,
            "groups": batch.groups,
            "skipped_groups": batch.skipped,
            "kl": update.kl,
            "episodes": len(live),
            "abandoned": len(rollouts) - len(live),
            "episode_rounds": rounds,
            "policy_tokens": update.tokens,
        },
    )


def _iterate(cfg, client, model, reference, optimizer, state, iteration) -> Progress:
    """One iteration: a batch rolled out, scored, and one optimizer step."""
    rollouts, runs = _rollouts(
        cfg, client, model, state.tokenizer, iteration, state.rng,
        state.window, state.rope, state.generator,
        state.lessons,
    )
    batch = policy.score(rollouts)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    update = policy.accumulate(
        model, reference, rollouts, batch,
        beta=cfg.beta, micro_batch=cfg.micro_batch, rope=state.rope,
    )
    # A batch every one of whose groups was all-pass or all-fail has an exactly
    # zero gradient. Stepping on it would be an optimizer step made of nothing
    # but weight decay, which is a real change to the weights for no reason.
    if batch.advantages:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
    model.eval()

    for run_id in runs:
        client.set_status(run_id, DONE)
    if state.lessons:
        state.lessons.training_groups += cfg.tasks_per_batch
    return _frame(cfg, iteration, batch, update, rollouts, state.started, cfg.lr)


class _State:
    """What every iteration needs and none of them changes."""

    def __init__(self, cfg: RlConfig, model, tokenizer, device, started: float):
        self.tokenizer = tokenizer
        self.lessons = None
        self.started = started
        self.rng = random.Random(cfg.seed_base)
        # No sampler seed. Pinning the draw would make a group's G rollouts G
        # copies of one rollout, and a group with no variance in its returns has
        # no gradient in it — which is the whole reason the group exists.
        self.generator = torch.Generator(device=device)
        self.window = round(model.ctx_window * cfg.context_multiplier)
        self.rope = (
            model.extended_rope(self.window) if self.window > model.ctx_window else None
        )


def _learn(cfg: RlConfig, client: Client, on_progress, cancel, started) -> dict:
    weights = storage.weights_path(cfg.base_model)
    if not weights.is_file():
        raise FileNotFoundError(
            f"No base model {cfg.base_model!r} at {weights} — fine-tune one first"
        )

    device = loop.pick_device()
    model, arch = checkpoint.load(weights, device)
    reference = _frozen(weights, device)
    model.eval()
    print(
        f"[1/3] Base {cfg.base_model}: {model.parameter_count():,} parameters, "
        f"{arch['layers']}L/{arch['heads']}H/{arch['dimensions']}d on {device}, "
        f"and a frozen copy of it"
    )

    # The base model's own tokenizer, never the shared build: a rollout tokenized
    # differently from the weights it is drawn with is a rollout whose mask
    # points at the wrong tokens.
    tokenizer = bpe.load(storage.model_dir(cfg.base_model))
    state = _State(cfg, model, tokenizer, device, started)
    state.lessons = practice.prepare(cfg, client)
    optimizer = _optimizer(model, cfg)

    provenance = {"base_model": cfg.base_model, "architecture": arch}
    write = train_pipeline.CheckpointWriter(
        tokenizer=tokenizer,
        config=cfg.model_dump(exclude_unset=True),
        provenance=provenance,
        state_extra={"base_model": cfg.base_model},
    )

    if state.lessons:
        _exam(cfg, client, model, state, provenance, on_progress, cancel, 0, initial=True)

    frames = []
    for iteration in range(cfg.iterations):
        if cancel is not None and cancel.is_set():
            break
        frame = _iterate(cfg, client, model, reference, optimizer, state, iteration)
        if state.lessons:
            if (iteration + 1) % cfg.curriculum.check_every == 0:
                _exam(cfg, client, model, state, provenance, on_progress, cancel, iteration + 1)
            provenance["curriculum"] = state.lessons.snapshot()
            frame.diag["mastery"] = state.lessons.report()
            frame.diag["curriculum_events"] = list(state.lessons.events)
            write.state_extra["curriculum"] = provenance["curriculum"]
        frames.append(frame)
        if on_progress is not None:
            on_progress(frame)
            _checkpoint(cfg, write, model, optimizer, iteration, frame, on_progress)

    if state.lessons and not (cancel is not None and cancel.is_set()):
        _exam(cfg, client, model, state, provenance, on_progress, cancel, len(frames), final=True)
    return _saved(cfg, model, tokenizer, arch, frames, started, provenance)


def _exam(cfg, client, model, state, provenance, on_progress, cancel, step, *, initial=False, final=False):
    lessons = state.lessons

    def notify(task):
        phase = "Fixed test" if testing else "Placement" if initial else "Validation"
        print(f"{phase}: {task}")
        if on_progress:
            on_progress(Progress(step=step, total_steps=cfg.iterations, epoch=1,
                                 total_epochs=1, chunk=step, total_chunks=cfg.iterations,
                                 loss=0.0, lr=cfg.lr, elapsed_secs=time.time() - state.started,
                                 diag={"step": step, "practice_phase": f"{phase}: {task}",
                                       "mastery": lessons.report(),
                                       "curriculum_events": list(lessons.events)}))

    testing = False
    if not final:
        practice.validate(cfg, client, model, state, lessons, step, cancel, notify)
    if initial or final:
        testing = True
        result = practice.benchmark(cfg, client, model, state, lessons, cancel, notify)
        if result is not None:
            provenance["test_before" if initial else "test_after"] = result
    provenance["curriculum"] = lessons.snapshot()
    provenance["mastery"] = lessons.report()
    provenance["curriculum_events"] = list(lessons.events)
    if on_progress:
        on_progress(Progress(step=step, total_steps=cfg.iterations, epoch=1, total_epochs=1,
                             chunk=step, total_chunks=cfg.iterations, loss=0.0, lr=cfg.lr,
                             elapsed_secs=time.time() - state.started,
                             diag={"step": step, "practice_phase": "Practice" if not final else "Finished",
                                   **{key: provenance[key] for key in ("mastery", "curriculum_events", "test_before", "test_after") if key in provenance}}))


def _checkpoint(cfg, write, model, optimizer, iteration, frame, on_progress) -> None:
    """Every iteration is a checkpoint, and a checkpoint is a model.

    That sameness is what makes measuring progress need nothing new: a gym run
    against `<the run's name> #n`, at temperature 0, on the held-out seeds this
    run never drew. `checkpoint_target` 0 turns them off, exactly as it does for
    a training run, and is a disk budget — a `train_state.pt` is the weights
    again plus AdamW's two moments per parameter.
    """
    if cfg.checkpoint_target == 0:
        return
    on_progress(
        write(
            model,
            optimizer,
            step=iteration + 1,
            total_steps=cfg.iterations,
            epoch=1,
            stage=0,
            stage_name=None,
            loss=frame.loss,
        )
    )


def _saved(cfg, model, tokenizer, arch, frames, started, provenance) -> dict:
    directory = storage.ensure_dir(storage.model_dir(cfg.model_name))
    checkpoint.save(model, storage.weights_path(cfg.model_name))
    tokenizer.save(str(storage.tokenizer_file(directory)))

    last = frames[-1].diag if frames else {}
    result = {
        "model_name": cfg.model_name,
        **provenance,
        "iterations": len(frames),
        "pass_rate": last.get("pass_rate", 0.0),
        "source_scores": last.get("source_scores", {}),
        "source_history": [
            {"step": frame.diag["step"], "source_scores": frame.diag["source_scores"],
             "task_sequence": frame.diag["task_sequence"],
             "pass_rate": frame.diag["pass_rate"], "signal": frame.diag["signal"]}
            for frame in frames
        ],
        "signal": last.get("signal", 0.0),
        "elapsed_secs": round(time.time() - started, 2),
    }
    storage.meta_path(cfg.model_name).write_text(
        json.dumps({"config": cfg.model_dump(exclude_unset=True), "result": result}, indent=2),
        encoding="utf-8",
    )
    print(f"[3/3] Saved {cfg.model_name}: pass rate {result['pass_rate']:.2f}")
    return result
