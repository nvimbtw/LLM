"""The gyms engine's whole HTTP surface.

The service owns three things and drives none of them: the tasks, the sandbox
they run in, and the record of what happened. **The episode loop lives in the
backend** (`backend/crates/api/src/turn.rs`) — see
`docs/decisions/gyms-are-driven-by-the-backend.md` for why — so every route here
is one move of a loop somebody else is running.

The ground truth never leaves this service while an episode is open. `/gym/runs`
answers with `Task.public()`, the step route answers with what the shell said,
and the answer only appears in the response to `finish`, which is the moment it
stops being able to help.
"""

import uuid
from typing import Annotated, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import store
from gyms.contracts import Task, UnsupportedAction
from gyms.registry import REGISTRY
from gyms.sandbox import SandboxError
from gyms.selection import Plan, resolve
from store import queries
from store.schema import EpisodeStatus, GymEpisode, GymRun, RunKind, RunStatus

router = APIRouter()

#: What one run may ask for. A ceiling because every episode seeds a filesystem
#: the sandbox holds until it expires, and the sandbox keeps 256 of them.
MAX_EPISODES = 256

#: What one demonstration request may ask for. These all share **one** workspace,
#: cleared between tasks, so the number costs sandbox time rather than sandbox
#: filesystems. A large ceiling makes a multi-thousand SFT corpus one request;
#: callers should still use smaller batches when they need progress visibility.
MAX_TRAJECTORIES = 5_000


Weight = Annotated[float, Field(gt=0, le=1_000_000, allow_inf_nan=False)]


class Selection(BaseModel):
    gym_id: str = "bash"
    task_types: list[str] | None = Field(default=None, min_length=1)
    gym_weights: dict[str, Weight] = Field(default_factory=dict)
    task_weights: dict[str, Weight] = Field(default_factory=dict)


class NewRun(Selection):
    model_name: str
    count: int = Field(ge=1, le=MAX_EPISODES)
    #: Absent means a fresh draw. Given, a run is reproducible: episode *i* is
    #: generated from `seed_base + i // group_size`.
    seed_base: int | None = None
    #: Rollouts per task. 1 is an evaluation — `count` distinct tasks, which is
    #: what a person watching wants. Above 1 the run is its own GRPO batch:
    #: `count // group_size` tasks, each attempted `group_size` times, because
    #: an advantage is measured against a *group's* own mean and a group of one
    #: has no mean to be better than.
    group_size: int = Field(default=1, ge=1, le=MAX_EPISODES)
    #: `rl` keeps the run off the board `GET /gym/runs` serves. A training run
    #: takes one of these per optimizer step, and they are batches rather than
    #: measurements — see `store.schema.RunKind`.
    kind: RunKind = RunKind.eval
    split: Literal["train", "validation", "test"] | None = None


class NewTrajectories(Selection):
    count: int = Field(ge=1, le=MAX_TRAJECTORIES)
    seed_base: int = 0


class NewStep(BaseModel):
    tool: str = "bash"
    command: str
    #: The assistant text of the round that made this call. Stored so the
    #: trajectory can show what the model was doing, not only what it ran.
    reply: str | None = None


class Finish(BaseModel):
    reply: str


class Status(BaseModel):
    status: RunStatus


@router.get("/gyms")
def gyms() -> list[dict]:
    """Which gyms exist and what they generate. The frontend's catalogue."""
    return REGISTRY.listing()


@router.post("/gym/runs", status_code=201)
def create_run(body: NewRun) -> dict:
    """Create a run, generate its episodes, and seed a workspace for each.

    Everything is done before answering, seeding included. That is the slow
    choice and the right one: a run whose sandboxes are prepared lazily fails in
    the middle of a rollout, where the failure is indistinguishable from a model
    that could not find anything.
    """
    split = body.split or ("train" if body.kind is RunKind.rl else "test")
    plan = _selection(body, split)

    # A trailing partial group is refused rather than truncated or allowed. It
    # would be a group whose baseline is drawn from fewer siblings than every
    # other — and at the limit a group of one, which contributes exactly zero
    # advantage and is the degenerate case `group_size` exists to prevent.
    if body.count % body.group_size:
        raise HTTPException(
            status_code=400,
            detail=f"count {body.count} is not a multiple of group_size {body.group_size}",
        )

    seed_base = body.seed_base if body.seed_base is not None else uuid.uuid4().int % 1_000_000
    seen: set[tuple[str, str]] = set()
    with store.session() as db:
        run = GymRun(
            gym_id=body.gym_id,
            model_name=body.model_name,
            task_types=list(plan.task_types),
            selection=plan.manifest,
            seed_base=seed_base,
            group_size=body.group_size,
            kind=body.kind,
            episode_count=body.count,
        )
        db.add(run)
        db.flush()
        episodes = []
        for group in range(body.count // body.group_size):
            task = REGISTRY.generate(plan.choose(seed_base, group), seed_base + group, split)
            if body.group_size == 1 and task.problem_key is not None:
                task = _distinct(task, split, seen)
            for attempt in range(body.group_size):
                episodes.append(_prepare(db, run, group * body.group_size + attempt, task))
        return {**_run_json(run, db), "episodes": [_episode_json(e, db) for e in episodes]}


def _selection(body: Selection, split: str) -> Plan:
    if REGISTRY.gym(body.gym_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown gym {body.gym_id!r}")
    try:
        return resolve(REGISTRY, body.gym_id, body.task_types,
                       body.gym_weights, body.task_weights, split)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


def _prepare(db, run: GymRun, index: int, task: Task) -> GymEpisode:
    # Each sibling gets a separate workspace, even though its task is identical.
    episode = GymEpisode(
        run_id=run.id, index=index, gym_id=task.gym_id,
        task_type=task.task_type, seed=task.seed, task_config=task.stored(),
    )
    db.add(episode)
    db.flush()
    try:
        REGISTRY.provider(task.gym_id).prepare(str(episode.id), task)
    except SandboxError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return episode


def _distinct(task: Task, split: str, seen: set[tuple[str, str]]) -> Task:
    # A validation count means different problems, not repeated samples of one.
    max_attempts = 10_000
    for offset in range(max_attempts):
        candidate = REGISTRY.generate(task.task_type, task.seed + offset, split, task.generator_version)
        key = (candidate.task_type, candidate.problem_key)
        if key not in seen:
            seen.add(key)
            return candidate
    raise HTTPException(status_code=400, detail="requested more distinct problems than this task type can supply")


@router.get("/gym/runs")
def list_runs(kind: RunKind = RunKind.eval) -> list[dict]:
    with store.session() as db:
        return [_run_json(run, db) for run in queries.runs(db, kind=kind)]


@router.get("/gym/runs/{run_id}")
def get_run(run_id: uuid.UUID) -> dict:
    with store.session() as db:
        run = queries.run(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")
        episodes = queries.episodes_of(db, run.id)
        return {**_run_json(run, db), "episodes": [_episode_json(e, db) for e in episodes]}


@router.get("/gym/episodes/{episode_id}")
def get_episode(episode_id: uuid.UUID) -> dict:
    with store.session() as db:
        episode = _episode_or_404(db, episode_id)
        return _episode_json(episode, db, steps=True)


@router.post("/gym/trajectories")
def trajectories(body: NewTrajectories) -> dict:
    """Expert demonstrations: the trajectory a model *should* have produced.

    Nothing here is a run and nothing is stored. A demonstration is training
    data, not a measurement, and writing it to `gym_runs` would put a perfect
    score in every board and league table this service feeds.

    **The observations come back raw.** What the model is shown is rendered by
    `tools::observation` on the Rust side, which is the whole of
    `docs/decisions/gyms-are-driven-by-the-backend.md`: a Python copy of that
    layout would train a model for a shell it is never served. So this answers
    with what the shell said and the backend assembles the conversation.
    """
    plan = _selection(body, "train")
    for task_type in plan.task_types:
        if not REGISTRY.provider(REGISTRY.gym_of(task_type)).demonstrations:
            raise HTTPException(status_code=400, detail=f"{task_type!r} has no demonstrations")
    session = f"expert-{uuid.uuid4()}"
    built = []
    try:
        for index in range(body.count):
            task = REGISTRY.generate(plan.choose(body.seed_base, index), body.seed_base + index)
            built.append(REGISTRY.provider(task.gym_id).demonstrate(session, task))
    except SandboxError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return {"count": len(built), "selection": plan.manifest, "trajectories": built}


@router.post("/gym/episodes/{episode_id}/step")
def step(episode_id: uuid.UUID, body: NewStep) -> dict:
    """Run one command in the episode's workspace and record it."""
    with store.session() as db:
        episode = _episode_or_404(db, episode_id)
        if episode.status in (EpisodeStatus.graded, EpisodeStatus.closed):
            raise HTTPException(status_code=409, detail="episode is closed")
        task = _task(episode)
        if body.tool not in task.tools:
            raise HTTPException(status_code=400, detail=f"this episode does not support {body.tool!r}")
        step_num = queries.next_step_num(db, episode.id)
        try:
            ran = REGISTRY.provider(episode.gym_id).step(
                str(episode.id), task, body.tool, body.command, step_num,
                queries.step_texts(db, episode.id),
            )
        except UnsupportedAction as e:
            raise HTTPException(status_code=400, detail=str(e)) from e
        except SandboxError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        reward = ran.reward
        queries.record_step(
            db,
            episode,
            reply=body.reply,
            action=body.command,
            stdout=ran.stdout,
            stderr=ran.stderr,
            exit_code=ran.exit_code,
            timed_out=ran.timed_out,
            reward=reward,
        )
        return {
            "step_num": step_num,
            "stdout": ran.stdout,
            "stderr": ran.stderr,
            "exit_code": ran.exit_code,
            "timed_out": ran.timed_out,
            "reward": reward,
            "reward_total": episode.reward_total,
        }


@router.post("/gym/episodes/{episode_id}/finish")
def finish(episode_id: uuid.UUID, body: Finish) -> dict:
    """Grade the reply that ended the episode.

    Takes the whole reply rather than an extracted answer: extraction is part of
    grading (`gyms.bash.grade`), and a caller that did it first would be a second
    grader nobody versions.
    """
    with store.session() as db:
        episode = _episode_or_404(db, episode_id)
        if episode.status in (EpisodeStatus.graded, EpisodeStatus.closed):
            raise HTTPException(status_code=409, detail="episode is already graded")
        ground_truth = episode.task_config.get("answer", "")
        try:
            score, answer = REGISTRY.provider(episode.gym_id).finish(
                str(episode.id), _task(episode), body.reply,
            )
        except SandboxError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        queries.grade_episode(db, episode, reply=body.reply, answer=answer, score=score)
        return {
            "episode_id": str(episode.id),
            "score": score,
            "correct": score >= 1.0,
            "answer": answer,
            "ground_truth": ground_truth,
            "reward_total": episode.reward_total,
        }


@router.post("/gym/episodes/{episode_id}/close")
def close(episode_id: uuid.UUID) -> dict:
    """Abandon an episode: it ran out of rounds, or its run went away."""
    with store.session() as db:
        episode = _episode_or_404(db, episode_id)
        queries.close_episode(db, episode)
        return {"episode_id": str(episode.id), "status": episode.status.value}


@router.post("/gym/runs/{run_id}/status")
def set_status(run_id: uuid.UUID, body: Status) -> dict:
    """What the driver says is happening to the run it is holding."""
    with store.session() as db:
        run = queries.run(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")
        queries.set_run_status(db, run, body.status)
        return _run_json(run, db)


def _episode_or_404(db, episode_id: uuid.UUID) -> GymEpisode:
    episode = queries.episode(db, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="no such episode")
    return episode


def _run_json(run: GymRun, db) -> dict:
    return {
        "id": str(run.id),
        "gym_id": run.gym_id,
        "model_name": run.model_name,
        "task_types": run.task_types,
        "selection": run.selection,
        "source_scores": queries.source_scores(db, run),
        "seed_base": run.seed_base,
        "group_size": run.group_size,
        "kind": run.kind.value,
        "episode_count": run.episode_count,
        "status": run.status.value,
        "score": run.score,
        "graded": run.graded,
        "correct": run.correct,
        "created_at": run.created_at.isoformat(),
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def _episode_json(episode: GymEpisode, db, *, steps: bool = False) -> dict:
    """One episode. The ground truth is included only once it can no longer
    help — a graded episode is one whose answer is already fixed."""
    task = _task(episode)
    body = {
        **task.public(),
        "id": str(episode.id),
        "run_id": str(episode.run_id),
        "index": episode.index,
        "status": episode.status.value,
        "answer": episode.answer,
        "score": episode.score,
        "reward_total": episode.reward_total,
        "created_at": episode.created_at.isoformat(),
        "closed_at": episode.closed_at.isoformat() if episode.closed_at else None,
    }
    if episode.status is EpisodeStatus.graded:
        body["ground_truth"] = task.private.get("answer", "")
        body["reply"] = episode.reply
    if steps:
        body["steps"] = [
            {
                "step_num": s.step_num,
                "reply": s.reply,
                "action": s.action,
                "stdout": s.stdout,
                "stderr": s.stderr,
                "exit_code": s.exit_code,
                "timed_out": s.timed_out,
                "reward": s.reward,
            }
            for s in queries.steps_of(db, episode.id)
        ]
    return body


def _task(episode: GymEpisode) -> Task:
    return Task.restore(episode.gym_id, episode.task_config, REGISTRY.provider(episode.gym_id).tools)
