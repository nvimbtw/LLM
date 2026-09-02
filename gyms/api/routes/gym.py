"""The gyms service's whole HTTP surface.

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

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

import store
from gyms.bash import expert
from gyms.bash.rewards import Earned, Step, step_reward
from gyms.registry import REGISTRY
from gyms.sandbox import WORKSPACE, SandboxError, clear, seed
from gyms.sandbox import run as run_command
from store import queries
from store.schema import EpisodeStatus, GymEpisode, GymRun, RunStatus

router = APIRouter()

#: What one run may ask for. A ceiling because every episode seeds a filesystem
#: the sandbox holds until it expires, and the sandbox keeps 64 of them.
MAX_EPISODES = 32

#: What one demonstration request may ask for. Far above `MAX_EPISODES` and for
#: a reason that is not generosity: these all share **one** workspace, cleared
#: between tasks, so the number costs sandbox time rather than sandbox
#: filesystems. Behaviour cloning wants thousands, and a caller pages.
MAX_TRAJECTORIES = 200


class NewRun(BaseModel):
    gym_id: str = "bash"
    model_name: str
    task_types: list[str] = Field(min_length=1)
    count: int = Field(ge=1, le=MAX_EPISODES)
    #: Absent means a fresh draw. Given, a run is reproducible: episode *i* is
    #: generated from `seed_base + i`.
    seed_base: int | None = None


class NewTrajectories(BaseModel):
    task_types: list[str] = Field(min_length=1)
    count: int = Field(ge=1, le=MAX_TRAJECTORIES)
    seed_base: int = 0


class NewStep(BaseModel):
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
    if REGISTRY.gym(body.gym_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown gym {body.gym_id!r}")
    for task_type in body.task_types:
        if REGISTRY.gym_of(task_type) != body.gym_id:
            raise HTTPException(
                status_code=400, detail=f"{task_type!r} is not a task type of {body.gym_id!r}"
            )

    seed_base = body.seed_base if body.seed_base is not None else uuid.uuid4().int % 1_000_000
    with store.session() as db:
        run = GymRun(
            gym_id=body.gym_id,
            model_name=body.model_name,
            task_types=list(body.task_types),
            seed_base=seed_base,
            episode_count=body.count,
        )
        db.add(run)
        db.flush()
        episodes = [
            _prepare(db, run, index, body.task_types[index % len(body.task_types)])
            for index in range(body.count)
        ]
        return {**_run_json(run), "episodes": [_episode_json(e, db) for e in episodes]}


def _prepare(db, run: GymRun, index: int, task_type: str) -> GymEpisode:
    """One episode: generated, stored, and its workspace filled.

    The row is written before the sandbox is touched so the episode id exists to
    key the filesystem by; a seeding failure then rolls the whole run back with
    it, which is why this raises rather than marking the episode failed.
    """
    task = REGISTRY.generate(task_type, run.seed_base + index)
    episode = GymEpisode(
        run_id=run.id,
        index=index,
        gym_id=run.gym_id,
        task_type=task_type,
        seed=task.seed,
        task_config=task.stored(),
    )
    db.add(episode)
    db.flush()
    try:
        seed(str(episode.id), WORKSPACE, task.files)
    except SandboxError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    return episode


@router.get("/gym/runs")
def list_runs() -> list[dict]:
    with store.session() as db:
        return [_run_json(run) for run in queries.runs(db)]


@router.get("/gym/runs/{run_id}")
def get_run(run_id: uuid.UUID) -> dict:
    with store.session() as db:
        run = queries.run(db, run_id)
        if run is None:
            raise HTTPException(status_code=404, detail="no such run")
        episodes = queries.episodes_of(db, run.id)
        return {**_run_json(run), "episodes": [_episode_json(e, db) for e in episodes]}


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
    unknown = [t for t in body.task_types if REGISTRY.task_type(t) is None]
    if unknown:
        raise HTTPException(status_code=400, detail=f"unknown task types: {unknown}")

    # One session for the whole batch, cleared between tasks — see
    # `sandbox.clear`. An episode may never do this; a demonstration must.
    session = f"expert-{uuid.uuid4()}"
    built = []
    try:
        for index in range(body.count):
            task_type = body.task_types[index % len(body.task_types)]
            task = REGISTRY.generate(task_type, body.seed_base + index)
            clear(session)
            seed(session, WORKSPACE, task.files)
            built.append(_demonstrate(session, task))
    except SandboxError as e:
        raise HTTPException(status_code=502, detail=str(e)) from e
    finally:
        clear(session)
    return {"count": len(built), "trajectories": built}


def _demonstrate(session: str, task) -> dict:
    """One task played through the shell by `gyms.bash.expert`."""
    moves = []
    for move in expert.solve(task).moves:
        ran = run_command(session, move.command)
        moves.append({
            "reply": move.reply,
            "command": move.command,
            "stdout": ran.stdout,
            "stderr": ran.stderr,
            "exit_code": ran.exit_code,
            "timed_out": ran.timed_out,
        })
    return {
        "task_type": task.task_type,
        "seed": task.seed,
        "system": task.system,
        "prompt": task.prompt,
        "moves": moves,
        "answer": task.answer,
    }


@router.post("/gym/episodes/{episode_id}/step")
def step(episode_id: uuid.UUID, body: NewStep) -> dict:
    """Run one command in the episode's workspace and record it."""
    with store.session() as db:
        episode = _episode_or_404(db, episode_id)
        if episode.status in (EpisodeStatus.graded, EpisodeStatus.closed):
            raise HTTPException(status_code=409, detail="episode is closed")
        try:
            ran = run_command(str(episode.id), body.command)
        except SandboxError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
        step_num = queries.next_step_num(db, episode.id)
        target = episode.task_config.get("target", "")
        evidence = episode.task_config.get("evidence", "")
        reward = step_reward(
            Step(body.command, ran.stdout, ran.stderr, ran.exit_code, ran.timed_out),
            step_num,
            target,
            evidence,
            Earned.over(queries.step_texts(db, episode.id), target, evidence),
        )
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
        if episode.status is EpisodeStatus.graded:
            raise HTTPException(status_code=409, detail="episode is already graded")
        ground_truth = episode.task_config.get("answer", "")
        score, answer = REGISTRY.grade(episode.task_type, body.reply, ground_truth)
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
        return _run_json(run)


def _episode_or_404(db, episode_id: uuid.UUID) -> GymEpisode:
    episode = queries.episode(db, episode_id)
    if episode is None:
        raise HTTPException(status_code=404, detail="no such episode")
    return episode


def _run_json(run: GymRun) -> dict:
    return {
        "id": str(run.id),
        "gym_id": run.gym_id,
        "model_name": run.model_name,
        "task_types": run.task_types,
        "seed_base": run.seed_base,
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
    task = episode.task_config
    body = {
        "id": str(episode.id),
        "run_id": str(episode.run_id),
        "index": episode.index,
        "gym_id": episode.gym_id,
        "task_type": episode.task_type,
        "seed": episode.seed,
        "prompt": task.get("prompt", ""),
        "system": task.get("system", ""),
        "status": episode.status.value,
        "answer": episode.answer,
        "score": episode.score,
        "reward_total": episode.reward_total,
        "created_at": episode.created_at.isoformat(),
        "closed_at": episode.closed_at.isoformat() if episode.closed_at else None,
    }
    if episode.status is EpisodeStatus.graded:
        body["ground_truth"] = task.get("answer", "")
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
