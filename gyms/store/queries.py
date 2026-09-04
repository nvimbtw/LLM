"""Reads and writes over the gyms tables.

Everything the routes do to the database is spelled here, so a route is the
shape of a request and not a place where a query lives. The aggregates on a run
are recomputed from its episodes rather than incremented in place: an episode
that is graded twice — a retried step, a driver that reconnected — must not move
the run's score twice.
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from store.schema import EpisodeStatus, GymEpisode, GymRun, GymStep, RunKind, RunStatus


def run(db: Session, run_id: uuid.UUID) -> GymRun | None:
    return db.get(GymRun, run_id)


def runs(db: Session, limit: int = 50, kind: RunKind = RunKind.eval) -> list[GymRun]:
    """The board. Filtered by kind, because a training run's iterations would
    otherwise bury every evaluation anybody made — there is one of them per
    optimizer step."""
    return list(
        db.scalars(
            select(GymRun)
            .where(GymRun.kind == kind)
            .order_by(GymRun.created_at.desc())
            .limit(limit)
        )
    )


def episode(db: Session, episode_id: uuid.UUID) -> GymEpisode | None:
    return db.get(GymEpisode, episode_id)


def episodes_of(db: Session, run_id: uuid.UUID) -> list[GymEpisode]:
    return list(
        db.scalars(select(GymEpisode).where(GymEpisode.run_id == run_id).order_by(GymEpisode.index))
    )


def steps_of(db: Session, episode_id: uuid.UUID) -> list[GymStep]:
    return list(
        db.scalars(
            select(GymStep).where(GymStep.episode_id == episode_id).order_by(GymStep.step_num)
        )
    )


def next_step_num(db: Session, episode_id: uuid.UUID) -> int:
    """The number the next step takes. Read rather than counted so a gap left by
    a failed write cannot make two steps claim the same position."""
    highest = db.scalar(
        select(func.max(GymStep.step_num)).where(GymStep.episode_id == episode_id)
    )
    return 0 if highest is None else highest + 1


def step_texts(db: Session, episode_id: uuid.UUID) -> list[tuple[str, str]]:
    """Every recorded `(command, output)` of an episode, oldest first.

    Read back rather than tracked in a column because the once-an-episode reward
    terms are derived from it (`gyms.bash.rewards.Earned.over`): the reward
    function is allowed to change what it pays for, and a stored flag written by
    an older version of it would keep paying the old rule forever.
    """
    return [
        (action or "", stdout or "")
        for action, stdout in db.execute(
            select(GymStep.action, GymStep.stdout)
            .where(GymStep.episode_id == episode_id)
            .order_by(GymStep.step_num)
        )
    ]


def record_step(
    db: Session,
    episode: GymEpisode,
    *,
    reply: str | None,
    action: str,
    stdout: str,
    stderr: str,
    exit_code: int,
    timed_out: bool,
    reward: float,
) -> GymStep:
    step = GymStep(
        episode_id=episode.id,
        step_num=next_step_num(db, episode.id),
        reply=reply,
        action=action,
        stdout=stdout,
        stderr=stderr,
        exit_code=exit_code,
        timed_out=timed_out,
        reward=reward,
    )
    db.add(step)
    episode.status = EpisodeStatus.running
    episode.reward_total = round(episode.reward_total + reward, 4)
    db.flush()
    return step


def grade_episode(db: Session, episode: GymEpisode, *, reply: str, answer: str, score: float) -> None:
    episode.reply = reply
    episode.answer = answer
    episode.score = score
    episode.status = EpisodeStatus.graded
    episode.closed_at = datetime.now(timezone.utc)
    db.flush()
    _resettle(db, episode.run_id)


def close_episode(db: Session, episode: GymEpisode) -> None:
    """Give up on an episode without an answer. Not a zero: an episode that was
    never answered is absent from the run's mean rather than dragging it down,
    because the reason is usually this side and not the model."""
    if episode.status is EpisodeStatus.graded:
        return
    episode.status = EpisodeStatus.closed
    episode.closed_at = datetime.now(timezone.utc)
    db.flush()
    _resettle(db, episode.run_id)


def set_run_status(db: Session, run: GymRun, status: RunStatus) -> None:
    run.status = status
    if status in (RunStatus.done, RunStatus.abandoned, RunStatus.failed):
        run.finished_at = datetime.now(timezone.utc)
    db.flush()


def _resettle(db: Session, run_id: uuid.UUID) -> None:
    """Recompute a run's aggregates from its graded episodes."""
    graded = list(
        db.scalars(
            select(GymEpisode).where(
                GymEpisode.run_id == run_id, GymEpisode.status == EpisodeStatus.graded
            )
        )
    )
    run = db.get(GymRun, run_id)
    if run is None:
        return
    run.graded = len(graded)
    run.correct = sum(1 for e in graded if (e.score or 0.0) >= 1.0)
    run.score = round(sum(e.score or 0.0 for e in graded) / len(graded), 4) if graded else None
    db.flush()
