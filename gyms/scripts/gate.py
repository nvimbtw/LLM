"""Does this model have anything a policy gradient could learn from?

The question RL is blocked on, and the reason it is a script rather than a
route: it is asked of a *set* of runs, not of one. A group is G rollouts of one
seed, and its advantage is how much better a rollout did than its siblings — so
a group whose returns are all identical contributes exactly zero gradient,
whether they all failed or all succeeded. What matters is not the mean score but
**how many groups have any variance at all**.

Runs are grouped by `(task_type, seed)` across every run sharing a `seed_base`,
which is what makes the gate cost no schema: eight ordinary 20-episode runs at
one seed base *are* twenty groups of eight, because episode *i* draws
`seed_base + i` in every one of them.

    .venv/bin/python -m scripts.gate --seed-base 12345

`GYMS_DATABASE_URL` is where it reads from, and the default is the *native*
database. A stack started with `docker compose up` keeps its Postgres on 5433
instead, which is the one mistake this makes easy — so a connection failure is
caught and answered with the URL it tried rather than a driver traceback.

Two things to get right when producing the runs, or the answer is meaningless:
**temperature above zero** (identical rollouts are one rollout counted G times)
and **no sampler seed** (`ChatConfig.seed` pins the draw, so eight runs would be
eight copies).
"""

import argparse
import collections
import os

from sqlalchemy import select
from sqlalchemy.exc import OperationalError

import store
from store.schema import EpisodeStatus, GymEpisode, GymRun, GymStep


def _rows(seed_base: int | None, model: str | None) -> list[tuple]:
    with store.session() as db:
        query = (
            select(
                GymEpisode.task_type,
                GymEpisode.seed,
                GymEpisode.score,
                GymEpisode.status,
                GymEpisode.reply,
                GymEpisode.reward_total,
                GymEpisode.id,
                GymRun.model_name,
            )
            .join(GymRun, GymRun.id == GymEpisode.run_id)
        )
        if seed_base is not None:
            query = query.where(GymRun.seed_base == seed_base)
        if model is not None:
            query = query.where(GymRun.model_name == model)
        episodes = list(db.execute(query))
        steps = collections.Counter(
            episode_id for (episode_id,) in db.execute(select(GymStep.episode_id))
        )
    return [(*row, steps[row.id]) for row in episodes]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-base", type=int, help="only runs created at this seed base")
    parser.add_argument("--model", help="only runs of this model id")
    args = parser.parse_args()

    store.configure(os.environ.get("GYMS_DATABASE_URL"))
    try:
        rows = _rows(args.seed_base, args.model)
    except OperationalError as e:
        raise SystemExit(
            f"cannot read the gyms database at {store.url()}\n"
            f"  {type(e.orig).__name__}: {e.orig}\n\n"
            "  If the stack is the one `docker compose up` started, its Postgres is\n"
            "  published on 5433 and this default is not it:\n\n"
            '    GYMS_DATABASE_URL="postgresql+psycopg://slate:slate@127.0.0.1:5433/slate_gyms" \\\n'
            "      .venv/bin/python -m scripts.gate"
        ) from None
    if not rows:
        raise SystemExit(
            "no episodes matched. `--seed-base` must be a value some run was "
            "created with — run without it to read every run in the database."
        )

    groups: dict[tuple[str, int], list[float]] = collections.defaultdict(list)
    never_answered = graded = 0
    steps_total = 0
    for task_type, seed, score, status, reply, _reward, _id, _model, steps in rows:
        steps_total += steps
        if status is not EpisodeStatus.graded:
            continue
        graded += 1
        groups[(task_type, seed)].append(score or 0.0)
        if reply and "tool-call" in reply:
            never_answered += 1

    passed = sum(1 for scores in groups.values() if any(s > 0 for s in scores))
    split = sum(1 for scores in groups.values() if 0 < sum(s > 0 for s in scores) < len(scores))
    sizes = collections.Counter(len(s) for s in groups.values())

    print(f"episodes          {len(rows)} ({graded} graded)")
    print(f"groups            {len(groups)}  sizes {dict(sorted(sizes.items()))}")
    print(f"steps per episode {steps_total / len(rows):.1f}")
    print(f"never answered    {never_answered}/{graded}  (reply still calling the shell)")
    print(f"episodes correct  {sum(sum(s) for s in groups.values()):.0f}/{graded}")
    print()
    print(f"pass@G            {passed}/{len(groups)} groups solved at least once")
    print(f"GROUPS WITH SIGNAL {split}/{len(groups)}  <- the gate")
    print()
    for task_type in sorted({t for t, _ in groups}):
        mine = {k: v for k, v in groups.items() if k[0] == task_type}
        solved = sum(1 for v in mine.values() if any(s > 0 for s in v))
        mixed = sum(1 for v in mine.values() if 0 < sum(s > 0 for s in v) < len(v))
        print(f"  {task_type:<12} {solved:>3}/{len(mine)} solved   {mixed:>3}/{len(mine)} with signal")
    print()
    if split:
        print(f"PASS — {split} groups carry a non-zero advantage. GRPO has something to learn from.")
    elif passed:
        print("BLOCKED — every group is all-pass or all-fail. Zero advantage everywhere.")
        print("  Widen the spread: raise temperature, or move the tier this model sits between.")
    else:
        print("BLOCKED — nothing was ever solved. There is nothing to reinforce and")
        print("  nothing to reject-sample. Behaviour cloning first; see")
        print("  docs/decisions/reinforcement-learning-in-the-engine.md.")


if __name__ == "__main__":
    main()
