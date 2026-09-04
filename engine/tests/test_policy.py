"""What a batch of rollouts is worth, and what has no gradient in it.

Two things here are specifications rather than tests. The clip invariant is the
one that makes the dense shaping safe to train on at all, and the skipped-group
accounting is the number a run is read by — a batch whose groups are all-pass or
all-fail is spending without learning, and nothing else in the frame says so.
"""

import pytest

from llm.gym.client import Episode
from llm.train import policy
from llm.train.rollout import Rollout

GROUP = "run-1:0"


def _roll(index: int, *, score: float, shaping: float = 0.0, group: str = GROUP, stopped="answered"):
    episode = Episode(
        id=f"ep-{group}-{index}",
        index=index,
        gym_id="bash",
        task_type="bash_ls_easy",
        seed=1,
        prompt="p",
        system="s",
    )
    return Rollout(
        episode=episode,
        ids=[1, 2, 3],
        policy=[False, True, True],
        group=group,
        score=score,
        shaping=shaping,
        stopped=stopped,
    )


def test_shaping_can_never_outrank_an_answer():
    """The whole reason the clip is a constant and not an argument about
    magnitudes. A wrong episode that farmed every shaping term the gym pays must
    still lose to a correct one that ran nothing at all."""
    farmed = policy.episode_return(_roll(0, score=0.0, shaping=1_000.0))
    bare = policy.episode_return(_roll(1, score=1.0, shaping=-1_000.0))
    assert farmed < bare


def test_shaping_still_breaks_ties_among_equals():
    """What the clip must *not* do. A binary reward is silent between two failures
    and between two successes, and that is the gap the shaping exists to fill."""
    worked = policy.episode_return(_roll(0, score=0.0, shaping=0.1))
    idled = policy.episode_return(_roll(1, score=0.0, shaping=0.0))
    assert worked > idled


def test_a_group_that_all_failed_contributes_nothing():
    batch = policy.score([_roll(i, score=0.0) for i in range(4)])
    assert batch.groups == 1
    assert batch.skipped == 1
    assert batch.advantages == {}
    assert batch.signal == 0.0


def test_a_group_that_all_passed_contributes_nothing_either():
    """The half that is easy to forget: a solved task teaches as little as an
    unsolvable one, because an advantage is a *difference*."""
    batch = policy.score([_roll(i, score=1.0) for i in range(4)])
    assert batch.skipped == 1
    assert batch.advantages == {}


def test_a_split_group_advantages_the_winners():
    rolls = [_roll(0, score=1.0), _roll(1, score=1.0), _roll(2, score=0.0), _roll(3, score=0.0)]
    batch = policy.score(rolls)

    assert batch.skipped == 0
    assert batch.signal == 1.0
    assert batch.pass_rate == 0.5
    advantages = [batch.advantages[roll.episode.id] for roll in rolls]
    assert advantages[0] > 0 and advantages[1] > 0
    assert advantages[2] < 0 and advantages[3] < 0
    assert sum(advantages) == pytest.approx(0.0, abs=1e-5)


def test_groups_from_different_runs_are_not_one_group():
    """A pooled batch holds a gym run per gym and every one of them numbers its
    episodes from zero. Grouping on the index would make two gyms' episode 0 each
    other's siblings — a baseline over two different questions."""
    rolls = [
        _roll(0, score=1.0, group="run-1:0"),
        _roll(1, score=1.0, group="run-1:0"),
        _roll(0, score=0.0, group="run-2:0"),
        _roll(1, score=0.0, group="run-2:0"),
    ]
    batch = policy.score(rolls)

    assert batch.groups == 2
    # Each group is internally identical, so each is skipped. Pooled on the index
    # they would have looked like two groups with signal.
    assert batch.skipped == 2
    assert len({roll.episode.id for roll in rolls}) == 4


def test_an_abandoned_episode_is_absent_rather_than_a_zero():
    """It measures this side giving up, not the model. Counting it as a failure
    would hand the group a spread that the model did not produce — and, worse,
    reward every sibling for a sandbox that fell over."""
    rolls = [_roll(0, score=1.0), _roll(1, score=1.0), _roll(2, score=0.0, stopped="unreachable")]
    batch = policy.score(rolls)

    assert batch.skipped == 1
    assert batch.pass_rate == 1.0
    assert rolls[2].episode.id not in batch.advantages


def test_an_episode_that_ran_out_of_rounds_is_a_real_zero():
    """Unlike an abandoned one. It was graded on what it last said, and never
    answering is a thing the model did."""
    rolls = [_roll(0, score=1.0), _roll(1, score=0.0, stopped="terminal")]
    batch = policy.score(rolls)

    assert batch.skipped == 0
    assert batch.advantages[rolls[1].episode.id] < 0
