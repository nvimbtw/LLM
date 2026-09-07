"""The adaptive pipeline, including the checkpoint seam and exam isolation."""

import json
from threading import Event

import pytest

from llm import storage
from llm.config import RlConfig
from llm.gym.client import Episode, Graded
from llm.model import checkpoint
from llm.orchestrate import practice, rl
from llm.train import policy, rollout
from llm.train.progress import Checkpoint
from tests.fixtures.tiny_env import TINY_MODEL

TASK = {"id": "math_addition_easy", "label": "Addition", "level": "easy", "family": "addition", "prerequisites": []}


class MathGym:
    def __init__(self):
        self.created = []
        self.statuses = []
        self.split = {}

    def catalogue(self):
        return [{"id": "math", "task_types": [TASK], "tools": []}]

    def create_run(self, **body):
        self.created.append(body)
        run_id = str(len(self.created))
        episodes = [Episode(id=f"{run_id}-{i}", index=i, gym_id="math", task_type=TASK["id"],
                            seed=body["seed_base"] + i // body["group_size"],
                            prompt="2 + 2?", system="Answer with a number.", tools=())
                    for i in range(body["count"])]
        for episode in episodes:
            self.split[episode.id] = body.get("split", "train")
        return run_id, episodes

    def finish(self, episode_id, reply):
        return Graded(score=1 if self.split[episode_id] != "train" else int(episode_id.rsplit("-", 1)[1]) % 2, answer="4")

    def step(self, *args):
        pytest.fail("a math episode called a tool")

    def set_status(self, run_id, status):
        self.statuses.append((run_id, status))

    def abandon(self, episode_id):
        pass


def config(**overrides):
    return RlConfig(**{"model_name": "math-trained", "base_model": TINY_MODEL,
                      "gym_id": "math", "task_types": [TASK["id"]], "iterations": 1,
                      "tasks_per_batch": 1, "group_size": 2, "max_tokens": 4,
                      "curriculum": {"check_every": 1, "validation_samples": 10, "test_samples": 1},
                      **overrides})


def test_only_training_rollouts_reach_backward_and_lessons_survive(monkeypatch):
    gym, frames, trained = MathGym(), [], []
    accumulate = policy.accumulate

    def checked(model, reference, rolls, batch, **kwargs):
        trained.extend(roll.episode.id for roll in rolls)
        assert all(gym.split[roll.episode.id] == "train" for roll in rolls)
        return accumulate(model, reference, rolls, batch, **kwargs)

    monkeypatch.setattr(policy, "accumulate", checked)
    result = rl._learn(config(), gym, frames.append, None, 0)
    assert len(trained) == 2
    assert result["mastery"][0]["status"] == "mastered"
    assert result["test_before"]["accuracy"] == result["test_after"]["accuracy"] == 1
    tests = [r for r in gym.created if r.get("split") == "test"]
    assert tests[0] == tests[1]
    saved = next(f for f in frames if isinstance(f, Checkpoint))
    meta = json.loads(storage.meta_path(saved.model_id).read_text())
    assert meta["result"]["curriculum"]["training_groups"] == 1
    extra = checkpoint.load_state(storage.train_state_path(saved.model_id)).extra
    assert extra["curriculum"] == meta["result"]["curriculum"]
    restored = practice.prepare(config(base_model=saved.model_id), gym)
    assert restored.status(TASK["id"]) == "mastered"
    assert restored.cycle == 2


def test_cancelled_placement_does_not_create_exams():
    gym = MathGym()
    cancel = Event()
    cancel.set()
    result = rl._learn(config(), gym, None, cancel, 0)
    assert not gym.created
    assert result["iterations"] == 0
    assert "test_after" not in result


def test_a_question_that_cannot_fit_is_refused(tokenizer):
    episode = Episode("1", 0, "math", TASK["id"], 0, "number " * 100, "", ())
    with pytest.raises(ValueError, match="does not fit"):
        rollout.start([episode], tokenizer, 32)
