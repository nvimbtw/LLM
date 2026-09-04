"""The loop closed: roll out, score, update, checkpoint, repeat.

Against a gym that lives in this process, because what is being checked is the
pipeline and not the sandbox — that a batch reaches an optimizer step, that the
weights actually move, that every iteration leaves a checkpoint laid out like a
model, and that a batch with no signal in it does *not* move the weights.

That last one is the test worth having. A run whose groups are all-pass or
all-fail has an exactly zero gradient, and an optimizer that stepped on it anyway
would be applying weight decay to a model nobody asked to shrink — a real change
to the weights, in a run that learned nothing, invisible in every frame.
"""

import json

import torch

from llm import storage
from llm.config import RlConfig
from llm.gym.client import Episode, Graded, Observed
from llm.orchestrate import rl
from llm.train.progress import Checkpoint
from tests.fixtures.tiny_env import TINY_MODEL

RUN_ID = "00000000-0000-4000-8000-000000000001"
MODEL_ID = "00000000-0000-4000-8000-0000000000aa"


class FakeGym:
    """One gym run per call, answering with a rendered observation and a score.

    `scores` is a cycle: the whole point of a group is that its members can
    differ, so a fake that answers the same thing every time is a fake that
    cannot exercise the interesting half of `policy.score`.
    """

    def __init__(self, scores=(1.0, 0.0)):
        self.scores = list(scores)
        self.created: list[dict] = []
        self.statuses: list[tuple[str, str]] = []
        self._graded = 0

    def create_run(self, **body):
        self.created.append(body)
        run = f"{RUN_ID[:-1]}{len(self.created)}"
        return run, [
            Episode(
                id=f"{run}-ep-{i}",
                index=i,
                gym_id=body["gym_id"],
                task_type=body["task_types"][i // body["group_size"]],
                seed=body["seed_base"] + i // body["group_size"],
                prompt=f"Which file is the odd one out? {i}",
                system="Use the bash tool to look around.",
            )
            for i in range(body["count"])
        ]

    def step(self, episode_id, tool, argument, reply):
        return Observed(observation="<|codeblock|>a.txt b.md<|/codeblock|>", reward=0.05, step_num=1)

    def finish(self, episode_id, reply):
        score = self.scores[self._graded % len(self.scores)]
        self._graded += 1
        return Graded(score=score, answer=reply[-16:])

    def close(self, episode_id):
        pass

    def set_status(self, run_id, status):
        self.statuses.append((run_id, status))


def _cfg(**overrides) -> RlConfig:
    return RlConfig(
        **{
            "model_name": MODEL_ID,
            "base_model": TINY_MODEL,
            "task_types": ["bash_ls_easy"],
            "group_size": 2,
            "tasks_per_batch": 1,
            "iterations": 2,
            "max_rounds": 2,
            "max_tokens": 8,
            **overrides,
        }
    )


def _weights(model_id: str) -> torch.Tensor:
    from llm.model import checkpoint

    model, _ = checkpoint.load(storage.weights_path(model_id), "cpu")
    return model.embedding.weight.detach().clone()


def _drive(cfg, gym):
    frames = []
    result = rl._learn(cfg, gym, frames.append, None, 0.0)
    return result, frames


def test_a_run_produces_a_model_and_a_checkpoint_per_iteration():
    cfg = _cfg()
    result, frames = _drive(cfg, FakeGym())

    assert result["iterations"] == 2
    checkpoints = [f for f in frames if isinstance(f, Checkpoint)]
    assert [c.index for c in checkpoints] == [1, 2]

    # A checkpoint is a model: same directory, same three files beside the state.
    for saved in checkpoints:
        directory = storage.model_dir(saved.model_id)
        assert storage.weights_path(saved.model_id).is_file()
        assert storage.tokenizer_file(directory).is_file()
        assert storage.meta_path(saved.model_id).is_file()
        assert saved.architecture["ctx_window"]

    assert storage.weights_path(cfg.model_name).is_file()
    meta = json.loads(storage.meta_path(cfg.model_name).read_text())
    assert meta["result"]["base_model"] == TINY_MODEL


def test_one_iteration_is_one_gym_run_and_it_is_not_an_evaluation():
    gym = FakeGym()
    cfg = _cfg()
    _drive(cfg, gym)

    assert len(gym.created) == cfg.iterations
    assert all(body["group_size"] == cfg.group_size for body in gym.created)
    # Seeds move by a whole batch per iteration, so no task is drawn twice.
    assert [body["seed_base"] for body in gym.created] == [0, cfg.tasks_per_batch]
    assert [status for _, status in gym.statuses] == [rl.DONE] * cfg.iterations


def test_the_holdout_seeds_are_never_drawn():
    """A seed that trained is a seed the model may have memorised the answer to,
    so measuring on one measures nothing."""
    gym = FakeGym()
    cfg = _cfg(iterations=3)
    _drive(cfg, gym)

    drawn = {
        body["seed_base"] + group
        for body in gym.created
        for group in range(cfg.tasks_per_batch)
    }
    assert cfg.holdout_base not in drawn


def test_a_batch_with_signal_moves_the_weights():
    cfg = _cfg(iterations=1, lr=1e-3)
    before = _weights(TINY_MODEL)
    _drive(cfg, FakeGym(scores=(1.0, 0.0)))
    assert not torch.equal(before, _weights(cfg.model_name))


def test_a_batch_with_no_signal_leaves_the_weights_alone():
    """Every group all-fail: the gradient is exactly zero, and stepping on it
    would be weight decay applied to a model nobody asked to shrink."""
    cfg = _cfg(iterations=1, lr=1e-3, weight_decay=0.1)
    before = _weights(TINY_MODEL)
    _, frames = _drive(cfg, FakeGym(scores=(0.0,)))

    reports = [f for f in frames if not isinstance(f, Checkpoint)]
    assert reports[0].diag["signal"] == 0.0
    assert reports[0].diag["skipped_groups"] == reports[0].diag["groups"]
    assert torch.equal(before, _weights(cfg.model_name))


def test_a_frame_carries_what_an_rl_run_is_read_by():
    _, frames = _drive(_cfg(iterations=1), FakeGym())
    diag = [f for f in frames if not isinstance(f, Checkpoint)][0].diag

    assert diag.keys() >= {"pass_rate", "signal", "groups", "skipped_groups", "kl", "episodes"}
    assert 0.0 <= diag["pass_rate"] <= 1.0
    assert diag["kl"] >= 0.0, "k3 is non-negative for every sample; a negative one is a sign flip"


def test_cancelling_stops_between_iterations():
    from threading import Event

    cancel = Event()
    cancel.set()
    frames = []
    result = rl._learn(_cfg(), FakeGym(), frames.append, cancel, 0.0)
    assert result["iterations"] == 0
