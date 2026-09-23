"""The trainer forwards a candidate pool, preserving per-episode capabilities."""

import random
from dataclasses import replace

import pytest

from llm.config import RlConfig
from llm.gym.client import Client, Episode
from llm.orchestrate import rl
from llm.train import rollout
from tests.test_rollout import FakeGym, loaded, cfg


def test_general_sampling_is_delegated_once_and_groups_keep_their_identity(monkeypatch):
    config = RlConfig(model_name="out", base_model="base", gym_id="general",
                      task_types=[{"id": "bash_ls_easy", "weight": 3}, "math_addition_easy"],
                      gym_weights={"math": 2}, tasks_per_batch=2, group_size=2)
    requests = []

    class Gym:
        def create_run(self, **body):
            requests.append(body)
            return "mixed", [
                Episode(str(i), i, "bash" if i < 2 else "math", config.task_types[i // 2].id,
                        3 + i // 2, "Question", "System", ("bash",) if i < 2 else ())
                for i in range(4)
            ]

    monkeypatch.setattr(rollout, "run", lambda model, tokenizer, client, episodes, *args, **kwargs:
                        [rollout.Rollout(episode=e, ids=[], policy=[]) for e in episodes])
    batch, runs = rl._rollouts(config, Gym(), None, None, 3, random.Random(0), 128, None, None)
    assert runs == ["mixed"]
    assert requests[0]["task_types"] == ["bash_ls_easy", "math_addition_easy"]
    assert requests[0]["task_weights"] == {"bash_ls_easy": 3, "math_addition_easy": 1}
    assert requests[0]["gym_weights"] == {"math": 2}
    assert requests[0]["seed_base"] == 6
    assert [r.group for r in batch] == ["mixed:0", "mixed:0", "mixed:1", "mixed:1"]
    assert [r.episode.tools for r in batch] == [("bash",), ("bash",), (), ()]


def test_client_transmits_weights_and_preserves_source_tools(monkeypatch):
    seen = []
    with Client("owner", url="http://unused.invalid", service_secret="test") as client:
        def request(method, path, **kwargs):
            seen.append(kwargs["json"])
            return {"id": "run", "episodes": [{"id": "math-episode", "index": 0,
                "gym_id": "math", "task_type": "math_addition_easy", "seed": 1, "tools": []}]}, 201
        monkeypatch.setattr(client, "request", request)
        _, episodes = client.create_run(gym_id="general", model_name="base",
            task_types=["math_addition_easy"], count=2, seed_base=1, group_size=2,
            gym_weights={"math": 2}, task_weights={"math_addition_easy": 3})
    assert seen[0]["gym_weights"] == {"math": 2}
    assert seen[0]["task_weights"] == {"math_addition_easy": 3}
    assert episodes[0].gym_id == "math" and episodes[0].tools == ()


def test_source_scores_exclude_ungraded_attempts_without_hiding_them():
    episode = Episode("ep", 0, "math", "math_addition_easy", 0, "1+1?", "Answer.", ())
    rolls = [
        rollout.Rollout(episode=episode, ids=[], policy=[], score=1, stopped="answered"),
        rollout.Rollout(episode=episode, ids=[], policy=[], score=0, stopped="ungraded"),
        rollout.Rollout(episode=replace(episode, gym_id="bash"), ids=[], policy=[], score=0, stopped="unreachable"),
    ]
    scores = rl._source_scores(rolls)
    assert scores["math"] == {"episode_count": 2, "graded": 1, "correct": 1, "score": 1}
    assert scores["bash"] == {"episode_count": 1, "graded": 0, "correct": 0, "score": None}


def test_task_sequence_keeps_group_order_without_repeating_siblings():
    episode = Episode("ep", 0, "math", "math_addition_easy", 0, "Question", "System", ())
    rolls = [
        rollout.Rollout(episode=replace(episode, index=index, gym_id=source),
                        ids=[], policy=[], group=group)
        for index, source, group in [(0, "math", "a"), (1, "math", "a"),
                                     (2, "bash", "b"), (3, "bash", "b"),
                                     (4, "math", "c")]
    ]
    assert rl._task_sequence(rolls) == ["math", "bash", "math"]


def test_a_mixed_rollout_masks_math_tools_on_its_first_round(loaded, cfg, monkeypatch):
    model, tokenizer = loaded
    gym = FakeGym()
    math = Episode("math", 0, "math", "math_addition_easy", 0, "1+1?", "Answer.", ())
    bash = replace(math, id="bash", index=1, gym_id="bash", task_type="bash_ls_easy", tools=("bash",))
    generate = rollout._generate
    masks = []

    def capture(model, rules, live, state, *args):
        masks.append([(roll.episode.id, state.calls[i].budget) for i, roll in enumerate(live)])
        return generate(model, rules, live, state, *args)

    # Inspect the decoder budget while still exercising the real batched sampler.
    monkeypatch.setattr(rollout, "_generate", capture)
    batch = rollout.run(model, tokenizer, gym, [math, bash], cfg,
                        window=model.ctx_window, max_rounds=2)
    assert masks[0][0] == ("math", 0)
    assert masks[0][1][1] > 0
    assert not any(step[0] == "math" for step in gym.stepped)
    assert all(not roll.live for roll in batch)


@pytest.mark.parametrize("extra", [{"gym_weights": {"math": 0}}, {"gym_weights": {"math": float("inf")}},
    {"task_types": ["math_addition_easy", "math_addition_easy"]}])
def test_invalid_general_weights_or_duplicate_candidates_are_refused(extra):
    with pytest.raises(ValueError):
        RlConfig(**{"model_name": "out", "base_model": "base", "gym_id": "general",
                    "task_types": ["math_addition_easy"], **extra})
