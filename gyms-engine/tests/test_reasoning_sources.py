"""Independent checks for generated state and ordering questions."""

import itertools
import re

import pytest

from gyms.contracts import Task
from gyms.registry import REGISTRY
from gyms.text_runtime import extract


LEVELS = ("easy", "medium", "hard")
SPLITS = ("train", "validation", "test")


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("level", LEVELS)
def test_state_histories_have_the_simulated_answer(level, split):
    for seed in range(80):
        task = REGISTRY.generate(f"state_tracking_{level}", seed, split)
        holders = dict(task.private["initial"])
        target = task.private["target"]
        other = next((item for item in holders if item != target), None)
        for event in task.private["events"]:
            transfer = re.fullmatch(r"([A-Za-z]+) gives the ([a-z]+) to ([A-Za-z]+)\.", event)
            if transfer:
                sender, item, receiver = transfer.groups()
                assert holders[item] == sender and receiver != sender
                holders[item] = receiver
            else:
                assert event == f"The holder of the {target} swaps objects with the holder of the {other}."
                assert holders[target] != holders[other]
                holders[target], holders[other] = holders[other], holders[target]
        assert holders[task.private["target"]] == task.private["answer"]
        assert task == REGISTRY.generate(task.task_type, task.seed, split, task.generator_version)
        assert Task.restore(task.gym_id, task.stored(), ()) == task
        assert len(task.prompt) < 500
        assert task.private["answer"] not in task.public().keys()


def _satisfies(order, clue):
    kind, left, right = clue
    if kind == "not_first":
        return order[0] != left
    if kind == "not_last":
        return order[-1] != left
    if kind == "before":
        return order.index(left) < order.index(right)
    assert kind == "next"
    return order.index(left) + 1 == order.index(right)


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("level", LEVELS)
def test_logic_clues_have_exactly_one_solution(level, split):
    for seed in range(80):
        task = REGISTRY.generate(f"constraint_logic_{level}", seed, split)
        people = task.private["people"]
        solutions = [order for order in itertools.permutations(people)
                     if all(_satisfies(order, clue) for clue in task.private["clues"])]
        assert solutions == [tuple(people)]
        assert solutions[0][task.private["queried"]] == task.private["answer"]
        assert task == REGISTRY.generate(task.task_type, task.seed, split, task.generator_version)
        assert Task.restore(task.gym_id, task.stored(), ()) == task
        assert len(task.prompt) < 500


@pytest.mark.parametrize("gym", ("state_tracking", "constraint_logic"))
async def test_sources_use_existing_routes_without_tools(gym, a_run, client, sandbox):
    sandbox.fail_writes = True
    run = await a_run(gym_id=gym, task_types=[f"{gym}_medium"], count=4,
                      group_size=2, seed_base=11)
    assert sandbox.commands == []
    first, second = run["episodes"][:2]
    assert first["prompt"] == second["prompt"]
    assert first["tools"] == [] and "ground_truth" not in first
    task = REGISTRY.generate(first["task_type"], first["seed"], first["split"])
    refused = await client.post(f"/gym/episodes/{first['id']}/step", json={"command": "ls"})
    assert refused.status_code == 400
    correct = await client.post(f"/gym/episodes/{first['id']}/finish",
                                json={"reply": task.private["solution"]})
    wrong = await client.post(f"/gym/episodes/{second['id']}/finish", json={"reply": "Answer: Nobody"})
    assert correct.json()["score"] == 1 and wrong.json()["score"] == 0
    assert sandbox.commands == []


@pytest.mark.parametrize("gym", ("state_tracking", "constraint_logic"))
async def test_generated_demonstrations_are_graduable(gym, client, sandbox):
    sandbox.fail_writes = True
    response = await client.post("/gym/trajectories", json={
        "gym_id": gym, "task_types": [f"{gym}_hard"], "count": 12, "seed_base": 100,
    })
    assert response.status_code == 200, response.text
    for demonstration in response.json()["trajectories"]:
        assert demonstration["moves"] == []
        task = REGISTRY.generate(demonstration["task_type"], demonstration["seed"])
        assert extract(demonstration["answer"]) == task.private["answer"]
    assert sandbox.commands == []


async def test_both_sources_work_in_general_groups(a_run, client, sandbox):
    sandbox.fail_writes = True
    run = await a_run(gym_id="general", task_types=[
        "state_tracking_easy", "constraint_logic_easy",
    ], count=20, group_size=2, seed_base=37)
    assert {episode["gym_id"] for episode in run["episodes"]} == {
        "state_tracking", "constraint_logic",
    }
    for first, second in zip(run["episodes"][::2], run["episodes"][1::2]):
        assert first["prompt"] == second["prompt"]
        assert first["tools"] == second["tools"] == []
        task = REGISTRY.generate(first["task_type"], first["seed"], first["split"])
        response = await client.post(f"/gym/episodes/{first['id']}/finish",
                                     json={"reply": task.private["solution"]})
        assert response.json()["score"] == 1
    assert sandbox.commands == []


@pytest.mark.parametrize("reply", [
    "Answer: Ava or Bo", "Answer: Ava\nAnswer: Bo", "<|thinking|>Ava",
    "<|thinking|>Ava<|/thinking|>", "<|tool-call|>Ava<|/tool-call|>\nAva",
    "Answer: Ava " * 5000,
])
def test_text_answer_rejects_ambiguous_or_unfinished_output(reply):
    assert extract(reply) != "Ava"
