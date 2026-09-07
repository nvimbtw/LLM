"""Answers checked independently of the generators and adversarial final replies."""

import ast
import operator
import re

import pytest

from gyms.math.grade import grade
from gyms.math.tasks import FAMILIES, LEVELS, generate
from gyms.registry import REGISTRY


def arithmetic(text):
    operations = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
    def visit(node):
        if isinstance(node, ast.Constant):
            return node.value
        return operations[type(node.op)](visit(node.left), visit(node.right))
    return visit(ast.parse(text, mode="eval").body)


@pytest.mark.parametrize("family", FAMILIES)
@pytest.mark.parametrize("level", LEVELS)
def test_generated_ground_truth_and_demonstration(family, level):
    for seed in range(100):
        task = generate(family, level, seed)
        assert task == generate(family, level, seed)
        if family == "word":
            nums = list(map(int, re.findall(r"\d+", task.prompt)))
            expected = nums[0] + nums[1] if level == "easy" else nums[0] - nums[1] + nums[2] if level == "medium" else nums[0] * nums[1] + nums[2]
        else:
            expected = arithmetic(task.problem_key)
        assert int(task.answer) == expected
        assert grade(task.solution, task.answer)[0] == 1
        assert "answer" not in task.public() and "solution" not in task.public()
        assert len(task.prompt) < 200


@pytest.mark.parametrize("reply", ["5", "5.0", "10/2", "Answer: 5", r"$\boxed{5}$", "<|par|>Answer: 5<|/par|>"])
def test_equivalent_final_values(reply):
    assert grade(reply, "5")[0] == 1


@pytest.mark.parametrize("reply", ["5 or 6", "The working contains 5.\nAnswer: 6", "Answer: 4.99999", "Answer: 5/0", "<|thinking|>5", "<|thinking|>5<|/thinking|>", "<|tool-call|>5<|/tool-call|>\n5", "__import__('os')", "5 " * 20000])
def test_wrong_or_ambiguous_answers_never_pass(reply):
    assert grade(reply, "5")[0] == 0


def test_the_actual_chat_failure():
    assert grade("<|par|>2+2=4<|/par|><|par|>4+2=6<|/par|><|par|>$\\boxed{6}$<|/par|>", "4")[0] == 0


async def test_math_has_no_sandbox_and_cannot_use_tools(client, sandbox, a_run):
    sandbox.fail_writes = True
    run = await a_run(gym_id="math", task_types=["math_addition_easy"], count=4, group_size=2)
    assert sandbox.commands == []
    first, second = run["episodes"][:2]
    assert first["prompt"] == second["prompt"]
    assert first["tools"] == []
    assert "ground_truth" not in first and "solution" not in first
    refused = await client.post(f"/gym/episodes/{first['id']}/step", json={"command": "echo 5"})
    assert refused.status_code == 400 and sandbox.commands == []
    task = REGISTRY.generate(first["task_type"], first["seed"], first["split"])
    result = await client.post(f"/gym/episodes/{first['id']}/finish", json={"reply": task.solution})
    assert result.json()["score"] == 1


async def test_math_demonstrations_do_not_touch_the_shell(client, sandbox):
    sandbox.fail_writes = True
    response = await client.post("/gym/trajectories", json={"gym_id": "math", "task_types": ["math_word_hard"], "count": 10})
    assert response.status_code == 200
    for trajectory in response.json()["trajectories"]:
        assert trajectory["moves"] == []
        task = REGISTRY.generate(trajectory["task_type"], trajectory["seed"])
        assert grade(trajectory["answer"], task.answer)[0] == 1
    assert sandbox.commands == []


async def test_exams_contain_distinct_problems(a_run):
    for family in FAMILIES:
        run = await a_run(gym_id="math", task_types=[f"math_{family}_easy"], count=100, split="validation")
        problems = [REGISTRY.generate(e["task_type"], e["seed"], e["split"]).problem_key for e in run["episodes"]]
        assert len(set(problems)) == 100


async def test_mixed_demonstrations_are_refused(client):
    result = await client.post("/gym/trajectories", json={"gym_id": "math", "task_types": ["bash_grep"], "count": 1})
    assert result.status_code == 400
