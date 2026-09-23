"""A mixture keeps the source gym's question, environment and grading intact."""

from collections import Counter
from dataclasses import replace
import uuid

import pytest

import store
from gyms.contracts import Task
from gyms.registry import BASH, GYMS, REGISTRY, Gym, GymRegistry
from gyms.selection import resolve
from store.schema import GymEpisode


def stored(episode):
    with store.session() as db:
        return db.get(GymEpisode, uuid.UUID(episode["id"])).task_config


@pytest.mark.parametrize("task_type", [t.id for gym in GYMS for t in gym.task_types])
def test_every_provider_generates_a_replayable_private_task(task_type):
    task = REGISTRY.generate(task_type, 41, "test")
    assert task == REGISTRY.generate(task_type, task.seed, "test", task.generator_version)
    assert Task.restore(task.gym_id, task.stored(), ()) == task
    assert task.gym_id == REGISTRY.gym_of(task_type)
    assert task.private["answer"]
    assert not {"answer", "solution", "files", "target", "evidence", "mutation", "problem_key"} & task.public().keys()
    assert task.split == ("test" if REGISTRY.provider(task.gym_id).splits else None)
    with pytest.raises(ValueError, match="version"):
        REGISTRY.generate(task_type, 41, version=999)


def test_the_composite_catalogue_does_not_reassign_task_ownership():
    listing = REGISTRY.listing()
    general = next(gym for gym in listing if gym["id"] == "general")
    assert general["composite"]
    source_types = {t["id"] for gym in listing if not gym["composite"] for t in gym["task_types"]}
    assert {t["id"] for t in general["task_types"]} == source_types
    assert all(t["gym_id"] == REGISTRY.gym_of(t["id"]) != "general" for t in general["task_types"])
    with pytest.raises(ValueError, match="duplicate task"):
        GymRegistry((BASH, replace(BASH, id="duplicate")))


def test_new_source_registration_is_enough_and_existing_plans_stay_fixed():
    plan = resolve(REGISTRY, "general", None, {}, {}, "test")
    before = [plan.choose(0, group) for group in range(50)]
    novel = replace(BASH.task_types[0], id="other_question")
    registry = GymRegistry((*GYMS, Gym("other", "Other", "Another source", (novel,), BASH.provider)))
    expanded = resolve(registry, "general", None, {}, {}, "test")
    assert "other_question" in expanded.task_types
    assert "other_question" not in plan.task_types
    assert [plan.choose(0, group) for group in range(50)] == before


def test_sampling_balances_sources_before_tasks_and_honours_weights():
    types = ["bash_ls_easy", "math_addition_easy", "math_subtraction_easy", "math_word_easy"]
    for weights, expected in [({}, 0.5), ({"bash": 3}, 0.75)]:
        plan = resolve(REGISTRY, "general", types, weights, {}, "train")
        counts = Counter(REGISTRY.gym_of(plan.choose(8, i)) for i in range(4000))
        assert abs(counts["bash"] / 4000 - expected) < 0.04
    plan = resolve(REGISTRY, "general", types[1:3], {}, {types[1]: 3}, "train")
    counts = Counter(plan.choose(8, i) for i in range(4000))
    assert 0.71 < counts[types[1]] / 4000 < 0.79


async def test_mixed_groups_keep_source_tools_private_state_and_separate_workspaces(a_run, client, sandbox):
    run = await a_run(gym_id="general", task_types=["bash_read_easy", "math_addition_easy"],
                      count=24, group_size=2, seed_base=0)
    assert run["gym_id"] == "general"
    assert {e["gym_id"] for e in run["episodes"]} == {"bash", "math"}
    for first, second in zip(run["episodes"][::2], run["episodes"][1::2]):
        assert stored(first) == stored(second)
        assert first["id"] != second["id"]
        assert first["tools"] == (["bash"] if first["gym_id"] == "bash" else [])
        assert first["answer"] is None
        assert not {"ground_truth", "solution", "files", "mutation"} & first.keys()
        response = await client.post(f"/gym/episodes/{first['id']}/step", json={"command": "ls"})
        assert response.status_code == (200 if first["gym_id"] == "bash" else 400)
    shell_ids = {e["id"] for e in run["episodes"] if e["gym_id"] == "bash"}
    assert set(sandbox.files) == shell_ids
    reread = (await client.get(f"/gym/runs/{run['id']}")).json()
    assert reread["selection"] == run["selection"]


async def test_source_grading_and_aggregates_survive_reload(a_run, client):
    run = await a_run(gym_id="general", task_types=["bash_ls_easy", "math_addition_easy"], count=16, seed_base=0)
    for episode in run["episodes"]:
        truth = stored(episode)["answer"]
        reply = f"Answer: {truth}.0" if episode["gym_id"] == "math" else "wrong"
        result = await client.post(f"/gym/episodes/{episode['id']}/finish", json={"reply": reply})
        assert result.status_code == 200
        assert result.json()["score"] == float(episode["gym_id"] == "math")
    result = (await client.get(f"/gym/runs/{run['id']}")).json()
    assert result["source_scores"]["math"]["score"] == 1
    assert result["source_scores"]["bash"]["score"] == 0
    assert result["score"] == round(result["source_scores"]["math"]["graded"] / 16, 4)


async def test_selection_replays_after_reload_and_request_reordering(a_run, client):
    options = dict(gym_id="general", task_types=["bash_ls_easy", "math_addition_easy"],
                   count=12, group_size=2, seed_base=9, gym_weights={"math": 2})
    first = await a_run(**options)
    second = await a_run(**{**options, "task_types": list(reversed(options["task_types"]))})
    assert [stored(e) for e in first["episodes"]] == [stored(e) for e in second["episodes"]]
    assert first["selection"] == second["selection"]
    for episode in first["episodes"]:
        task = REGISTRY.generate(episode["task_type"], episode["seed"],
                                 episode["split"] or "train", episode["generator_version"])
        assert task.stored() == stored(episode)


async def test_new_source_provider_works_through_the_same_routes(a_run, client, monkeypatch):
    from gyms.registry import MATH
    import api.routes.gym as routes

    original = MATH.task_types[0]
    task_type = replace(original, id="other_question", generate=lambda request:
                        replace(original.generate(request), gym_id="other", task_type="other_question"))
    source = Gym("other", "Other", "A new tool-free provider", (task_type,), MATH.provider)
    registry = GymRegistry((*GYMS, source))
    monkeypatch.setattr(routes, "REGISTRY", registry)
    run = await a_run(gym_id="general", task_types=["other_question"])
    episode = run["episodes"][0]
    assert episode["gym_id"] == "other" and episode["tools"] == []
    result = await client.post(f"/gym/episodes/{episode['id']}/finish",
                               json={"reply": stored(episode)["answer"]})
    assert result.json()["score"] == 1


async def test_failed_mutation_check_is_retryable_and_success_requires_written_content(a_run, client, monkeypatch):
    from gyms import sandbox
    run = await a_run(gym_id="general", task_types=["bash_write_easy"])
    episode = run["episodes"][0]
    config = stored(episode)

    def unreachable(*args):
        raise sandbox.SandboxError("sandbox is unavailable")

    monkeypatch.setattr(sandbox, "run", unreachable)
    result = await client.post(f"/gym/episodes/{episode['id']}/finish", json={"reply": config["answer"]})
    assert result.status_code == 502
    current = (await client.get(f"/gym/episodes/{episode['id']}")).json()
    assert current["status"] == "pending" and "ground_truth" not in current
    monkeypatch.setattr(sandbox, "run", lambda *args:
                        sandbox.Ran(config["mutation"]["contains"], "", 0, False))
    result = await client.post(f"/gym/episodes/{episode['id']}/finish", json={"reply": config["answer"]})
    assert result.json()["score"] == 1


async def test_mixed_mutation_grading_requires_the_edit_and_legacy_rows_still_grade(a_run, client, monkeypatch):
    from gyms import sandbox
    run = await a_run(gym_id="general", task_types=["bash_write_easy"])
    episode = run["episodes"][0]
    config = stored(episode)
    monkeypatch.setattr(sandbox, "run", lambda *args: sandbox.Ran("", "", 0, False))
    result = await client.post(f"/gym/episodes/{episode['id']}/finish", json={"reply": config["answer"]})
    assert result.json()["score"] == 0
    run = await a_run(gym_id="general", task_types=["math_addition_easy"])
    episode = run["episodes"][0]
    with store.session() as db:
        row = db.get(GymEpisode, uuid.UUID(episode["id"]))
        # Old rows had no common envelope metadata and must remain gradeable.
        row.task_config = {k: v for k, v in row.task_config.items()
                           if k not in {"gym_id", "tools", "generator_version"}}
    result = await client.post(f"/gym/episodes/{episode['id']}/finish", json={"reply": stored(episode)["answer"]})
    assert result.json()["score"] == 1


async def test_general_can_default_to_all_sources_and_math_only_never_touches_shell(client, sandbox):
    response = await client.post("/gym/runs", json={"gym_id": "general", "model_name": "tiny", "count": 4})
    assert response.status_code == 201
    assert len(response.json()["task_types"]) == sum(len(g.task_types) for g in GYMS)
    sandbox.fail_writes = True
    sandbox.commands.clear()
    response = await client.post("/gym/runs", json={"gym_id": "general", "model_name": "tiny",
                                                  "task_types": ["math_addition_easy"], "count": 4})
    assert response.status_code == 201 and sandbox.commands == []


async def test_mixed_demonstrations_share_the_envelope_and_keep_provenance(client):
    response = await client.post("/gym/trajectories", json={"gym_id": "general",
        "task_types": ["bash_read_easy", "math_addition_easy"], "count": 16, "seed_base": 0})
    body = response.json()
    assert response.status_code == 200 and body["count"] == 16
    assert {t["gym_id"] for t in body["trajectories"]} == {"bash", "math"}
    for task in body["trajectories"]:
        assert bool(task["moves"]) == (task["gym_id"] == "bash")
        assert task["generator_version"] == 1
        assert task["split"] == ("train" if task["gym_id"] == "math" else None)
    assert (await client.get("/gym/runs")).json() == []


@pytest.mark.parametrize("extra", [
    {"gym_weights": {"unknown": 2}}, {"gym_weights": {"general": 2}},
    {"task_weights": {"unknown": 2}}, {"gym_weights": {"math": 0}},
    {"gym_weights": {"math": -1}}, {"task_weights": {"math_addition_easy": "NaN"}},
    {"task_types": ["math_addition_easy", "math_addition_easy"]},
    {"task_types": ["unknown"]}, {"task_types": []},
])
async def test_invalid_mixtures_fail_before_preparing_any_environment(client, sandbox, extra):
    response = await client.post("/gym/runs", json={"gym_id": "general", "model_name": "tiny",
        "task_types": ["bash_ls_easy", "math_addition_easy"], "count": 2, **extra})
    assert response.status_code in (400, 422)
    assert sandbox.commands == []
    assert (await client.get("/gym/runs")).json() == []
