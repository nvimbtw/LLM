"""The service's half of an episode: create, step, grade.

The loop these routes serve lives in the backend, so what is tested here is
each move on its own — and, above all, the two things a driver cannot check for
itself: that a step is recorded in the order it happened, and that the ground
truth is not handed out while it could still be used.
"""

import pytest

from gyms.sandbox import SandboxError


async def test_the_catalogue_lists_the_bash_gym_and_its_task_types(client):
    """What the frontend's run form is built out of.

    The ids are not enumerated here. This test used to pin the three that
    existed, which made adding a task type a change in two places and said
    nothing about whether the route served a *usable* row — the registry's own
    suite is where the roster is checked. What a caller cannot do without is
    every field it renders, so that is what this asserts.
    """
    listing = (await client.get("/gyms")).json()
    assert [gym["id"] for gym in listing] == ["bash"]
    task_types = listing[0]["task_types"]
    assert task_types
    for task_type in task_types:
        assert task_type.keys() == {"id", "label", "description", "level", "family"}
        assert all(task_type.values())


async def test_a_run_seeds_a_workspace_for_every_episode(a_run, sandbox):
    run = await a_run(count=3, task_types=["bash_grep", "bash_csv", "bash_code"])

    assert [e["index"] for e in run["episodes"]] == [0, 1, 2]
    assert [e["task_type"] for e in run["episodes"]] == ["bash_grep", "bash_csv", "bash_code"]
    # One filesystem per episode, keyed by the episode's own id.
    assert {e["id"] for e in run["episodes"]} <= set(sandbox.files)


async def test_a_run_is_reproducible_from_its_seed(a_run):
    first = await a_run(seed_base=99)
    second = await a_run(seed_base=99)
    assert first["episodes"][0]["prompt"] == second["episodes"][0]["prompt"]


async def test_seeding_that_fails_is_an_error_not_a_healthy_run(a_run, sandbox, client):
    """The failure this whole module exists to prevent: a run handed back over
    an empty filesystem, where every episode looks like a model that found
    nothing."""
    sandbox.fail_writes = True
    response = await client.post(
        "/gym/runs",
        json={"gym_id": "bash", "model_name": "tiny", "task_types": ["bash_grep"], "count": 1},
    )
    assert response.status_code == 502
    assert (await client.get("/gym/runs")).json() == []


async def test_an_unreachable_sandbox_is_a_bad_gateway(a_run, sandbox, client, monkeypatch):
    run = await a_run()
    episode = run["episodes"][0]

    def unreachable(*args, **kwargs):
        raise SandboxError("just-bash did not answer")

    import api.routes.gym as routes

    monkeypatch.setattr(routes, "run_command", unreachable)
    response = await client.post(
        f"/gym/episodes/{episode['id']}/step", json={"command": "ls"}
    )
    assert response.status_code == 502


async def test_steps_are_numbered_in_the_order_they_happened(a_run, client):
    run = await a_run()
    episode = run["episodes"][0]["id"]

    for command in ["ls", "grep -r x .", "ls -R"]:
        answered = await client.post(
            f"/gym/episodes/{episode}/step", json={"command": command, "reply": "thinking"}
        )
        assert answered.status_code == 200

    steps = (await client.get(f"/gym/episodes/{episode}")).json()["steps"]
    assert [s["step_num"] for s in steps] == [0, 1, 2]
    assert [s["action"] for s in steps] == ["ls", "grep -r x .", "ls -R"]
    assert all(s["reward"] is not None for s in steps)
    assert all(s["reply"] == "thinking" for s in steps)


async def test_the_ground_truth_is_withheld_until_the_episode_is_graded(a_run, client):
    run = await a_run()
    episode = run["episodes"][0]

    open_view = (await client.get(f"/gym/episodes/{episode['id']}")).json()
    assert "ground_truth" not in open_view
    assert open_view["prompt"] and open_view["system"]

    await client.post(f"/gym/episodes/{episode['id']}/finish", json={"reply": "guess.txt"})
    graded = (await client.get(f"/gym/episodes/{episode['id']}")).json()
    assert graded["ground_truth"]
    assert graded["status"] == "graded"


async def test_grading_scores_the_reply_against_the_stored_answer(a_run, client):
    run = await a_run()
    episode = run["episodes"][0]
    truth = _answer_of(episode["id"])

    verdict = (
        await client.post(
            f"/gym/episodes/{episode['id']}/finish", json={"reply": f"<|par|>{truth}<|/par|>"}
        )
    ).json()
    assert verdict["score"] == 1.0
    assert verdict["correct"] is True
    assert verdict["answer"] == truth


async def test_a_run_score_is_the_mean_over_graded_episodes(a_run, client):
    run = await a_run(count=2)
    right, wrong = run["episodes"]

    await client.post(
        f"/gym/episodes/{right['id']}/finish", json={"reply": _answer_of(right["id"])}
    )
    await client.post(f"/gym/episodes/{wrong['id']}/finish", json={"reply": "nonsense"})

    scored = (await client.get(f"/gym/runs/{run['id']}")).json()
    assert scored["score"] == 0.5
    assert (scored["graded"], scored["correct"]) == (2, 1)


async def test_grading_twice_is_refused_rather_than_counted_twice(a_run, client):
    run = await a_run()
    episode = run["episodes"][0]["id"]

    await client.post(f"/gym/episodes/{episode}/finish", json={"reply": "a.txt"})
    again = await client.post(f"/gym/episodes/{episode}/finish", json={"reply": "a.txt"})
    assert again.status_code == 409


async def test_a_closed_episode_takes_no_more_steps(a_run, client):
    run = await a_run()
    episode = run["episodes"][0]["id"]

    await client.post(f"/gym/episodes/{episode}/close")
    refused = await client.post(f"/gym/episodes/{episode}/step", json={"command": "ls"})
    assert refused.status_code == 409


async def test_a_closed_episode_is_absent_from_the_run_mean(a_run, client):
    """Never answered is not the same as answered wrongly, and a run that
    counted it as zero would blame the model for this side giving up."""
    run = await a_run(count=2)
    graded, abandoned = run["episodes"]

    await client.post(
        f"/gym/episodes/{graded['id']}/finish", json={"reply": _answer_of(graded["id"])}
    )
    await client.post(f"/gym/episodes/{abandoned['id']}/close")

    scored = (await client.get(f"/gym/runs/{run['id']}")).json()
    assert scored["score"] == 1.0
    assert scored["graded"] == 1


async def test_a_task_type_from_another_gym_is_refused(client):
    response = await client.post(
        "/gym/runs",
        json={"gym_id": "bash", "model_name": "tiny", "task_types": ["nope"], "count": 1},
    )
    assert response.status_code == 400


async def test_an_unknown_episode_is_a_404(client, unknown_id):
    assert (await client.get(f"/gym/episodes/{unknown_id}")).status_code == 404


def _answer_of(episode_id: str) -> str:
    """The ground truth, read the way only a test may read it."""
    import uuid

    import store
    from store.schema import GymEpisode

    with store.session() as db:
        return db.get(GymEpisode, uuid.UUID(episode_id)).task_config["answer"]


async def test_demonstrations_are_generated_without_becoming_runs(client, sandbox):
    """A demonstration is training data, not a measurement.

    Recording one as a run would put a perfect score into every board and mean
    this service feeds — the model did not earn it, the generator did.
    """
    body = {"task_types": ["bash_grep", "bash_csv"], "count": 4, "seed_base": 5}
    answered = (await client.post("/gym/trajectories", json=body)).json()

    assert answered["count"] == 4
    assert [t["task_type"] for t in answered["trajectories"]] == [
        "bash_grep", "bash_csv", "bash_grep", "bash_csv",
    ]
    assert (await client.get("/gym/runs")).json() == []


async def test_a_demonstration_ends_on_the_answer_and_shows_its_work(client, sandbox):
    """The shape behaviour cloning needs: commands that ran, and a last turn
    that is the answer rather than a plan to look for it."""
    body = {"task_types": ["bash_grep"], "count": 1, "seed_base": 5}
    built = (await client.post("/gym/trajectories", json=body)).json()["trajectories"][0]

    assert built["answer"]
    assert built["moves"], "a demonstration with no commands demonstrates nothing"
    for move in built["moves"]:
        assert move["command"] and "stdout" in move and "exit_code" in move
    # Raw, never rendered: the observation the model reads is built in Rust.
    assert not any("tool-result" in move["stdout"] for move in built["moves"])


async def test_demonstrations_share_one_workspace_rather_than_one_each(client, sandbox):
    """The sandbox keeps a bounded number of filesystems, and cloning wants
    thousands of tasks — so a batch reuses a session and clears it between
    tasks. An episode may never do that; its id *is* its filesystem."""
    body = {"task_types": ["bash_csv"], "count": 6, "seed_base": 5}
    await client.post("/gym/trajectories", json=body)

    sessions = {session for session, _ in sandbox.commands}
    assert len(sessions) == 1


async def test_an_unknown_task_type_is_refused_before_anything_is_seeded(client, sandbox):
    body = {"task_types": ["bash_nothing"], "count": 1}
    response = await client.post("/gym/trajectories", json=body)
    assert response.status_code == 400
    assert not sandbox.files
