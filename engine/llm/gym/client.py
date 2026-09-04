"""Reaching a gym episode through the backend.

The engine cannot reach the sandbox: the Python services never speak, so every
command a rollout runs is a hop toward the backend regardless of who renders the
observation. This is that hop, and the rendering rides it — `step` answers with
the *rendered* block and the engine never sees a `{stdout, stderr, exit_code}`
triple at all. A Python copy of that layout would be a gym teaching a shell the
model is never served.

**Nothing here retries.** `llm.backend.Hop` makes one attempt, and that is the
right number for every call in this file: a step is a command that ran in a real
filesystem, and asking again after a lost response runs it twice, records two
steps and pays the shaping reward twice for one action.

**Nothing here is written down.** The gyms service owns the record — runs,
episodes, steps, rewards — exactly as it does for a browser-driven run. A batch
is held in memory, its step is taken, and it is dropped. A trajectory the engine
keeps a copy of is a trajectory the engine owns, and it owns models.
"""

from dataclasses import dataclass

from llm.backend import Hop, RemoteError

#: How long a batch of concurrent steps may take. Generous against the per-call
#: timeout because the whole point of the lockstep rollout is that `G x T`
#: commands are one wait rather than `G x T` waits, and just-bash is answering
#: all of them at once.
BATCH_TIMEOUT_SECS = 120.0


@dataclass(frozen=True)
class Episode:
    """One task, as the engine is allowed to see it.

    No answer. `Task.public()` is what an open episode's row renders to, and the
    ground truth appears only in what `finish` returns — at which point it can no
    longer help anyone. That the trainer is on this side of the wall and not the
    grader's side is not an accident of layering: a rollout that could read the
    answer is a rollout that could be paid for having been handed it.
    """

    id: str
    index: int
    gym_id: str
    task_type: str
    seed: int
    prompt: str
    system: str


@dataclass(frozen=True)
class Observed:
    """What one tool call produced: what the model reads, and what it earned."""

    observation: str
    reward: float
    step_num: int


@dataclass(frozen=True)
class Graded:
    score: float
    answer: str


class Client(Hop):
    """One rollout's view of the gym runs it drives."""

    def create_run(
        self,
        *,
        gym_id: str,
        model_name: str,
        task_types: list[str],
        count: int,
        seed_base: int,
        group_size: int,
    ) -> tuple[str, list[Episode]]:
        """A run and its episodes, workspaces already seeded.

        `kind` is `rl` and not the caller's to choose. A training run takes one
        of these per optimizer step; leaving them on the board a person reads to
        compare models would bury every evaluation anybody made.
        """
        payload, _ = self.request(
            "POST",
            "/internal/gym/runs",
            json={
                "gym_id": gym_id,
                "model_name": model_name,
                "task_types": task_types,
                "count": count,
                "seed_base": seed_base,
                "group_size": group_size,
                "kind": "rl",
            },
        )
        episodes = [
            Episode(
                id=str(row["id"]),
                index=int(row["index"]),
                gym_id=str(row["gym_id"]),
                task_type=str(row["task_type"]),
                seed=int(row["seed"]),
                prompt=str(row.get("prompt", "")),
                system=str(row.get("system", "")),
            )
            for row in payload.get("episodes", [])
        ]
        return str(payload["id"]), episodes

    def step(self, episode_id: str, tool: str, argument: str, reply: str) -> Observed:
        payload, _ = self.request(
            "POST",
            f"/internal/gym/episodes/{episode_id}/step",
            json={"tool": tool, "argument": argument, "reply": reply},
        )
        return Observed(
            observation=str(payload.get("observation", "")),
            reward=float(payload.get("reward") or 0.0),
            step_num=int(payload.get("step_num") or 0),
        )

    def finish(self, episode_id: str, reply: str) -> Graded:
        payload, _ = self.request(
            "POST", f"/internal/gym/episodes/{episode_id}/finish", json={"reply": reply}
        )
        return Graded(
            score=float(payload.get("score") or 0.0),
            answer=str(payload.get("answer") or ""),
        )

    def close(self, episode_id: str) -> None:
        """Abandon an episode without grading it.

        An abandoned episode is absent from a run's mean rather than a zero, and
        it is absent from a batch for the same reason: it measures this side
        giving up, not the model. An episode that exhausted its rounds is a
        different thing and is graded on what it last said.
        """
        self.request("POST", f"/internal/gym/episodes/{episode_id}/close")

    def set_status(self, run_id: str, status: str) -> None:
        """What the driver says is happening to the run it holds.

        Best-effort: a batch whose gradient has already been taken is not worth
        failing over a status nobody is watching, and the run row is the gyms
        service's own bookkeeping rather than anything this iteration reads back.
        """
        try:
            self.request(
                "POST", f"/internal/gym/runs/{run_id}/status", json={"status": status}
            )
        except RemoteError:
            pass
