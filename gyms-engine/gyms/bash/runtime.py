"""Bash owns workspace setup, shaping, demonstrations and final file checks."""

import shlex
from typing import Callable

from gyms import sandbox
from gyms.bash import expert, tasks
from gyms.bash.grade import grade, mutation_matches
from gyms.bash.rewards import Earned, Step, step_reward
from gyms.contracts import Generation, Observation, Task, UnsupportedAction

VERSION = 1


def generator(make: Callable[[int], tasks.Task]) -> Callable[[Generation], Task]:
    def generate(request: Generation) -> Task:
        if request.version != VERSION:
            raise ValueError(f"unsupported bash generator version {request.version}")
        task = make(request.seed)
        return Task.restore("bash", {**task.stored(), "generator_version": VERSION}, ("bash",))
    return generate


class Provider:
    tools = ("bash",)
    splits = ()
    version = VERSION
    demonstrations = True

    def prepare(self, session: str, task: Task) -> None:
        sandbox.seed(session, sandbox.WORKSPACE, task.private["files"])

    def step(self, session: str, task: Task, tool: str, argument: str,
             number: int, history: list[tuple[str, str]]) -> Observation:
        if tool not in self.tools:
            raise UnsupportedAction(f"this gym does not support {tool!r}")
        ran = sandbox.run(session, argument)
        target, evidence = task.private.get("target", ""), task.private.get("evidence", "")
        reward = step_reward(
            Step(argument, ran.stdout, ran.stderr, ran.exit_code, ran.timed_out),
            number, target, evidence, Earned.over(history, target, evidence),
        )
        return Observation(ran.stdout, ran.stderr, ran.exit_code, ran.timed_out, reward)

    def finish(self, session: str, task: Task, reply: str) -> tuple[float, str]:
        score, answer = grade(reply, task.private["answer"])
        mutation = task.private.get("mutation")
        if score >= 1.0 and mutation:
            checked = sandbox.run(session, f"cat {shlex.quote(mutation['path'])}")
            if not mutation_matches(checked.stdout, checked.exit_code, mutation["contains"]):
                score = 0.0
        return score, answer

    def demonstrate(self, session: str, task: Task) -> dict:
        original = tasks.Task(
            task.task_type, task.seed, task.prompt, task.private["answer"],
            task.private["files"], task.system, task.private.get("target", ""),
            task.private.get("evidence", ""),
            {key: value for key, value in task.private.items()
             if key not in {"answer", "files", "target", "evidence"}},
        )
        moves = []
        try:
            sandbox.clear(session)
            self.prepare(session, task)
            for move in expert.solve(original).moves:
                ran = sandbox.run(session, move.command)
                moves.append({"reply": move.reply, "command": move.command,
                              "stdout": ran.stdout, "stderr": ran.stderr,
                              "exit_code": ran.exit_code, "timed_out": ran.timed_out})
            return {**task.public(), "moves": moves, "answer": task.private["answer"]}
        finally:
            sandbox.clear(session)
