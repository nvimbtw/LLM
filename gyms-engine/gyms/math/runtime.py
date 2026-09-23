"""The same episode contract as a tool gym, without an environment."""

from gyms.contracts import Generation, Observation, Task, UnsupportedAction
from gyms.math import tasks
from gyms.math.grade import grade


def generate(family: str, level: str, request: Generation) -> Task:
    if request.version != tasks.VERSION:
        raise ValueError(f"unsupported math generator version {request.version}")
    problem = tasks.generate(family, level, request.seed, request.split)
    return Task.restore("math", problem.stored(), ())


class Provider:
    tools = ()
    splits = ("train", "validation", "test")
    version = tasks.VERSION
    demonstrations = True

    def prepare(self, session: str, task: Task) -> None:
        pass

    def step(self, session: str, task: Task, tool: str, argument: str,
             number: int, history: list[tuple[str, str]]) -> Observation:
        raise UnsupportedAction("this gym has no tools")

    def finish(self, session: str, task: Task, reply: str) -> tuple[float, str]:
        return grade(reply, task.private["answer"])

    def demonstrate(self, session: str, task: Task) -> dict:
        return {**task.public(), "moves": [], "answer": task.private["solution"]}
