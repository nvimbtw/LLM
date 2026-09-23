"""Shared no-tool runtime for short, exactly graded text answers."""

import re

from gyms.contracts import Observation, Task, UnsupportedAction

MAX_REPLY = 32_768
MAX_ANSWER = 80


def extract(reply: str) -> str:
    if len(reply) > MAX_REPLY or "<|tool-call|>" in reply:
        return ""
    reply = re.sub(r"<\|thinking\|>.*?(?:<\|/thinking\|>|$)", "", reply, flags=re.S)
    reply = re.sub(r"<\|/?[^|]+\|>", "\n", reply)
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return ""
    final = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", lines[-1], flags=re.I)
    final = final.removesuffix(".").strip()
    return final if len(final) <= MAX_ANSWER and re.fullmatch(r"[A-Za-z]+", final) else ""


class Provider:
    tools = ()
    splits = ("train", "validation", "test")
    version = 1
    demonstrations = True

    def prepare(self, session: str, task: Task) -> None:
        pass

    def step(self, session: str, task: Task, tool: str, argument: str,
             number: int, history: list[tuple[str, str]]) -> Observation:
        raise UnsupportedAction("this gym has no tools")

    def finish(self, session: str, task: Task, reply: str) -> tuple[float, str]:
        answer = extract(reply)
        return float(bool(answer) and answer.casefold() == task.private["answer"].casefold()), answer

    def demonstrate(self, session: str, task: Task) -> dict:
        return {**task.public(), "moves": [], "answer": task.private["solution"]}
