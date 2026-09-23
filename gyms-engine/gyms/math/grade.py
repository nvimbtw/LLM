"""Parse one final numeric answer; never search the working for a correct number.

Only bounded numeric literals are accepted. No expression evaluation, symbolic
execution, or tolerance that could reward a nearby integer.
"""

import re
from fractions import Fraction

MAX_REPLY = 32_768
MAX_NUMBER = 80
NUMBER = re.compile(r"[+-]?(?:\d+(?:\.\d+)?|\.\d+)(?:/[+-]?\d+)?")
BOX = re.compile(r"\\boxed\{([^{}]*)\}")


def extract(reply: str) -> str:
    if len(reply) > MAX_REPLY or "<|tool-call|>" in reply:
        return ""
    # An unfinished thinking block is not a final answer.
    reply = re.sub(r"<\|thinking\|>.*?(?:<\|/thinking\|>|$)", "", reply, flags=re.S)
    reply = re.sub(r"<\|/?[^|]+\|>", "\n", reply)
    lines = [line.strip() for line in reply.splitlines() if line.strip()]
    if not lines:
        return ""
    final = lines[-1].replace("−", "-")
    final = re.sub(r"^(?:final\s+answer|answer)\s*:\s*", "", final, flags=re.I)
    final = final.strip("$ ").removeprefix(r"\[").removesuffix(r"\]").strip()
    match = BOX.fullmatch(final)
    if match:
        final = match[1].strip()
    final = final.removesuffix(".").strip()
    if len(final) > MAX_NUMBER or not NUMBER.fullmatch(final):
        return ""
    try:
        Fraction(final)
    except (ValueError, ZeroDivisionError):
        return ""
    return final


def grade(reply: str, ground_truth: str) -> tuple[float, str]:
    answer = extract(reply)
    return float(bool(answer) and Fraction(answer) == Fraction(ground_truth)), answer
