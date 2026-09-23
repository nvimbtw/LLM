"""Judging what the model finally said.

The one place an answer is turned into a score, and it lives on this side of
the wire on purpose: the ground truth is in `gym_episodes.task_config` and never
leaves the service until the episode it belongs to is closed. A grader in the
browser is a benchmark that grades itself.

## Why this is not an exact match

The episode ends when the reply stops calling the shell, so what arrives here is
a whole assistant turn — markup, possibly a sentence around the answer, possibly
the path the model just grepped. A strict `==` scores a model zero for saying
"The file is `abc/def.txt`." when it did the work and found the file, and the
number that comes out then measures the model's manners rather than its shell
use. So the reply is narrowed to its last line and stripped of the decoration a
small model reliably adds, and a reply that *ends* with the answer counts too.

What it deliberately will not do is search the whole reply for the answer. A
model that lists ten candidate filenames somewhere in a paragraph and then says
something else has not answered, and a substring search anywhere would mark that
correct. Only the end of the last line counts.
"""

import re

#: Markup the reply arrives wrapped in. Blocks whose *content* is not the answer
#: are dropped whole; every other marker is unwrapped and its text kept.
_DISCARDED_BLOCKS = ("thinking", "tool-call", "tool-result")

_MARKER = re.compile(r"<\|/?([a-zA-Z0-9:/-]+)\|>")

#: Stripped from either end of the candidate, repeatedly. Ordinary sentence
#: decoration; none of it can be part of an answer, because every task's answer
#: is a filename, an identifier or a `VAL_`/`ANSWER_` token.
_TRIM = " \t\r\n\"'`*.,;:!?()[]{}<>"

#: Prefixes a model adds when it answers with a path it has just been reading.
#: A task's files sit in the directory the shell opens in, so an answer given as
#: an absolute path has still identified the file.
_PATH_PREFIXES = ("/home/user/", "home/user/", "./", "/")


def strip_markup(reply: str) -> str:
    """The prose of a reply: reasoning and tool traffic removed, markers gone.

    Reasoning is dropped rather than searched because it is where a model talks
    about candidates it went on to reject, and a tool call is dropped because it
    contains the command — for the grep task, a command that frequently contains
    the answer.
    """
    for name in _DISCARDED_BLOCKS:
        reply = re.sub(rf"<\|{name}\|>.*?<\|/{name}\|>", " ", reply, flags=re.S)
        # An unclosed block is the tail of a turn the token cap cut short.
        reply = re.sub(rf"<\|{name}\|>.*", " ", reply, flags=re.S)
    return _MARKER.sub(" ", reply)


def normalize(text: str) -> str:
    """One candidate answer, reduced to the form a comparison can trust."""
    text = text.strip(_TRIM)
    for prefix in _PATH_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    return text.strip(_TRIM)


def extract(reply: str) -> str:
    """The model's answer: the last non-empty line of what it said, normalized.

    A line rather than the whole reply because a model that explains itself puts
    the explanation above the answer far more often than beside it.
    """
    prose = strip_markup(reply)
    lines = [line for line in prose.splitlines() if line.strip()]
    return normalize(lines[-1]) if lines else ""


def grade(reply: str, ground_truth: str) -> tuple[float, str]:
    """`(score, answer)` for one finished episode.

    The answer is reported alongside the score because it is what the trajectory
    view shows: a run whose scores are all zero is only diagnosable if you can
    see what the model actually said.
    """
    truth = normalize(ground_truth)
    answer = extract(reply)
    if not truth:
        return 0.0, answer
    if answer == truth:
        return 1.0, answer
    # The answer ending the line it is on: "The file is abc/def.txt" — the model
    # found it and wrote a sentence. Bounded by a word edge so `xdef.txt` is not
    # accepted for `def.txt`.
    if re.search(rf"(?:^|[\s/]){re.escape(truth)}$", answer):
        return 1.0, answer
    return 0.0, answer


def mutation_matches(stdout: str, exit_code: int, expected: str) -> bool:
    """Whether a post-answer file check proves the requested edit landed."""
    return exit_code == 0 and expected in stdout
