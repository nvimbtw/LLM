"""What a single step inside an episode is worth.

Shaping, not scoring. The score of an episode is the grade of its final answer
and nothing else (`gyms.bash.grade`); this is the smaller, denser signal that
says whether the step the model just took was *work* — the thing a policy
gradient needs between the sparse ones, and the thing a person reading the
trajectory chart wants to see rising.

Kept deliberately small against the terminal 1.0. A step reward large enough to
compete with the answer is a reward a model can farm by running commands
forever, which is the failure mode this shape is chosen to avoid: the positive
terms are bounded, the two that are about the task are paid once an episode, and
everything decays with how long the episode has gone on.

**What the first runs of this measured.** Across 174 steps of a correctly seeded
run, `RAN_CLEANLY` fired 127 times and `ON_TARGET` fired *zero* — so the only
signal a model actually received was "your command parsed", which `ls` earns as
readily as `grep`. `SAW_TARGET` exists because of that: it is the term that can
tell those two apart. See
`docs/decisions/reinforcement-learning-in-the-llm-engine.md`.
"""

from dataclasses import dataclass

#: A command the shell accepted. The floor of "did something real", and the
#: only term that is not a penalty.
RAN_CLEANLY = 0.05

#: A command that named the thing the task is about — the needle, the marker.
#: Small, because recognising what to search for is a step toward the answer and
#: not the answer.
#:
#: Paid **once an episode**. It used to be payable every step, which under
#: evaluation is harmless — a model types the needle once — and under RL is a
#: term farmed by repeating one good command forever.
ON_TARGET = 0.03

#: The task's evidence string came back in a command's *output*: the answer is
#: now on the model's screen. The largest shaping term, because it is the only
#: one that is about the task rather than about the shell — `RAN_CLEANLY` pays
#: for `ls` run a third time, and this does not.
#:
#: It is charged against the **output**, never the command, which is what makes
#: it unfarmable: the model cannot put the evidence on screen by typing it,
#: because the shell has to produce it. Paid once an episode for the same reason
#: `ON_TARGET` is.
SAW_TARGET = 0.10

#: Charged per step past `FREE_STEPS`, so an episode that wanders costs more the
#: longer it wanders. Bounded by MAX_GYM_ROUNDS on the driver's side, so this
#: can only ever subtract a few hundredths in total.
DAWDLE = 0.02
FREE_STEPS = 3

#: A command the shell rejected outright — a typo, or a tool this sandbox does
#: not implement. Distinguished from a non-zero exit, which is often a legitimate
#: `grep` that matched nothing.
NOT_FOUND_EXIT = 127
NOT_A_COMMAND = -0.05


@dataclass(frozen=True)
class Step:
    """One executed command, as the sandbox reported it."""

    command: str
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False


@dataclass(frozen=True)
class Earned:
    """The once-an-episode terms an earlier step has already been paid.

    Derived from the steps on record (`store.queries.step_texts`) rather than
    stored beside them, so a replayed or re-graded episode cannot pay twice and
    no column has to be kept in step with what this module pays for.
    """

    on_target: bool = False
    saw_target: bool = False

    @classmethod
    def over(cls, steps: list[tuple[str, str]], target: str, evidence: str) -> "Earned":
        """What `steps` — `(command, output)` pairs — has already earned."""
        return cls(
            on_target=bool(target) and any(target in command for command, _ in steps),
            saw_target=bool(evidence) and any(evidence in out for _, out in steps),
        )


def step_reward(
    step: Step,
    step_num: int,
    target: str,
    evidence: str = "",
    earned: Earned = Earned(),
) -> float:
    """`step`'s contribution to the episode's return.

    `target` is what the task told the model to look for and is matched against
    the **command**; `evidence` is what proves the answer is on screen and is
    matched against the **output**. Neither is ever the answer on the command
    side: rewarding the answer appearing in a command would pay a model for
    typing something it could only know by having already found it, which
    teaches nothing and inflates every episode that got lucky.
    """
    reward = 0.0
    if step.timed_out:
        reward += NOT_A_COMMAND
    elif step.exit_code == NOT_FOUND_EXIT:
        reward += NOT_A_COMMAND
    elif step.exit_code == 0:
        reward += RAN_CLEANLY

    if target and not earned.on_target and target in step.command:
        reward += ON_TARGET

    if evidence and not earned.saw_target and evidence in step.stdout:
        reward += SAW_TARGET

    reward -= DAWDLE * max(0, step_num - FREE_STEPS)
    return round(reward, 4)
