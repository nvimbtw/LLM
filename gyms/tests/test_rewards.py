"""The shape of the step reward, not its constants.

What matters is the ordering — a command that ran beats one that did not, and
neither ever beats the answer — because that is what a policy learns from. The
constants themselves are tuning and are allowed to move.
"""

from gyms.bash.rewards import Earned, Step, step_reward

TARGET = "ANSWER_ab12cd"


def ran(command: str, exit_code: int = 0, timed_out: bool = False) -> Step:
    return Step(command, "out", "", exit_code, timed_out)


def saw(command: str, stdout: str) -> Step:
    return Step(command, stdout, "", 0, False)


def test_a_command_that_ran_beats_one_the_shell_never_heard_of():
    assert step_reward(ran("ls"), 0, TARGET) > step_reward(ran("lss", 127), 0, TARGET)


def test_searching_for_the_needle_is_worth_more_than_looking_around():
    assert step_reward(ran(f"grep -r {TARGET} ."), 0, TARGET) > step_reward(ran("ls"), 0, TARGET)


def test_a_grep_that_matched_nothing_is_not_punished_like_a_typo():
    """Exit 1 from grep is an answer — the needle is not there — and a model
    that learns to avoid it learns to avoid searching."""
    assert step_reward(ran("grep x .", 1), 0, TARGET) > step_reward(ran("nope", 127), 0, TARGET)


def test_a_timeout_is_charged_like_a_command_that_did_not_run():
    assert step_reward(ran("sleep 100", 124, timed_out=True), 0, TARGET) < 0


def test_wandering_costs_more_the_longer_it_goes_on():
    early = step_reward(ran("ls"), 0, TARGET)
    late = step_reward(ran("ls"), 9, TARGET)
    assert late < early


def test_no_run_of_steps_can_outearn_a_correct_answer():
    """A shaping reward large enough to compete with the terminal one is a
    reward a model can farm by running commands forever."""
    best = sum(step_reward(ran(f"grep {TARGET} ."), n, TARGET) for n in range(8))
    assert best < 1.0


def test_seeing_the_evidence_beats_a_command_that_merely_ran():
    """The point of the term: `ls` run a third time earns `RAN_CLEANLY` and
    nothing else, while a command that put the answer on screen is progress."""
    found = step_reward(saw("cat data.csv", "12,target,VAL_1234"), 0, "data.csv", "VAL_1234")
    looked = step_reward(saw("ls", "data.csv"), 0, "data.csv", "VAL_1234")
    assert found > looked


def test_evidence_is_read_from_the_output_and_never_from_the_command():
    """A model cannot earn it by typing its guess — the shell has to print it."""
    typed = step_reward(saw("echo VAL_1234", ""), 0, "data.csv", "VAL_1234")
    printed = step_reward(saw("cat data.csv", "VAL_1234"), 0, "data.csv", "VAL_1234")
    assert typed < printed


def test_neither_task_term_is_paid_twice_in_one_episode():
    """Both are farmable otherwise: one good command, repeated to the round
    budget, would out-earn finding the answer."""
    first = step_reward(saw(f"grep {TARGET} .", TARGET), 0, TARGET, TARGET)
    again = step_reward(
        saw(f"grep {TARGET} .", TARGET), 1, TARGET, TARGET,
        Earned(on_target=True, saw_target=True),
    )
    assert again < first


def test_no_run_of_steps_can_outearn_a_correct_answer_with_evidence_in_hand():
    """The bound `test_no_run_of_steps_can_outearn_a_correct_answer` checks,
    re-checked with every term firing as generously as the rules allow."""
    earned = Earned()
    best = 0.0
    for n in range(8):
        best += step_reward(saw(f"grep {TARGET} .", TARGET), n, TARGET, TARGET, earned)
        earned = Earned(on_target=True, saw_target=True)
    assert best < 1.0
