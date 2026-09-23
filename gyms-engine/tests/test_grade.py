"""How forgiving the grader is, stated as cases.

Each of these is something a small model actually writes. The line between the
ones that score and the ones that do not is the one thing about this module
worth pinning down, because moving it moves every score on record.
"""

import pytest

from gyms.bash.grade import extract, grade, normalize, strip_markup

TRUTH = "sub/report.txt"


@pytest.mark.parametrize(
    "reply",
    [
        "sub/report.txt",
        "  sub/report.txt  ",
        "`sub/report.txt`",
        '"sub/report.txt".',
        "/home/user/sub/report.txt",
        "./sub/report.txt",
        "<|par|>sub/report.txt<|/par|>",
        "<|par|>I looked around.<|/par|><|par|>sub/report.txt<|/par|>",
        "The file is sub/report.txt",
        "<|thinking|>could be other.txt<|/thinking|><|par|>sub/report.txt<|/par|>",
    ],
)
def test_these_all_count_as_the_answer(reply):
    score, _ = grade(reply, TRUTH)
    assert score == 1.0


@pytest.mark.parametrize(
    "reply",
    [
        "",
        "other.txt",
        # The answer, but not as the thing being said at the end.
        "sub/report.txt is one candidate; I could not narrow it down",
        # A longer name that merely ends the same way.
        "notsub/report.txt".replace("notsub", "xsub"),
        # Only ever said inside the reasoning it was rejected in.
        "<|thinking|>sub/report.txt maybe<|/thinking|><|par|>other.txt<|/par|>",
        # Only ever said inside a command.
        "<|tool-call|>{\"tool\": \"bash\", \"argument\": \"cat sub/report.txt\"}<|/tool-call|>",
    ],
)
def test_these_do_not(reply):
    score, _ = grade(reply, TRUTH)
    assert score == 0.0


def test_the_answer_is_reported_whatever_the_score():
    """A run of zeroes is only diagnosable if you can see what was said."""
    _, answer = grade("<|par|>other.txt<|/par|>", TRUTH)
    assert answer == "other.txt"


def test_an_unclosed_block_is_dropped_with_what_it_holds():
    """The token cap cuts a turn mid-block; what is inside is still reasoning."""
    assert strip_markup("<|thinking|>sub/report.txt").strip() == ""


def test_normalisation_stops_at_the_workspace_prefix():
    assert normalize("/home/user/a.txt") == "a.txt"
    assert normalize("/etc/a.txt") == "etc/a.txt"


def test_extraction_takes_the_last_line_not_the_longest():
    assert extract("first.txt\nsecond.txt") == "second.txt"
