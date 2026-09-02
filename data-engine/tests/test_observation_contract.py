"""The shell-result layout, pinned against the renderer that actually serves it.

`processors/hermes.py::_result_body` says it "mirrors
`backend/crates/tools/src/bash.rs::render` deliberately and closely". That was a
comment, and a comment does not fail when somebody changes the Rust. This makes
it fail.

It matters more here than the wording suggests. A tool result rendered one way
in training and another way at serving is invisible everywhere — every test
passes, the corpus looks fine, the fine-tune converges — until a model meets the
real thing and reads a block it was never taught. `contracts/shell-observation.json`
is generated from the Rust side, which is the authority because it is what the
model is actually served.

The same bargain `test_auth.py` strikes with the token seam, for the same reason.
"""

import json
from pathlib import Path

import pytest

from processors.hermes import _result_body

CONTRACT = Path(__file__).resolve().parents[2] / "contracts" / "shell-observation.json"
CASES = json.loads(CONTRACT.read_text())["cases"]


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_a_recorded_result_renders_as_the_live_tool_would(case):
    assert _result_body(case["output"], case["exit_code"], case["budget"]) == case["expected"]


def test_the_vector_covers_the_branches_that_can_disagree():
    """A fixture nobody extends is a fixture that pins the easy half. These are
    the four places the two implementations could drift apart without either
    looking wrong on its own: a silent command, a marker in the output, and both
    ways a budget can cut."""
    names = {case["name"] for case in CASES}
    assert {"silence on success", "output carrying a marker"} <= names
    assert {"cut on a line boundary", "cut with no boundary to use"} <= names
    # Rust indexes the cut by chars and then searches for the newline by *byte*
    # offset, where Python does both in characters. They agree on this case;
    # they are not obviously required to agree on every one.
    assert "cut in the middle of non-ascii" in names
