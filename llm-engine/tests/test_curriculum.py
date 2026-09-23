import random

import pytest

from llm.config import PracticeSchedule
from llm.train.curriculum import Lessons

TASKS = [
    {"id": "add", "label": "Add", "family": "addition", "level": "easy", "prerequisites": []},
    {"id": "carry", "label": "Carry", "family": "addition", "level": "medium", "prerequisites": ["add"]},
    {"id": "multiply", "label": "Multiply", "family": "multiplication", "level": "easy", "prerequisites": []},
]


def test_two_checks_promote_one_skill_without_promoting_another():
    lessons = Lessons(TASKS, PracticeSchedule())
    lessons.observe("add", 95, 100, 0)
    assert lessons.status("add") == "practice"
    assert lessons.status("carry") == "locked"
    lessons.observe("add", 91, 100, 10)
    assert lessons.status("add") == "mastered"
    assert lessons.status("carry") == "practice"
    assert lessons.status("multiply") == "practice"
    assert lessons.events[-1]["correct"] == 91


def test_regression_returns_prerequisites_to_practice():
    lessons = Lessons(TASKS, PracticeSchedule())
    for key in ("add", "carry"):
        for step in (0, 10):
            lessons.observe(key, 100, 100, step)
    for step in (20, 30):
        lessons.observe("add", 50, 100, step)
    assert lessons.status("add") == "practice"
    assert lessons.status("carry") == "locked"
    assert not lessons.mastery["carry"].mastered
    lessons.observe("carry", 100, 100, 30)
    assert lessons.mastery["carry"].passes == 0


def test_sft_needed_and_state_survives_a_checkpoint():
    lessons = Lessons(TASKS, PracticeSchedule())
    for step in (0, 10):
        lessons.observe("multiply", 0, 100, step)
    lessons.cycle = 2
    lessons.training_groups = 20
    restored = Lessons(TASKS, PracticeSchedule(), lessons.snapshot())
    assert restored.snapshot() == lessons.snapshot()
    assert restored.status("multiply") == "needs_sft"


def test_sampling_keeps_review_and_frontier_and_some_stretch():
    lessons = Lessons(TASKS, PracticeSchedule())
    for step in (0, 10):
        lessons.observe("multiply", 100, 100, step)
    drawn = lessons.draw(random.Random(42), 10000)
    assert 5700 < drawn.count("add") < 6300
    assert 2700 < drawn.count("multiply") < 3300
    assert 800 < drawn.count("carry") < 1200


def test_incomplete_validation_and_missing_prerequisites_are_refused():
    lessons = Lessons(TASKS, PracticeSchedule())
    with pytest.raises(ValueError, match="incomplete"):
        lessons.observe("add", 1, 1, 0)
    with pytest.raises(ValueError, match="prerequisites"):
        Lessons(TASKS[1:], PracticeSchedule())


def test_diagnostics_survive_and_changed_generators_cannot_inherit_mastery():
    tasks = [{**task, "generator_version": 1} for task in TASKS]
    lessons = Lessons(tasks, PracticeSchedule())
    lessons.observe("add", 40, 100, 0, invalid_answers=12, truncated=7)
    restored = Lessons(tasks, PracticeSchedule(), lessons.snapshot())
    assert restored.report()[0]["invalid_answers"] == 12
    assert restored.report()[0]["truncated"] == 7
    changed = [{**task, "generator_version": 2} for task in tasks]
    with pytest.raises(ValueError, match="catalogue"):
        Lessons(changed, PracticeSchedule(), lessons.snapshot())
