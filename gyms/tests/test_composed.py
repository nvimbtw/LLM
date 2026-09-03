"""Contracts for the generated primitive and pipeline tasks."""

import pytest

from gyms.bash import composed, expert
from gyms.bash.grade import mutation_matches


GENERATORS = tuple(composed.GENERATORS.items())


@pytest.mark.parametrize("name,generate", GENERATORS)
def test_composed_tasks_vary_and_stay_short(name, generate):
    prompts = {generate(seed).prompt for seed in range(60)}
    assert len(prompts) >= 50, name
    assert all(len(generate(seed).prompt) <= 80 for seed in range(60))


@pytest.mark.parametrize("name,generate", GENERATORS)
def test_composed_expert_command_has_declared_pipeline_shape(name, generate):
    task = generate(31)
    command = expert.solve(task).commands[0]
    stages = task.extra.get("pipeline_stages")
    if stages is not None:
        assert command.count("|") == stages - 1, (name, command)


def test_mutation_requires_the_written_content_and_a_successful_read():
    assert mutation_matches("KEY=value\n", 0, "KEY=value")
    assert not mutation_matches("KEY=value\n", 1, "KEY=value")
    assert not mutation_matches("KEY=other\n", 0, "KEY=value")


def test_edit_tasks_record_the_expected_file_change():
    for generate in (
        composed.generate_write_easy,
        composed.generate_pipe_write_medium,
        composed.generate_pipe_transform_hard,
        composed.generate_pipe_discover_edit_hard,
    ):
        task = generate(31)
        assert task.extra["mutation"]["path"]
        assert task.extra["mutation"]["contains"]
