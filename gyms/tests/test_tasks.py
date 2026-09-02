"""What a generated task guarantees.

The point of every assertion here is the same one: a task whose answer is not
actually findable in the files it ships scores every model zero and looks like a
model problem.
"""

import pytest

from gyms.bash import tasks
from gyms.registry import REGISTRY

#: Every generator, so a new task type inherits every invariant below by
#: being added here rather than by anyone remembering to test it.
GENERATORS = [
    tasks.generate_ls_easy,
    tasks.generate_ls_medium,
    tasks.generate_ls_hard,
    tasks.generate_read_easy,
    tasks.generate_read_medium,
    tasks.generate_csv,
    tasks.generate_code,
    tasks.generate_find_easy,
    tasks.generate_find_medium,
    tasks.generate_grep,
]

#: What seed 5 produces for the three types that predate the levels.
#:
#: A run is reproducible from `(model, task_type, seed)`, and that promise is
#: only worth anything if a generator's *draw sequence* is stable: one extra
#: `rng` call anywhere in `_needle_task` re-answers every episode already on
#: record, while the run row still says `(bash_grep, 5)`. Nothing else in the
#: suite would notice — every other assertion here holds just as well for a
#: different task.
#:
#: These three are pinned rather than all ten because they are the ones with
#: history behind them and the ones now sharing a helper with new levels: the
#: risk this catches is a change made for `bash_find_easy` reaching `bash_grep`
#: through `_needle_task`.
FROZEN = {
    tasks.generate_grep: "gwash.txt",
    tasks.generate_csv: "VAL_8628",
    tasks.generate_code: "func_zwxuq",
}


@pytest.mark.parametrize("generate", GENERATORS)
def test_a_seed_generates_the_same_task_twice(generate):
    assert generate(11).stored() == generate(11).stored()


@pytest.mark.parametrize("generate", GENERATORS)
def test_different_seeds_generate_different_tasks(generate):
    assert generate(11).answer != generate(12).answer


@pytest.mark.parametrize("generate", GENERATORS)
def test_the_answer_is_in_the_files(generate):
    task = generate(5)
    assert task.answer
    assert any(task.answer in body or task.answer == name for name, body in task.files.items())


@pytest.mark.parametrize("generate", GENERATORS)
def test_the_prompt_never_contains_the_answer(generate):
    """A task whose prompt gives the answer away measures nothing."""
    task = generate(5)
    assert task.answer not in task.prompt


def test_the_grep_needle_appears_in_exactly_one_file():
    task = tasks.generate_grep(3)
    holding = [name for name, body in task.files.items() if task.target in body]
    assert holding == [task.answer]


def test_the_marked_function_is_the_only_one_marked():
    task = tasks.generate_code(3)
    mark = task.extra["mark"]
    body = task.files[task.extra["file"]]
    assert body.count(f"# {mark}") == 1
    marked = next(line for line in body.splitlines() if f"# {mark}" in line)
    assert task.answer in body[: body.index(marked)] + marked


def test_the_csv_row_the_prompt_names_holds_the_answer():
    task = tasks.generate_csv(3)
    rows = task.files[task.extra["file"]].splitlines()
    named = next(row for row in rows if row.startswith(f"{task.extra['row_id']},"))
    assert named.endswith(task.answer)


def test_an_open_task_never_carries_its_answer():
    """`public()` is what a route may serve while an episode is still running."""
    task = tasks.generate_grep(1)
    assert task.answer not in str(task.public())


def test_every_registered_task_type_generates():
    for gym in REGISTRY.listing():
        for task_type in gym["task_types"]:
            assert REGISTRY.generate(task_type["id"], 1).task_type == task_type["id"]


def test_an_unknown_task_type_is_refused_rather_than_defaulted():
    with pytest.raises(ValueError):
        REGISTRY.generate("bash_nothing", 1)


@pytest.mark.parametrize("generate", GENERATORS)
def test_a_prompt_stays_short_enough_to_be_asked(generate):
    """The budget is real and it is small.

    `build_prompt` reserves half a model's window for the reply, so a model
    trained at 256 has ~128 tokens for the system block and the task together,
    and the shell's first answer has to fit after that. A wordy prompt is not a
    style problem here — it is a question the model is only shown part of, which
    reads afterwards as a model that scored zero.

    Characters rather than tokens because a task has no tokenizer: at roughly
    three characters a token this is ~25 tokens, which leaves room for the
    system block and an observation inside a 128-token allowance.
    """
    assert len(generate(5).prompt) <= 80


def test_the_system_prompt_stays_short_too():
    assert len(tasks.SYSTEM_PROMPT) <= 100


def test_the_shaping_target_is_never_the_answer():
    """`target` is matched against the model's *command*, so a target equal to
    the answer pays a model for typing a guess it could only have by having
    already found it. Every `bash_csv` task in the first runs of this gym had
    exactly that shape."""
    for generate in GENERATORS:
        for seed in range(20):
            task = generate(seed)
            assert task.target, f"{task.task_type} has no shaping target"
            assert task.target != task.answer, f"{task.task_type} pays for its own answer"


def test_the_evidence_string_is_what_the_shell_prints_when_the_answer_is_found():
    """`evidence` is matched against output, so it has to actually occur in the
    files an episode is seeded with — otherwise `SAW_TARGET` can never fire,
    which is the failure `ON_TARGET` already had."""
    for generate in GENERATORS:
        for seed in range(20):
            task = generate(seed)
            assert task.evidence, f"{task.task_type} has no evidence string"
            # A filename counts as printable too: the `bash_ls` family hides
            # nothing in a body — its answer is the name itself, and `ls` is the
            # command that puts it on screen.
            printable = any(task.evidence in body for body in task.files.values())
            assert printable or task.evidence in task.files, (
                f"{task.task_type} evidence {task.evidence!r} is neither in a "
                f"file nor the name of one, so SAW_TARGET can never fire"
            )


@pytest.mark.parametrize("generate", GENERATORS)
def test_a_type_asks_more_than_one_question(generate):
    """Behaviour cloning is what makes this load-bearing rather than cosmetic.

    A generator with one prompt template varies the haystack and never the
    question — `bash_code` produced a byte-identical prompt for every seed it
    was ever asked for. Thousands of expert trajectories over that teach three
    sentences, and a model that has memorised the template reads afterwards as
    a model that solved the gym.
    """
    prompts = {generate(seed).prompt for seed in range(60)}
    assert len(prompts) >= 8, f"{generate.__name__} asks too few distinct questions"


#: Task types whose prompt names no file, because the file is what is being
#: asked for. Everything else has to name one it actually ships.
NAMES_NO_FILE = {
    "bash_grep", "bash_find_easy", "bash_find_medium",
    "bash_ls_easy", "bash_ls_medium", "bash_ls_hard",
}


@pytest.mark.parametrize("generate", GENERATORS)
def test_the_file_a_prompt_names_is_a_file_the_task_ships(generate):
    """A prompt naming `data.csv` over a workspace holding `table.csv` is a
    task nothing can solve, and it is the failure varying the filename invites."""
    for seed in range(40):
        task = generate(seed)
        for name in task.files:
            if name in task.prompt:
                break
        else:
            assert task.task_type in NAMES_NO_FILE, f"{task.prompt!r} names no shipped file"


@pytest.mark.parametrize("generate,answer", list(FROZEN.items()), ids=lambda v: str(v)[:20])
def test_a_type_with_history_still_draws_the_same_task(generate, answer):
    """See `FROZEN`. The three pre-level generators share `_needle_task` and
    their pools with the new levels now, so this is what stands between a tweak
    made for `bash_find_easy` and every `bash_grep` episode on record quietly
    meaning something else."""
    assert generate(5).answer == answer


def test_the_only_file_with_the_answer_extension_is_the_answer():
    """`bash_ls` asks "exactly one file is a .log" and that has to be true — a
    second one makes the task unanswerable and every model wrong for it."""
    for generate in (tasks.generate_ls_easy, tasks.generate_ls_medium):
        for seed in range(40):
            task = generate(seed)
            wearing = [name for name in task.files if name.endswith(task.target)]
            assert wearing == [task.answer], f"{task.task_type} seed {seed}: {wearing}"


def test_the_only_filename_holding_the_mark_is_the_answer():
    """The same promise for `bash_ls_hard`, where the marker is in the name.

    It holds without a rejection loop because every mark in `LS_MARKS` carries a
    digit and `_rand_filename` draws from lowercase letters alone — this is the
    test that would catch a mark being added without one.
    """
    for seed in range(40):
        task = tasks.generate_ls_hard(seed)
        holding = [name for name in task.files if task.target in name]
        assert holding == [task.answer], f"seed {seed}: {holding}"


def test_the_key_a_pairs_prompt_names_appears_once():
    for seed in range(40):
        task = tasks.generate_read_medium(seed)
        lines = task.files[task.extra["file"]].splitlines()
        named = [line for line in lines if line.startswith(f"{task.extra['key']} ")]
        assert len(named) == 1, f"seed {seed}: {named}"
        assert named[0].endswith(task.answer)


def test_an_easy_task_is_smaller_than_its_hard_sibling():
    """The levels have to be a ladder rather than three labels.

    Not a proxy for difficulty in general — a level is really "how many shell
    skills does this need" — but within a family the haystack is what carries
    that, and a medium level that shipped fewer files than its easy one would be
    mislabelled in the run form, which is where somebody picks what to run next.
    """
    for family in ("find", "ls"):
        sizes = [
            (t.level, sum(len(REGISTRY.generate(t.id, seed).files) for seed in range(20)))
            for t in _family(family)
        ]
        assert sizes == sorted(sizes, key=lambda pair: pair[1]), f"{family}: {sizes}"


def _family(name: str) -> list:
    gym = next(g for g in REGISTRY.listing() if g["id"] == "bash")
    types = [t for t in gym["task_types"] if t["family"] == name]
    return [REGISTRY.task_type(t["id"]) for t in types]


def test_every_task_type_declares_a_level_and_a_family():
    """The listing is what the run form groups by, so a task type that arrives
    without either is one nobody can find among the ten."""
    from gyms.registry import LEVELS

    for gym in REGISTRY.listing():
        for task_type in gym["task_types"]:
            assert task_type["level"] in LEVELS, task_type
            assert task_type["family"], task_type


def test_every_family_offers_every_level():
    """A family missing its easy rung is the shape this whole change exists to
    fix: nowhere to start, and nothing between zero and the hard task."""
    from gyms.registry import LEVELS

    gym = next(g for g in REGISTRY.listing() if g["id"] == "bash")
    families = {t["family"] for t in gym["task_types"]}
    for family in families:
        levels = {t["level"] for t in gym["task_types"] if t["family"] == family}
        assert levels == set(LEVELS), f"{family} offers {levels}"
