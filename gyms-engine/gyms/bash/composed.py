"""Generated shell compositions built from small, reusable primitives.

The original tasks vary the haystack. These tasks vary the shell program too:
names, markers, values, layouts and short prompt fragments all come from the
seed. A task is therefore reproducible for a run, while the space of tasks is
large enough that memorising a handful of commands is not a useful strategy.
"""

import random
import string

from gyms.bash.tasks import Task


WORDS = (
    "alpha", "amber", "cedar", "cobalt", "delta", "ember", "frost",
    "granite", "hazel", "indigo", "jade", "kilo", "linen", "maple",
    "navy", "ochre", "pearl", "quartz", "ruby", "sage", "teal", "umber",
)

WRITE_PROMPTS = (
    "Write {value} to {file}. Reply with the file.",
    "Put {value} in {file}; answer with its name.",
    "Create {file} containing {value}. Give its path.",
)
PIPE_PROMPTS = (
    "Find the {needle} file and reply with its path.",
    "Which file matches {needle}? Give its path.",
    "Locate {needle} and answer with the file path.",
)
VALUE_PROMPTS = (
    "Filter {file} for {key}; reply with the value.",
    "In {file}, find {key} and give its value.",
    "Read {file}: what value follows {key}?",
)
EDIT_PROMPTS = (
    "Copy the {key} line to {out}; reply with its value.",
    "Write the {key} entry into {out}. Give its value.",
    "Extract {key} into {out}; answer with the value.",
)


def _stem(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(6, 12)))


def _token(rng: random.Random, prefix: str) -> str:
    return f"{prefix}-{rng.randrange(16**8):08x}"


def _file(rng: random.Random, extension: str = "txt") -> str:
    return f"{_stem(rng)}.{extension}"


def _lines(rng: random.Random, count: int, chosen: str, value: str) -> str:
    keys = [_token(rng, "FIELD") for _ in range(count - 1)] + [chosen]
    rng.shuffle(keys)
    return "\n".join(
        f"{key}={value}" if key == chosen else f"{key}={rng.choice(WORDS)}"
        for key in keys
    ) + "\n"


def generate_write_easy(seed: int) -> Task:
    """Create one file with one redirection command."""
    rng = random.Random(seed)
    name = _file(rng)
    value = _token(rng, "NOTE")
    return Task(
        task_type="bash_write_easy",
        seed=seed,
        prompt=rng.choice(WRITE_PROMPTS).format(value=value, file=name),
        answer=name,
        files={},
        target=value,
        evidence=name,
        extra={
            "mutation": {"path": name, "contains": f"{value}\n"},
            "answer_visible_in_output": False,
            "value": value,
        },
    )


def generate_pipe_find_medium(seed: int) -> Task:
    """Find a marked filename with one discovery/filter pipeline."""
    rng = random.Random(seed)
    needle = _token(rng, "FILE")
    files = {_file(rng): "irrelevant\n" for _ in range(rng.randint(5, 10))}
    holder = f"{_stem(rng)}-{needle}.log"
    files[holder] = "irrelevant\n"
    return Task(
        task_type="bash_pipe_find_medium",
        seed=seed,
        prompt=rng.choice(PIPE_PROMPTS).format(needle=needle),
        answer=holder,
        files=files,
        target=needle,
        evidence=holder,
        extra={"pipeline_stages": 2},
    )


def generate_pipe_extract_medium(seed: int) -> Task:
    """Read and select a value with one pipe."""
    rng = random.Random(seed)
    name = _file(rng)
    key = _token(rng, "KEY")
    value = _token(rng, "VALUE")
    return Task(
        task_type="bash_pipe_extract_medium",
        seed=seed,
        prompt=rng.choice(VALUE_PROMPTS).format(file=name, key=key),
        answer=value,
        files={name: _lines(rng, rng.randint(5, 12), key, value)},
        target=key,
        evidence=value,
        extra={"file": name, "key": key, "pipeline_stages": 2},
    )


def generate_pipe_write_medium(seed: int) -> Task:
    """Filter a source and write the matching line with one pipe."""
    rng = random.Random(seed)
    source = _file(rng)
    output = _file(rng)
    key = _token(rng, "KEY")
    value = _token(rng, "VALUE")
    return Task(
        task_type="bash_pipe_write_medium",
        seed=seed,
        prompt=rng.choice(EDIT_PROMPTS).format(key=key, out=output),
        answer=value,
        files={source: _lines(rng, rng.randint(5, 12), key, value)},
        target=key,
        evidence=value,
        extra={
            "key": key,
            "source": source,
            "output": output,
            "mutation": {"path": output, "contains": f"{key}={value}"},
            "pipeline_stages": 2,
        },
    )


def generate_pipe_search_hard(seed: int) -> Task:
    """Search, extract paths and deduplicate with three commands."""
    rng = random.Random(seed)
    needle = _token(rng, "NEEDLE")
    files = {_file(rng, "log"): "quiet\n" for _ in range(rng.randint(8, 18))}
    holder = _file(rng, "log")
    files[holder] = f"quiet\n{needle}\n{needle}\n"
    return Task(
        task_type="bash_pipe_search_hard",
        seed=seed,
        prompt=rng.choice(PIPE_PROMPTS).format(needle=needle),
        answer=holder,
        files=files,
        target=needle,
        evidence=holder,
        extra={"pipeline_stages": 3},
    )


def generate_pipe_transform_hard(seed: int) -> Task:
    """Read, select, extract and write a value with three commands."""
    rng = random.Random(seed)
    source = _file(rng)
    output = _file(rng)
    key = _token(rng, "KEY")
    value = _token(rng, "VALUE")
    return Task(
        task_type="bash_pipe_transform_hard",
        seed=seed,
        prompt=rng.choice(EDIT_PROMPTS).format(key=key, out=output),
        answer=value,
        files={source: _lines(rng, rng.randint(8, 18), key, value)},
        target=key,
        evidence=value,
        extra={
            "key": key,
            "source": source,
            "output": output,
            "mutation": {"path": output, "contains": value},
            "pipeline_stages": 4,
        },
    )


def generate_pipe_discover_edit_hard(seed: int) -> Task:
    """Discover a file, read its match and write the result."""
    rng = random.Random(seed)
    needle = _token(rng, "FILE")
    marker = _token(rng, "MARK")
    output = _file(rng)
    files = {_file(rng): "nothing useful\n" for _ in range(rng.randint(6, 12))}
    holder = f"{_stem(rng)}-{needle}.txt"
    files[holder] = f"header\n{marker}\n"
    return Task(
        task_type="bash_pipe_discover_edit_hard",
        seed=seed,
        prompt=f"Find {needle}, copy its {marker} line to {output}.",
        answer=marker,
        files=files,
        target=needle,
        evidence=marker,
        extra={
            "needle": needle,
            "marker": marker,
            "output": output,
            "mutation": {"path": output, "contains": marker},
            "pipeline_stages": 4,
        },
    )


GENERATORS = {
    "bash_write_easy": generate_write_easy,
    "bash_pipe_find_medium": generate_pipe_find_medium,
    "bash_pipe_extract_medium": generate_pipe_extract_medium,
    "bash_pipe_write_medium": generate_pipe_write_medium,
    "bash_pipe_search_hard": generate_pipe_search_hard,
    "bash_pipe_transform_hard": generate_pipe_transform_hard,
    "bash_pipe_discover_edit_hard": generate_pipe_discover_edit_hard,
}
