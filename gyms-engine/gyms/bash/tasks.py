"""The tasks a bash episode is built from.

Generated rather than curated, and generated from a seed: a run is reproducible
from `(model, task_type, seed)` alone, which is what makes two runs of the same
model comparable and a regression bisectable. Nothing is stored on disk, so
adding a task kind is a function here and a row in `gyms.registry`.

Every kind hides one short, unambiguous string somewhere a shell can reach it,
and asks for that string back. That is deliberate: the thing being measured is
whether the model can *drive a shell to find something*, and a task whose answer
needs reasoning about the contents would measure two things and attribute the
failure to the wrong one.

## Three families, three levels each, and why the levels exist

The first ten runs of this gym scored 0.00 across 105 episodes, and the database
says why. The model drove the shell competently — 127 commands exited clean —
but it almost never *read* anything: over those 105 episodes it typed `grep`
four times and `cat` ten. In the 28 `bash_code` episodes, whose workspace holds
one 146-byte file, the answer reached the screen **zero** times; the model
listed the directory, saw `code.py`, and listed it again. Meanwhile `bash_grep`
put the answer on screen in 13 of 41 episodes — a listing prints filenames —
and still scored 0/41.

So the three original types all demanded the same four skills at once: locate a
file, open it, pick a string out of it, and end the turn on that string alone.
A model that can do the first and not the second scores exactly what a model
that can do neither scores, which is a benchmark with no gradient in it.

The families below each isolate one of those skills at their easy level and add
one per level:

```
bash_ls      the answer is in the listing        ls  ->  find      ->  find -name
bash_read    the answer is in a file it names    cat ->  cat/grep  ->  the old csv/code
bash_find    the answer is which file holds it   grep over 3 files -> 8-15 -> the old grep
```

`bash_grep`, `bash_csv` and `bash_code` are the hard tier of two of those
families and keep their ids and their generators **byte for byte**: a run is
reproducible from `(model, task_type, seed)`, so a generator that draws one more
random number makes every episode already on record unreproducible. That is why
`_needle_task` below is a refactor with `nested=True` as its old behaviour
rather than a rewrite, and why `tests/test_tasks.py` pins a golden answer for
each of the three.

The filler is lorem rather than English prose so a model cannot answer from the
language model alone — there is nothing in `consectetur adipiscing` that
predicts which file the needle landed in.
"""

import random
import string
from dataclasses import dataclass, field

from gyms.sandbox import WORKSPACE

#: What the model is told it is doing, ahead of the task itself.
#:
#: **Kept short on purpose, and the reason is arithmetic.** Every token here is
#: a token the question cannot have: `build_prompt` reserves half a small
#: model's window for the reply, so a model trained at 256 has ~128 tokens for
#: the system block *and* the task, and one trained at 64 has ~32 for both. A
#: paragraph of instructions would spend the whole budget saying how to answer a
#: question that then gets truncated away.
#:
#: Two things it has to say and cannot leave out: that the shell is reached by
#: calling a tool, and that the episode ends when the model answers bare —
#: `gyms.bash.grade` is forgiving about the second, but a model that says the
#: answer and nothing else has ended its own episode cleanly.
#:
#: It names no directory, because it no longer has to: the files are in the one
#: the shell opens in (`gyms.sandbox.WORKSPACE`), so `ls` finds them.
SYSTEM_PROMPT = "Use the bash tool to look around. Reply with the answer alone."

LOREM_WORDS = [
    "lorem", "ipsum", "dolor", "sit", "amet", "consectetur", "adipiscing", "elit",
    "sed", "do", "eiusmod", "tempor", "incididunt", "labore", "dolore", "magna",
    "aliqua", "enim", "minim", "veniam", "quis", "nostrud", "exercitation",
    "ullamco", "laboris", "nisi", "aliquip", "ex", "ea", "commodo", "consequat",
]

#: File-count bounds for the grep task. The floor is what makes `ls` alone
#: insufficient and a search actually necessary; the ceiling keeps a listing
#: inside the output budget the sandbox clamps to.
GREP_FILES = (10, 50)
CSV_ROWS = (10, 30)
CODE_DECOYS = (2, 5)

#: **Every generator draws its surface from a pool rather than an f-string.**
#:
#: The reason is behaviour cloning, and it is worth stating because the old
#: shape looked fine: one prompt template per type meant a seed varied the
#: haystack and never the question, so `bash_code` produced a byte-identical
#: prompt for every seed it was ever asked for. A few thousand expert
#: trajectories built on that would have taught three sentences — the model
#: learns the template and answers the next one from memory, which reads as a
#: model that solved the gym.
#:
#: Every phrasing must stay inside the ~80-character budget
#: (`tests/test_tasks.py`), name the file where the task names one, and never
#: contain the answer.
GREP_PROMPTS = (
    "Which file contains {needle}? Answer with its path.",
    "Find {needle}. Reply with the path of the file holding it.",
    "One file mentions {needle}. Which one? Give its path.",
    "Search the files for {needle} and answer with that file's path.",
)
#: The needle's *spelling* varies too, so a model cannot key on the literal
#: `ANSWER_` and skip reading the question.
GREP_NEEDLES = ("ANSWER_{hex}", "TOKEN-{hex}", "KEY_{hex}", "MARK{hex}")

CSV_PROMPTS = (
    "In {file}, what is the value of the row with id {row}?",
    "Read {file}. What value does row id {row} carry?",
    "{file} has a row with id {row}. What is its value?",
    "Look up id {row} in {file} and answer with its value.",
)
CSV_FILES = ("data.csv", "records.csv", "table.csv", "rows.csv")
#: Kept distinctive rather than a bare integer: a plain number would collide
#: with the filler in the other rows, and `grade` compares a whole last line.
CSV_VALUES = ("VAL_{n}", "RES-{n}", "V{n}x", "code{n}")

CODE_PROMPTS = (
    "In {file}, what is the name of the function marked {mark}?",
    "Which function in {file} is marked {mark}? Name it.",
    "{file} marks one function {mark}. What is it called?",
    "Find the function tagged {mark} in {file} and give its name.",
)
CODE_FILES = ("code.py", "main.py", "utils.py", "helpers.py")
CODE_MARKS = ("TARGET", "FIXME", "REVIEW", "CHECK")

#: `bash_ls`: the answer is a filename, so the listing the model already
#: produces contains it. The extension is what the prompt names and therefore
#: what the shaping may pay for; it is never the answer.
LS_PROMPTS = (
    "One file here ends in .{ext}. Name it.",
    "Which file has the .{ext} extension? Give its name.",
    "Exactly one file is a .{ext} file. Which one?",
    "Find the .{ext} file and reply with its name.",
)
#: The nested levels ask for a path rather than a name, because the answer has a
#: directory in front of it and `grade` compares a whole last line.
LS_NESTED_PROMPTS = (
    "One file below here ends in .{ext}. Give its path.",
    "Which file has the .{ext} extension? Reply with its path.",
    "Exactly one file is a .{ext} file. Give its path.",
    "Find the .{ext} file and answer with its path.",
)
#: Split into two pools so a task can pick an answer extension the crowd cannot
#: also be wearing — "exactly one file is a .log" has to be true. **The two must
#: stay disjoint**, and that is the whole invariant: a filler extension that
#: also appears here makes the question unanswerable, which
#: `test_the_only_file_with_the_answer_extension_is_the_answer` refuses.
#:
#: Fifteen rather than the seven this started with, because the extension is
#: what the question varies and seven is few enough to become a reflex: a model
#: shown `.log` in a seventh of every batch can learn "answer with the odd one"
#: as "answer with the `.log` one", which looks identical until it meets an
#: extension it was never trained on.
LS_ANSWER_EXTENSIONS = (
    "log", "json", "yaml", "ini", "sql", "toml", "cfg", "env",
    "csv", "tsv", "xml", "conf", "lock", "bak", "tmp",
)
LS_FILLER_EXTENSIONS = ("txt", "md", "py")
LS_FILES = (6, 9)
LS_NESTED_FILES = (14, 24)

LS_MARK_PROMPTS = (
    "One filename contains {mark}. Give its path.",
    "Which file has {mark} in its name? Reply with the path.",
    "Find the file whose name holds {mark}. Give its path.",
    "A filename includes {mark}. Which file? Give its path.",
)
#: Every mark carries a digit, and `_rand_filename` draws from lowercase letters
#: alone — so a crowd filename can never collide with the mark by accident, and
#: "one filename contains it" stays true without a rejection loop. **A mark
#: added without a digit breaks that silently**; the guard is
#: `test_the_only_filename_holding_the_mark_is_the_answer`.
LS_MARKS = (
    "v7", "x9", "k3", "q8", "b2", "m5", "z4", "j6", "r1", "d0",
    "t3x", "p2n", "w8k", "g5q", "n4b", "c7v", "h9m", "s6z", "f1r", "y0d",
)
LS_MARK_FILES = (20, 40)

#: `bash_read`: the prompt names the file, so nothing has to be located. The
#: easy level's file is one line long, which makes `cat` sufficient — the one
#: command the measured runs never issued.
READ_PROMPTS = (
    "Read {file} and reply with the token it holds.",
    "{file} holds one token. What is it?",
    "What token is in {file}? Reply with it.",
    "Open {file} and give the token inside.",
)
#: Twenty filenames and four extensions rather than five files all named
#: `*.txt`, and this is the pool that most needed it.
#:
#: `bash_read_easy` is the narrowest task in the gym by design — one file, one
#: `cat` — which leaves the filename as almost the only thing a seed varies. At
#: five names, a few thousand demonstrations show `cat note.txt` some hundreds
#: of times, and the habit that gets cloned is `cat note.txt` rather than "read
#: the file the question names". That failure is invisible in evaluation right
#: up until a filename the model was never trained on, because the training and
#: the evaluation draw from the same five.
READ_FILES = (
    "note.txt", "token.txt", "value.txt", "answer.txt", "secret.txt",
    "key.txt", "label.txt", "marker.txt", "handle.txt", "ticket.txt",
    "id.dat", "serial.dat", "record.dat", "entry.dat", "slug.dat",
    "badge.val", "stamp.val", "ref.val", "passcode.key", "signature.key",
)
READ_TOKENS = ("TOK_{n}", "ID-{n}", "T{n}z", "key{n}")

PAIRS_PROMPTS = (
    "In {file}, what value does {key} have?",
    "Read {file} and give the value of {key}.",
    "{file} sets {key}. To what?",
    "Look up {key} in {file} and reply with its value.",
)
PAIRS_FILES = ("config.txt", "settings.ini", "env.txt", "params.txt")
#: Deliberately longer than `PAIRS_LINES` can ask for, so every line of a
#: generated file is a plausible key. A pool topped up with random stems would
#: put `nfeickj = 41` next to `timeout = 90`, which tells the model which line
#: is the interesting one before it has read the question.
PAIRS_KEYS = (
    "timeout", "region", "bucket", "shard", "retries", "mode", "backend",
    "replicas", "driver", "cursor", "endpoint", "quota", "workers", "prefix",
    "encoding", "level", "adapter", "window", "channel", "profile", "cache",
    "limit", "origin", "scheme", "tier", "queue",
)
PAIRS_LINES = (12, 24)

#: The flat, small end of the needle family. The floor is what keeps a search
#: cheaper than reading every file by hand; three files is deliberately below
#: the point where that is true, because at the easy level the skill being
#: measured is reading files at all rather than searching them efficiently.
FIND_EASY_FILES = (3, 4)
FIND_MEDIUM_FILES = (8, 15)


@dataclass(frozen=True)
class Task:
    """One episode's worth of work.

    `answer` is the ground truth and is the one field that must never reach a
    browser before the episode closes — see `gyms.api.routes.gym`, which builds
    every open-episode response out of `public()` rather than out of this.
    """

    task_type: str
    seed: int
    prompt: str
    answer: str
    files: dict[str, str]
    system: str = SYSTEM_PROMPT
    #: What the model should be *searching for*, and the one thing the shaping
    #: may pay it for typing. It must be derivable from the prompt — a model can
    #: only aim at something it has been told — and it must never be the answer:
    #: paying for the answer appearing in a command pays a model for typing
    #: something it could only know by having already found it.
    target: str = ""
    #: What the model has *seen* once this string appears in a command's output:
    #: the answer is on screen. A separate field from `target` because the two
    #: are only the same string when the needle is both what you search for and
    #: what proves you found it — true for grep and code, false for the CSV,
    #: whose needle is a filename anything would print.
    #:
    #: Unlike `target` this may be the answer itself, and for the CSV it is:
    #: it is matched against what the *shell* printed, so a model cannot earn it
    #: by typing a guess.
    evidence: str = ""
    extra: dict = field(default_factory=dict)

    def stored(self) -> dict:
        """The whole task, as it is written to `gym_episodes.task_config`."""
        return {
            "task_type": self.task_type,
            "seed": self.seed,
            "prompt": self.prompt,
            "answer": self.answer,
            "files": self.files,
            "system": self.system,
            "target": self.target,
            "evidence": self.evidence,
            **self.extra,
        }

    def public(self) -> dict:
        """What a caller may see while the episode is still open."""
        return {
            "task_type": self.task_type,
            "seed": self.seed,
            "prompt": self.prompt,
            "system": self.system,
            "file_count": len(self.files),
        }


def _rand_text(rng: random.Random, words: int = 20) -> str:
    return " ".join(rng.choice(LOREM_WORDS) for _ in range(words))


def _rand_stem(rng: random.Random) -> str:
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(5, 10)))


def _rand_filename(rng: random.Random, nested: bool = True) -> str:
    """A filename, a third of the time inside a one-level subdirectory.

    `nested=False` draws the same name from the same number of `rng` calls and
    then keeps it flat, so a level that wants a flat workspace does not have to
    reject-and-retry — a rejection loop would consume a variable number of draws
    and make the task unreproducible from its seed.
    """
    name = _rand_stem(rng)
    ext = rng.choice(["txt", "csv", "py", "log", "md"])
    sub = "".join(rng.choice(string.ascii_lowercase) for _ in range(4)) if rng.random() < 0.3 else ""
    if sub and nested:
        return f"{sub}/{name}.{ext}"
    return f"{name}.{ext}"


def _under(rng: random.Random, name: str, nested: bool) -> str:
    """`name`, in a fresh one-level subdirectory when the level wants nesting.

    Always a subdirectory rather than sometimes one, unlike `_rand_filename`:
    the `bash_ls` levels are a ladder from `ls` to a recursive listing, and a
    nested level whose files landed flat by chance would be its easy level
    wearing a harder label.
    """
    if not nested:
        return name
    sub = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
    return f"{sub}/{name}"


def _needle_task(seed: int, task_type: str, count: tuple[int, int], nested: bool) -> Task:
    """One needle across `count` files; the file holding it is the answer.

    The whole `bash_find` family, levels and all, because the levels differ only
    in how much haystack there is and whether any of it is in a subdirectory.

    **The order of the `rng` calls below is load-bearing.** `generate_grep` is
    this function with the original bounds and `nested=True`, and episodes on
    record were generated by the code it replaced — a draw added, removed or
    reordered here silently changes what every historical seed produces.
    `tests/test_tasks.py` pins seed 5's answer against that.
    """
    rng = random.Random(seed)
    low, high = count
    target = rng.choice(GREP_NEEDLES).format(hex=f"{rng.randint(0, 0xFFFFFF):06x}")
    wanted = rng.randint(low, high)
    files: dict[str, str] = {}
    while len(files) < wanted:
        files[_rand_filename(rng, nested)] = _rand_text(rng, rng.randint(10, 30))
    chosen = rng.choice(sorted(files))
    files[chosen] += f"\n{target}\n"
    return Task(
        task_type=task_type,
        seed=seed,
        prompt=rng.choice(GREP_PROMPTS).format(needle=target),
        answer=chosen,
        files=files,
        target=target,
        evidence=target,
    )


def generate_find_easy(seed: int) -> Task:
    """Three or four flat files, one needle. `cat *` is enough to solve it."""
    return _needle_task(seed, "bash_find_easy", FIND_EASY_FILES, nested=False)


def generate_find_medium(seed: int) -> Task:
    """Enough files that reading them all is worse than searching, still flat."""
    return _needle_task(seed, "bash_find_medium", FIND_MEDIUM_FILES, nested=False)


def generate_grep(seed: int) -> Task:
    """A needle in one of many files, some in subdirectories; the hard level."""
    return _needle_task(seed, "bash_grep", GREP_FILES, nested=True)


def generate_csv(seed: int) -> Task:
    """A cell at a stated row of a CSV; the cell's value is the answer.

    The answer stays the **last** column whatever else varies: `grade` compares
    a whole last line, so a value with a column after it would be an answer no
    reply can spell.
    """
    rng = random.Random(seed)
    low, high = CSV_ROWS
    n_rows = rng.randint(low, high)
    target_row = rng.randint(1, n_rows)
    name = rng.choice(CSV_FILES)
    answer = rng.choice(CSV_VALUES).format(n=rng.randint(1000, 9999))
    rows = ["id,name,value"]
    for i in range(1, n_rows + 1):
        if i == target_row:
            rows.append(f"{i},target,{answer}")
        else:
            rows.append(f"{i},{_rand_text(rng, 1)},{rng.randint(10, 999)}")
    return Task(
        task_type="bash_csv",
        seed=seed,
        prompt=rng.choice(CSV_PROMPTS).format(file=name, row=target_row),
        answer=answer,
        files={name: "\n".join(rows) + "\n"},
        # The file, not the value. `target` used to be `answer` here, which paid
        # the model to type its own guess into a command — the one thing this
        # field's docstring forbids — and it was the only task type where the
        # two coincided.
        target=name,
        evidence=answer,
        extra={"row_id": target_row, "file": name},
    )


def generate_code(seed: int) -> Task:
    """One marked function among decoys; the function's name is the answer."""
    rng = random.Random(seed)
    name = rng.choice(CODE_FILES)
    mark = rng.choice(CODE_MARKS)
    answer = f"func_{''.join(rng.choice(string.ascii_lowercase) for _ in range(5))}"
    low, high = CODE_DECOYS
    names = [
        "".join(rng.choice(string.ascii_lowercase) for _ in range(6))
        for _ in range(rng.randint(low, high))
    ] + [answer]
    rng.shuffle(names)
    body = "\n\n".join(
        f"def {fn}(x):\n    return x * 2  # {mark}" if fn == answer
        else f"def {fn}(x):\n    return x + 1"
        for fn in names
    )
    return Task(
        task_type="bash_code",
        seed=seed,
        prompt=rng.choice(CODE_PROMPTS).format(file=name, mark=mark),
        answer=answer,
        files={name: body + "\n"},
        target=mark,
        evidence=mark,
        extra={"file": name, "mark": mark},
    )


def _listing_task(
    seed: int, task_type: str, count: tuple[int, int], nested: bool, prompts: tuple[str, ...]
) -> Task:
    """One file wearing an extension none of the others wears; its name is the
    answer, and a listing is all it takes to read it off.

    The easy end of the whole gym, and it is aimed at something specific the
    measured runs showed: the model's first command is nearly always `ls -la` or
    `find . -name '*'`, and it repeats it rather than acting on it. Here that
    command *is* the work, so the only thing left to fail at is naming one line
    of the output — which is the failure worth isolating, since `bash_grep` put
    the answer on screen 13 times out of 41 and scored zero every time.

    `evidence` is the answer, which is allowed and unfarmable for the usual
    reason — it is matched against output, and the shell has to print it — but
    it is a weaker signal here than elsewhere, because a bare `ls` prints every
    filename and so prints this one. `target` is the extension: the thing the
    prompt names, and the thing a command that actually *filters* contains.
    """
    rng = random.Random(seed)
    ext = rng.choice(LS_ANSWER_EXTENSIONS)
    low, high = count
    wanted = rng.randint(low, high)
    files: dict[str, str] = {}
    while len(files) < wanted:
        crowd = _rand_stem(rng) + "." + rng.choice(LS_FILLER_EXTENSIONS)
        files[_under(rng, crowd, nested)] = _rand_text(rng, rng.randint(8, 20))
    holder = _under(rng, f"{_rand_stem(rng)}.{ext}", nested)
    files[holder] = _rand_text(rng, rng.randint(8, 20))
    return Task(
        task_type=task_type,
        seed=seed,
        prompt=rng.choice(prompts).format(ext=ext),
        answer=holder,
        files=files,
        target=f".{ext}",
        evidence=holder,
    )


def generate_ls_easy(seed: int) -> Task:
    """A flat handful of files, one with a unique extension. `ls` solves it."""
    return _listing_task(seed, "bash_ls_easy", LS_FILES, False, LS_PROMPTS)


def generate_ls_medium(seed: int) -> Task:
    """The same question with the files in subdirectories: `ls` is no longer
    enough and the listing has to be recursive."""
    return _listing_task(seed, "bash_ls_medium", LS_NESTED_FILES, True, LS_NESTED_PROMPTS)


def generate_ls_hard(seed: int) -> Task:
    """The marker is in a *filename* rather than an extension, among enough
    files that reading the listing by eye stops working: the step up is from
    recursing to filtering, `find . -name '*mark*'`.

    Nothing in the files matters, which is what keeps this a different skill
    from `bash_find` — that family searches contents, this one searches names.
    """
    rng = random.Random(seed)
    mark = rng.choice(LS_MARKS)
    low, high = LS_MARK_FILES
    wanted = rng.randint(low, high)
    files: dict[str, str] = {}
    while len(files) < wanted:
        files[_rand_filename(rng)] = _rand_text(rng, rng.randint(8, 20))
    sub = "".join(rng.choice(string.ascii_lowercase) for _ in range(4))
    stem = _rand_stem(rng)
    holder = f"{sub}/{stem}{mark}.{rng.choice(LS_FILLER_EXTENSIONS)}"
    files[holder] = _rand_text(rng, rng.randint(8, 20))
    return Task(
        task_type="bash_ls_hard",
        seed=seed,
        prompt=rng.choice(LS_MARK_PROMPTS).format(mark=mark),
        answer=holder,
        files=files,
        target=mark,
        evidence=holder,
        extra={"mark": mark},
    )


def generate_read_easy(seed: int) -> Task:
    """One file, named by the prompt, holding one token and nothing else.

    The narrowest task in the gym, and the one the data most directly asks for:
    across 28 `bash_code` episodes over a workspace of exactly one file, the
    answer reached the screen zero times, because the model never typed `cat`.
    Everything except that one command is removed here — no search, no
    selection, no second file to be wrong about.
    """
    rng = random.Random(seed)
    name = rng.choice(READ_FILES)
    answer = rng.choice(READ_TOKENS).format(n=rng.randint(1000, 9999))
    return Task(
        task_type="bash_read_easy",
        seed=seed,
        prompt=rng.choice(READ_PROMPTS).format(file=name),
        answer=answer,
        files={name: f"{answer}\n"},
        target=name,
        evidence=answer,
        extra={"file": name},
    )


def generate_read_medium(seed: int) -> Task:
    """A named file of `key value` lines; the prompt names the key.

    `cat` still solves it — the file is short enough to fit an observation — but
    the answer now has to be picked out of what came back rather than being all
    of it. `grep {key}` is the step this level is teaching, and it is the same
    move `bash_read`'s hard tier (`bash_csv`) needs over a longer file.
    """
    rng = random.Random(seed)
    name = rng.choice(PAIRS_FILES)
    low, high = PAIRS_LINES
    n_lines = rng.randint(low, high)
    keys = list(PAIRS_KEYS)
    rng.shuffle(keys)
    keys = keys[:n_lines]
    key = rng.choice(keys)
    answer = rng.choice(CSV_VALUES).format(n=rng.randint(1000, 9999))
    lines = [
        f"{k} = {answer}" if k == key else f"{k} = {rng.randint(10, 999)}" for k in keys
    ]
    return Task(
        task_type="bash_read_medium",
        seed=seed,
        prompt=rng.choice(PAIRS_PROMPTS).format(file=name, key=key),
        answer=answer,
        files={name: "\n".join(lines) + "\n"},
        target=key,
        evidence=answer,
        extra={"file": name, "key": key},
    )
