# Gyms — where a model is put to work

A model is measured here by *doing* something rather than by predicting text:
it is given a shell, a workspace and a question it can only answer by looking
around, and what comes back is scored. One module, one database
(`slate_gyms`), one sandboxed filesystem per episode.

The module exists ahead of the reinforcement learning it is named for. Nothing
here trains anything yet — a run is an evaluation — but every part of it is the
part RL will need: generated tasks, a shell, a dense per-step reward, a terminal
score, and the whole trajectory on disk.

## Ownership, and who drives

`gyms/` owns `gym_runs`, `gym_episodes` and `gym_steps`, the task generators,
the sandbox seeding and the grading. The backend owns users; the engine owns
models.

**It does not own the loop.** A run is driven from
`backend/crates/api/src/gym.rs`, over a WebSocket the browser holds, and this
service serves one move of that loop per request. The reason is a single
sentence: what the model reads after a command is rendered by
`tools::observation` in Rust, the identical function the chat `bash` tool uses,
and a second renderer in Python would be a gym that teaches a layout serving
never shows. See
[`../decisions/gyms-are-driven-by-the-backend.md`](../decisions/gyms-are-driven-by-the-backend.md).

The service is spawned by the backend as a child (`GYMS_*`, port 8003) like
`engine`/`data-engine`, and its tokens carry `aud: gyms`. It talks to
[`just-bash`](../backend/README.md#the-sandboxed-shell) at `BASH_URL` and to nothing
else.

## A run, an episode, a step

```
run       one model against N generated tasks       gym_runs
 └ episode one task, one sandbox filesystem         gym_episodes
    └ step  one command and what it printed         gym_steps
```

A run exists because one episode says almost nothing: these tasks are pass or
fail, so ability is only visible across a batch. Episode *i* is generated from
`seed_base + i`, so a run is reproducible from its row alone — the same model,
seed and task type produce the same tasks every time.

**`gym_runs.model_name` holds the model's id, not its name.** The field is
named after the engine's own, which has always carried the id: weights live at
`artifacts/models/<id>/model.pt`, so a display name resolves to no path at all
and the run dies on its first round with `No weights at …`. Chat sends the same
thing under the same misleading key. The frontend turns it back into something
readable with `modelLabel` (`lib/gyms.ts`).

**The episode id is the sandbox session key.** One episode is one filesystem,
which is why the row is written before the workspace is seeded: the id has to
exist to key the files under, and a seeding failure rolls back the whole run
rather than handing out a healthy-looking episode over an empty directory.

**A task's files go in `/home/user`, because that is where the shell opens.**
Not a directory of our choosing: just-bash starts every command there, and —
the part that bites — *each command gets a fresh shell*, so the filesystem is
shared but the working directory is not. A `cd` in one step is gone by the next,
which means a model whose files are elsewhere has to spell an absolute path in
every command it ever writes. Seeding to `/workspace/files` cost exactly that:
the first model to run the gym typed `ls -la`, saw an empty `/home/user`, and
answered that the directory was empty and could it please be given a path. It
was not wrong. `tests/test_sandbox.py` pins this against a real sandbox — the
in-process fake the other tests use answers from wherever it is asked, so it can
never catch it.

## Three families, three levels each

The first ten runs of this gym scored **0.00 over 105 episodes**, and the
trajectories on record say why. The model drove the shell competently — 127
commands exited clean — but it barely *read* anything: across those 105
episodes it typed `grep` four times and `cat` ten. Its top commands were
`find . -name '*'` (40), `ls -la` (29) and `pwd` (7), often the same one three
steps running. In the 28 `bash_code` episodes, whose workspace holds a single
146-byte file, the answer reached the screen **zero** times: the model listed
the directory, saw `code.py`, and listed it again.

The three original task types all demanded the same four skills at once —
locate a file, open it, pick a string out of it, end the turn on that string
alone — so a model that could do the first two scored exactly what a model that
could do neither scored. That is a benchmark with no gradient in it, which is
the one thing an RL layer cannot work with.

So the task types are now three **families**, each a ladder whose rungs add one
shell skill apiece:

| family | easy | medium | hard |
| --- | --- | --- | --- |
| `ls` — the answer is a filename | `bash_ls_easy`: a flat handful, one odd extension. `ls` solves it. | `bash_ls_medium`: the same in subdirectories — the listing has to recurse. | `bash_ls_hard`: a marker *in a filename* among forty. `find -name`. |
| `read` — the answer is inside a file the prompt names | `bash_read_easy`: one file, one token. A single `cat`. | `bash_read_medium`: `key = value` lines; the stated key's value. | `bash_csv`, `bash_code`: a row of a CSV, a marked function. |
| `find` — the answer is *which* file holds a needle | `bash_find_easy`: three or four flat files. | `bash_find_medium`: eight to fifteen, still flat. | `bash_grep`: ten to fifty, some nested. |

`bash_ls_easy` is deliberately solvable by the command the model already types
unprompted. That is the point: the only thing left to fail at is naming one
line of output, which is the failure worth isolating — `bash_grep` put the
answer on screen in 13 of 41 episodes and still scored 0/41.

**A level is a claim about skills, not about size.** The haystack grows with
the rung because that is how the skill is forced, but what a level actually
promises is "one more thing than the rung below". `tests/test_tasks.py` pins
the size ordering within a family; nothing can pin the skill claim except
reading the generators.

**`bash_grep`, `bash_csv` and `bash_code` kept their ids and their generators
byte for byte.** A run is reproducible from `(model, task_type, seed)`, so a
generator that draws one more random number re-answers every episode on record
while the row still says `(bash_grep, 5)`. `generate_grep` is now
`_needle_task` with the old bounds and `nested=True`, and `tests/test_tasks.py`
pins seed 5's answer for each of the three against exactly that drift.

The run form defaults to the **easy rung of every family** rather than to
everything, for the reason the table exists.

## The ground truth never leaves early

Grading happens here, in `gyms/bash/grade.py`, against the answer stored in
`gym_episodes.task_config`. A response about an open episode is built from
`Task.public()` and carries the prompt, never the answer; `ground_truth`
appears only once the episode's status is `graded`, at which point it can no
longer help anyone. A grader on the client would be a benchmark that grades
itself.

Grading is forgiving in a bounded way, and the bound is the interesting part:
the reply is stripped of markup, narrowed to its **last line**, normalized
(quotes, trailing punctuation, a `/home/user/` prefix) and compared. A
reply that *ends* with the answer counts; the answer appearing anywhere else —
inside the reasoning, inside a command, halfway through a paragraph — does not.
`tests/test_grade.py` is the specification.

## Routes

Everything is admin-only, gated on the backend side
(`crates/api/src/routes/gym.rs`), and every path below is reached from a browser
under `/gym`.

| | |
| --- | --- |
| `GET /gyms` | the registry: gyms and the task types each generates. The frontend's catalogue. |
| `POST /gym/runs` `{gym_id, model_name, task_types[], count, seed_base?}` | creates the run, generates the tasks, seeds a workspace per episode. 502 if the sandbox could not be filled. |
| `GET /gym/runs`, `GET /gym/runs/{id}` | the record. |
| `GET /gym/episodes/{id}` | one episode with its steps. |
| `POST /gym/episodes/{id}/step` `{command, reply?}` | runs one command, records it with its shaping reward. |
| `POST /gym/episodes/{id}/finish` `{reply}` | grades the reply that ended the episode. |
| `POST /gym/episodes/{id}/close` | abandon: ran out of rounds, or the run went away. |
| `POST /gym/runs/{id}/status` | what the driver says is happening to the run it holds. |
| `POST /gym/trajectories` `{task_types[], count, seed_base?}` | expert demonstrations. Not a run, and nothing is stored **here** — the backend renders them and writes them to the data engine as an `sft` dataset. |

`finish` takes the whole reply rather than an extracted answer, because
extraction is part of grading and a caller that did it first would be a second
grader nobody versions.

An episode that is closed rather than graded is **absent from the run's mean**
rather than counted as zero: never answered is not the same as answered wrongly,
and counting it would blame the model for this side giving up.

**The last round of an episode is a terminal round**: the driver generates it
with `max_tool_calls: 0`, which masks the tool-call marker out of the engine's
grammar, so the model can only answer in prose. It is a mask and not an
instruction because a model this size does not reliably follow one — 22 of the
30 episodes in the first correctly seeded runs ended still calling the shell,
scoring zero for a reason that says nothing about whether the answer had been
found.

## A task has to fit the model

The shortest task plus its system block is ~72 tokens, and `build_prompt`
reserves half a model's window for the reply. **So a model trained under 256
tokens cannot be asked a gym task in full**: the prompt is cut (the engine
truncates rather than failing — see
[`../engine/api.md`](../engine/api.md)) and the model answers a question it was
shown part of, which reads afterwards as a model that scored zero.

That is why the prompts here are terse to the point of being curt, and why
`tests/test_tasks.py` pins their length. It is also why the run form warns below
`MIN_CONTEXT` and the run socket carries `context_multiplier` — stretching a
small model's window is the only way to run one against these tasks at all, so
the warning is a warning and not a refusal.

## Rewards

Two signals, and they are not the same thing.

- **The score** is the grade of the final answer: 1 or 0, and the only number a
  run reports.
- **The step reward** (`gyms/bash/rewards.py`) is shaping — a few hundredths for
  a command that ran, more for one that named what the task told it to look for,
  most for one whose *output* proved the answer is on screen, and a penalty for
  dawdling past the third step. It is the dense signal a policy gradient wants
  between the sparse ones, and it is kept deliberately small: a step reward that
  could compete with the answer is a reward a model can farm by running commands
  forever. What consumes the two of them, and the clip that makes that sentence
  enforceable rather than a matter of magnitudes, is
  [`../decisions/reinforcement-learning-in-the-engine.md`](../decisions/reinforcement-learning-in-the-engine.md).

**A task carries two strings for the shaping, and confusing them is the bug this
already had.** `Task.target` is matched against the model's **command** and must
never be the answer — paying for the answer appearing in a command pays a model
for typing something it could only know by having already found it. Every
`bash_csv` task set `target` to the answer, which is exactly that, and
`tests/test_tasks.py` now refuses it for every task type. `Task.evidence` is
matched against the command's **output**, may be the answer, and is unfarmable
for the same reason it is allowed to be: the shell has to print it.

**The two task-shaped terms are paid once an episode.** Under evaluation the
distinction is invisible — a model names the needle once — and under RL a term
payable every step is a term farmed by repeating one good command to the round
budget. What has already been paid is recomputed from the steps on record
(`Earned.over`) rather than stored, so changing what the shaping pays for does
not leave old episodes being paid by the old rule.

Why `SAW_TARGET` exists at all: across 174 steps of the first correctly seeded
runs, `RAN_CLEANLY` fired 127 times and `ON_TARGET` fired **zero**. The only
signal a model actually received was "your command parsed", which `ls` earns as
readily as `grep`.

## Demonstrations, and why a task type varies

`gyms/bash/expert.py` writes the correct trajectory for a task — the commands,
and the answer alone as the last turn — which costs nothing because the
generator already holds the ground truth: the gym demonstrates itself, and
behaviour cloning needs no collected dataset and no larger model. That is also
why every generator draws its phrasing, filenames and markers from a pool rather
than an f-string: a few thousand demonstrations over one template per type would
teach three sentences.

They become an **ordinary `sft` dataset in the data engine**, assembled by the
backend, so the fine-tune needs no new machinery in the LLM engine at all. See
[`demonstrations.md`](demonstrations.md).

## Adding a gym, or a task type

1. Write `generate(seed) -> Task` in `gyms/<name>/tasks.py`, and a grader if the
   bash one does not fit.
2. Add a `TaskType` row to the gym in `gyms/registry.py`, with its `level` and
   `family`.
3. Add the generator to `GENERATORS` in `tests/test_tasks.py` **and** in
   `tests/test_expert.py`, which is how it inherits every invariant the others
   are held to rather than by anyone remembering to test it.
4. Write its moves in `gyms/bash/expert.py` and map them in `_MOVES`. Optional
   for a type that is only ever evaluated; required for one that will be cloned
   from, and `tests/test_expert.py` runs them against a real shell.

The frontend reads `GET /gyms`, so a new task type appears in the run form —
under its level — with no frontend change at all.

Two things a new type has to get right, both of which have already been got
wrong here: `target` is matched against the model's **command** and must never
be the answer, and `evidence` is matched against a command's **output** and has
to be something the shell actually prints — a string in a file's body, or the
name of a file. `tests/test_tasks.py` refuses both mistakes for every
registered type.

## Local development

```sh
cd deploy/just-bash && docker compose up -d   # the sandbox, on 8899
cd gyms && .venv/bin/python -m pytest tests/
# The gate: reads runs, trains nothing. Point it at the database the runs are in
# — `docker compose up` publishes Postgres on 5433, which the default is not.
GYMS_DATABASE_URL="postgresql+psycopg://slate:slate@127.0.0.1:5433/slate_gyms" \
  .venv/bin/python -m scripts.gate
```

The tests stand in for just-bash in-process (`tests/conftest.py`) rather than
running it, and use SQLite in memory, so they need neither Docker nor Postgres —
apart from two marked suites, both of which skip when what they need is not
there: `tests/test_migrations.py` (`postgres`, `GYMS_TEST_ADMIN_URL`) and
`tests/test_sandbox.py` (`sandbox`, `BASH_URL`).

**Alembic owns the schema**, as it does in the engine and the data engine —
`main.py` binds the session factory and creates nothing. It used to pass
`create=True`, and that is worth knowing about because of what it cost: a
database stuck on an older revision had its *missing* tables quietly added by
`create_all` while the one that already existed kept its old definition, so the
disagreement surfaced as a 500 the first time a model ran a command rather than
as a failure at startup.

`0002` exists for the same episode. `0001` was rewritten in place instead of
superseded — the module had never been committed and no database was believed
to hold the old shape — and reusing the revision id is what made the stale
database invisible: `alembic upgrade head` saw `0001` stamped and did nothing.
So `0002` is conditional, which no other migration here should be: one revision
id names two schemas, and it branches on what is actually there.
`tests/test_migrations.py` covers both ends, against a real Postgres.
