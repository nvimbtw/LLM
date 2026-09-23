# Gyms engine — where a model is put to work

A model is measured here by doing a task and having its answer scored. One
module, one database (`slate_gyms`). Bash tasks use a sandbox; [math](math.md),
[state tracking](state-tracking.md) and [constraint logic](constraint-logic.md)
generate exact-answer tasks without tools. The [general gym](general.md) mixes
their questions across episodes, delegating each episode to its source provider.

The LLM engine trains against this service through the backend. This service
owns the tasks, environments and scores, not the optimizer.

## Ownership, and who drives

`gyms-engine/` owns `gym_runs`, `gym_episodes` and `gym_steps`, the task generators,
the sandbox seeding and the grading. The backend owns users; the LLM engine owns
models.

**It does not own the loop.** A run is driven from
`backend/crates/api/src/gym.rs`, over a WebSocket the browser holds, and this
service serves one move of that loop per request. The reason is a single
sentence: what the model reads after a command is rendered by
`tools::observation` in Rust, the identical function the chat `bash` tool uses,
and a second renderer in Python would be a gym that teaches a layout serving
never shows. See
[`../decisions/gyms-are-driven-by-the-backend.md`](../decisions/gyms-are-driven-by-the-backend.md).

The service is spawned by the backend as a child (`GYMS_ENGINE_*`, port 8003) like
`llm-engine`/`data-engine`, and its tokens carry `aud: gyms-engine`. It talks to
[`just-bash`](../backend/README.md#the-sandboxed-shell) at `BASH_URL` and to nothing
else.

## Who a run is for

`gym_runs.kind` is `eval` or `rl`, and `GET /gym/runs` serves the `eval` ones.
The two use the same three tables, source providers and grading — a
training run's iteration is N episodes of a model against generated tasks,
which is exactly what an evaluation is — and there is nothing else in the row
that could tell them apart.

The filter exists because **a training run creates one gym run per optimizer
step**. Dozens of them, in minutes. Without the kind, the board a person reads
to compare models is the last fifty batches of whichever run happens to be
training, and every evaluation anybody made is off the bottom of it.
`GET /gym/runs?kind=rl` is how the batches are read back.

## A run, an episode, a step

```
run       one model against N generated tasks       gym_runs
 └ episode one task, its source gym and environment gym_episodes
    └ step  one command and what it printed         gym_steps
```

A run exists because one episode says almost nothing: these tasks are pass or
fail, so ability is only visible across a batch. Episode *i* starts from seed
`seed_base + i // group_size`. Math may advance that seed to deduplicate
single-attempt questions; use the recorded episode seed, split and version for
replay. General runs also freeze the source catalogue and sampling weights in
`gym_runs.selection`. See [general.md](general.md) for selection and replay.

**`group_size` is what makes a run its own GRPO batch.** At 1 — the default, and
what every run made before it existed is — a run is `count` distinct tasks,
which is what an evaluation wants. Above 1, consecutive episodes are repeated
attempts at *one* task. Source runs cycle types; general runs sample them. The
group index moves the seed in both cases, so a source run with `group_size` 8 over four types is eight rollouts of a `bash_grep`
task, then eight of a `bash_csv` one, and so on.

Both halves have to key off the group or a "group" is G different questions with
G different answers, which is a baseline over nothing — an advantage is how much
better a rollout did than its *siblings*, and siblings are attempts at the same
task. A count that would leave a partial group is refused rather than truncated:
the remainder would be a group whose baseline came from fewer samples than every
other, and at the limit a group of one, which contributes exactly zero gradient
and is the case `group_size` exists to prevent.

Identical tasks are still not a shared workspace — the episode id is the sandbox
key, so eight siblings get eight filesystems and cannot see each other's work.

Two things make a grouped run meaningless and neither is visible in its row:
**temperature above zero**, or the rollouts are one rollout counted G times, and
**no sampler seed** — `ChatConfig.seed` pins the draw. Sampling reaches a run
from the browser socket (`gym.rs`) and is not recorded anywhere, so that is a
property of how a run was produced and not something the record can check.

**`gym_runs.model_name` holds the model's id, not its name.** The field is
named after the LLM engine's own, which has always carried the id: weights live at
`artifacts/models/<id>/model.pt`, so a display name resolves to no path at all
and the run dies on its first round with `No weights at …`. Chat sends the same
thing under the same misleading key. The frontend turns it back into something
readable with `modelLabel` (`lib/gyms.ts`).

**A bash episode id is its sandbox session key.** One bash episode is one filesystem,
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

## Bash tasks

The shell skill ladders, historical generator guarantees and shaping rewards
are documented in [bash.md](bash.md).

## The ground truth never leaves early

Complete grading belongs to the source provider, including bash file-change
checks. Bash answer parsing lives in `gyms/bash/grade.py`; math has its own
numeric grader, and the two reasoning gyms share a short-name grader. All use
private state stored in `gym_episodes.task_config`.
An open-episode response is built from `Task.public()` and carries the prompt, never the answer; `ground_truth`
appears only once the episode's status is `graded`, at which point it can no
longer help anyone. A grader on the client would be a benchmark that grades
itself.

Bash answer grading is forgiving in a bounded way:
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
| `POST /gym/runs` `{gym_id, model_name, task_types[]?, count, seed_base?, group_size?, kind?, split?, gym_weights?, task_weights?}` | creates the run and prepares each source environment. General selection is described in [general.md](general.md). `group_size` defaults to 1 and must divide `count` — 400 otherwise. `kind` defaults to `eval`. 502 if the sandbox could not be filled. |
| `GET /gym/runs?kind=`, `GET /gym/runs/{id}` | the record. The listing is one kind at a time, defaulting to `eval`. |
| `GET /gym/episodes/{id}` | one episode with its steps. |
| `POST /gym/episodes/{id}/step` `{command, tool?, reply?}` | runs one command with its shaping reward. `tool` defaults to `bash` and must be allowed by the episode. |
| `POST /gym/episodes/{id}/finish` `{reply}` | grades the reply that ended the episode. |
| `POST /gym/episodes/{id}/close` | abandon: ran out of rounds, or the run went away. |
| `POST /gym/runs/{id}/status` | what the driver says is happening to the run it holds. |
| `POST /gym/trajectories` `{gym_id?, task_types[]?, count, seed_base?, gym_weights?, task_weights?}` | expert demonstrations. Not a run, and nothing is stored **here** — the backend renders them and writes them to the data engine as an `sft` dataset. Up to 5,000 per request. |

`finish` takes the whole reply rather than an extracted answer, because
extraction is part of grading and a caller that did it first would be a second
grader nobody versions.

**A second driver reaches these through the backend's `/internal/gym/*`.** The
LLM engine rolls out a training batch there, and the difference between the two
seams is one field: the internal step route answers with the *rendered*
observation beside the raw output, so the LLM engine never lays one out. Both
renderings are the same call to `tools::observation` in
`crates/api/src/gym.rs::play`. Nothing in this service knows which driver is
asking.

An episode that is closed rather than graded is **absent from the run's mean**
rather than counted as zero: never answered is not the same as answered wrongly,
and counting it would blame the model for this side giving up.

**The last round of an episode is a terminal round**: the driver generates it
with `max_tool_calls: 0`, which masks the tool-call marker out of the LLM engine's
grammar, so the model can only answer in prose. It is a mask and not an
instruction because a model this size does not reliably follow one — 22 of the
30 episodes in the first correctly seeded runs ended still calling the shell,
scoring zero for a reason that says nothing about whether the answer had been
found.

## A task has to fit the model

The shortest task plus its system block is ~72 tokens, and `build_prompt`
reserves half a model's window for the reply. **So a model trained under 256
tokens cannot be asked a gym task in full**: the prompt is cut (the LLM engine
truncates rather than failing — see
[`../llm-engine/api.md`](../llm-engine/api.md)) and the model answers a question it was
shown part of, which reads afterwards as a model that scored zero.

That is why the prompts here are terse to the point of being curt, and why
`tests/test_tasks.py` pins their length. It is also why the run form warns below
`MIN_CONTEXT` and the run socket carries `context_multiplier` — stretching a
small model's window is the only way to run one against these tasks at all, so
the warning is a warning and not a refusal.

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

Source gyms implement the shared task/provider contract in `gyms/contracts.py`;
registration makes their task types available to the general gym automatically.
The contract, capability declarations, testing requirements and registration
steps are in [general.md](general.md#adding-a-source).

For another bash task, retain the seeded generator's historical behavior, wrap
it with `bash.runtime.generator`, and register its `TaskType`. Add tests to
`tests/test_tasks.py` (or the composition suite), and expert moves with tests
if it supports demonstrations. `target` is command-facing and must be derivable
from the question; `evidence` is output-facing and must actually be printed.

The frontend reads `GET /gyms`, so registered tasks appear under their levels.
Composite gym tools are resolved per episode; an empty composite catalogue tool
list must not be interpreted as a tool-free run.

## Local development

```sh
cd deploy/just-bash && docker compose up -d   # the sandbox, on 8899
cd gyms-engine && .venv/bin/python -m pytest tests/
# The gate: reads runs, trains nothing. Point it at the database the runs are in
# — `docker compose up` publishes Postgres on 5433, which the default is not.
GYMS_ENGINE_DATABASE_URL="postgresql+psycopg://slate:slate@127.0.0.1:5433/slate_gyms" \
  .venv/bin/python -m scripts.gate
```

The tests stand in for just-bash in-process (`tests/conftest.py`) rather than
running it, and use SQLite in memory, so they need neither Docker nor Postgres —
apart from the marked Postgres migration suite (`GYMS_ENGINE_TEST_ADMIN_URL`)
and the sandbox/expert suites (`BASH_URL`), which skip when their services are
unavailable.

**Alembic owns the schema**, as it does in the LLM engine and the data engine —
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
