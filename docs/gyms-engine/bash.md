# Bash — tasks and shaping rewards
## Bash families and levels

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

The original task types form three **families**, each a ladder whose rungs add
one shell skill apiece. A fourth, `compose`, adds writing and pipelines; see
[composition and mutation](demonstrations.md#composition-and-mutation).

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
  [`../decisions/reinforcement-learning-in-the-llm-engine.md`](../decisions/reinforcement-learning-in-the-llm-engine.md).

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
