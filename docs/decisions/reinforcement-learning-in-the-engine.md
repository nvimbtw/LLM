# Reinforcement learning runs in the engine, on rollouts the backend serves

- **Date**: 2026-08-30
- **Status**: Accepted, not implemented. The gate below was measured on
  2026-08-29 and failed. Of the prerequisites it named, the **terminal round**,
  the **reward fixes** and the **behaviour-cloning path** are now built; the
  fine-tune has not been run and the gate has not been re-run. RL itself waits
  on that re-run.

## Context

The gym produces two numbers per episode and nothing consumes either. The score
is the grade of the final answer; the step reward is the dense shaping beside it
(`gyms/bash/rewards.py`). Both exist because
[`gyms-are-driven-by-the-backend.md`](gyms-are-driven-by-the-backend.md) built
the module ahead of the training it is named for: "generated tasks, a shell, a
dense per-step reward, a terminal score, and the whole trajectory on disk."
This record is what turns that record into a gradient.

It also has to settle something that record left open. Its own "what this costs"
says the RL driver "will run in the engine, reaching `/gym/*` through the
backend the way `llm/corpus/remote.py` already reaches the data engine" — while
its "rejected" list rejects *the engine drives, rendering observations in
Python* and *the engine drives, calling the backend to render*. Read together,
those two sentences describe a driver that cannot be built. One of them has to
give, and the resolution is in the decision below: it is the second rejection
that was argued against the wrong loop.

Three facts bound everything here.

**A policy gradient is computed from token ids and log-probabilities, and no
HTTP frame carries them.** That is not a preference for co-location; a
trajectory that arrives as text has to be re-tokenized to be trained on, and a
re-tokenized rollout is a *guess* at what the sampler drew — under a masked
grammar (`_ToolCalls.mask`) it is not even a well-defined guess.

**The sandbox is not reachable from the engine.** The Python services never
speak to each other, so every command an engine-driven rollout runs is a hop
toward the backend regardless of who renders the observation.

**A model this size may have nothing to reinforce.** These tasks are pass or
fail, and a group of rollouts that all fail has a zero gradient — exactly, not
approximately. That makes the base model's pass rate a **precondition of the
feature**, not a result of it. See *Gates*.

## Decision

**The engine drives an RL rollout, and the backend serves one move of it —
rendered.** The training loop, the sampler, the trajectory and the update are
all in one process; the shell and the grader stay where they are.

The move that resolves the tension above: the observation is still rendered
exactly once, by `tools::observation` in Rust, and the engine never sees a
`{stdout, stderr, exit_code}` triple at all. `POST /internal/gym/episodes/{id}/step`
runs the command through the gyms service and answers with the *rendered* block,
the step reward, and the step number. The old rejection — "a round trip per
command purely to format a string" — priced a round trip the engine would
otherwise not make. It makes this one anyway: the sandbox is behind the backend.
The rendering rides a hop that already had to happen, and costs nothing.

So there are two rollout loops and still one renderer. `gym.rs` drives an
evaluation for a browser that is watching; `llm/orchestrate/rl.py` drives a
training batch nobody is watching. Neither renders an observation.

### The shape of a run

```
iteration
 └ batch      T tasks x G rollouts of each          one optimizer step
    └ group   G rollouts of ONE seed                one advantage baseline
       └ episode  rounds of (generate, run, observe)   one return
```

A **group is the baseline**. G rollouts of the same seed at temperature give G
returns; the advantage is how much better a rollout did than its siblings on the
identical task. This is GRPO, and it is chosen over PPO for one reason beyond
the usual: a critic is a second network the size of the policy, on one device,
predicting a binary outcome — and the gym's own structure (a run is N episodes
of one task type) already hands us the group for free.

`gym_episodes.seed` is not unique, so a group is only a run created with
`group_size`: episode *i* draws `seed_base + i // group_size`. One line in the
generator, and every episode still keeps its own filesystem — the episode id is
the sandbox key, so identical tasks are not a shared workspace.

### The return, and why shaping cannot outrank an answer

```
R = score + clip(sum of step rewards, -C, +C)      C = 0.2
```

The invariant is `2C < 1`: **no incorrect episode can outrank a correct one in
its own group**, whatever it did with the shell. The shaping breaks ties among
rollouts that all failed and among rollouts that all succeeded — which is
precisely the two places a binary reward is silent — and it can do nothing else.
The clip is what makes that a guarantee rather than an argument about magnitudes;
`rewards.py` already keeps the terms small, and this stops the bound depending on
it continuing to.

Advantage is group-relative, `A_i = (R_i - mean) / (std + eps)`, and a group whose
returns are identical contributes **nothing** — all-failed and all-passed alike.
Those groups are skipped, and the skipped fraction is a reported frame, because
it is the number that says whether a run is learning or merely spending.

Every token the policy generated in an episode carries that episode's advantage.
Per-step credit assignment — attaching a step's reward only to the turn that
earned it — was considered and deferred: at eight rounds with a dominant terminal
signal it moves little, and it is an addition to this shape rather than a change
of it.

### What a token gets loss, and what never does

The mask is the one SFT already writes: **the model's own generated tokens, and
nothing else.** A `<|tool-result|>` block is not a training target under
`template.training_segments` and must not become one here — training it teaches a
model to invent the shell's output, which is the same failure as teaching it to
invent search results. The prompt, the system block and every observation are
context; the assistant turns are the policy.

Because the rollout is generated in-process the mask is not reconstructed from
text — it is what the sampler wrote down as it drew.

### One update per batch, so there is no ratio

The batch is generated by the weights that are about to be updated, and it is
used exactly once. The importance ratio is 1 by construction, so there is no
ratio, no clipping and no stored old log-probabilities: the loss is REINFORCE
with a group baseline, which is what GRPO reduces to at one inner epoch.

`inner_epochs > 1` is the first thing to want later and the change it needs is
written down here so it is not re-derived: keep the sampled log-probabilities,
add the clipped ratio, and the frames gain a `clip_frac` that is **absent, not
zero**, on a run that did one epoch.

### KL to a frozen reference

`beta * KL(policy || base)`, k3 estimator, against the model the run started
from. Not a diagnostic: without it a binary reward and a forgiving grader
converge on a model that has learned to end every reply with a plausible-looking
token and has forgotten how to write a sentence.

It costs what `SftConfig.measure_base_kl` already documents — a second resident
copy of the base weights — and that cost is now mandatory rather than optional,
which is a real change in what a run needs to fit.

### An iteration is a checkpoint, and evaluation is an ordinary gym run

Every iteration writes a checkpoint, which is a model
([`checkpoints-are-models.md`](checkpoints-are-models.md)), so measuring
progress needs nothing new: a gym run against `<the RL run> #n`, at temperature
0, on seeds the training batches never drew. Held-out seeds are the whole of the
methodology — a seed that trained is a seed the model may have memorised the
answer to, and `seed_base` makes holding them out a subtraction rather than a
feature.

### The engine writes down no trajectory

`gyms/` owns the record — runs, episodes, steps, rewards — exactly as it does
for a browser-driven run. The engine holds a batch in memory, takes its step,
and drops it. This is the rule `llm/corpus/remote.py` already states for
corpora: a trajectory the engine keeps a copy of is a trajectory the engine
owns, and it owns models.

## The seams that have to move

Ordered by which module they land in, and none of them is large.

**`gyms/`** — `group_size` on run creation, and the seed arithmetic above. A run
kind flag so an RL run's episodes are not listed on the eval board beside a
person's. Nothing else: the tasks, the sandbox, the grader and the rewards are
already what RL needs, which was the point of building them first.

**`backend/`** — an `/internal/gym/*` family mirroring `/internal/datasets/*`,
authenticated by `BACKEND_SERVICE_SECRET` (the hop that runs toward the backend),
and answering with `tools::observation`-rendered blocks. `crates/api/src/gym.rs`
keeps the browser loop; the shared part is the renderer, which it already
imports.

**`engine/`** — the actual work:

- `llm/config.py::RlConfig`, and `RunKind.rl` in `store/schema.py`.
- `llm/orchestrate/rl.py` — the pipeline: load, roll out a batch, score, update,
  checkpoint, repeat.
- `llm/train/rollout.py` — a batch of episodes carried in lockstep: every live
  episode generates its round in one batched forward, then their commands run
  concurrently over HTTP, then the finished ones drop out. Sequential episodes
  are the phase-1 fallback and are roughly `G x T` times slower.
- `llm/train/policy.py` — return, group advantage, masked loss, KL.
- **A batched sampler.** `sample.pick` takes a 1-D logit vector and
  `_ToolCalls.mask` masks one sequence's logits; both are per-sequence today.
  Lockstep rollouts need a row-wise version of each, and this is the single
  largest piece of engine work in the list.
- Reuse, not re-implementation: `loop.lr_at`, `loop.step_plan`,
  `loop._Checkpoints`, `loop.release_device`, and the `Progress` frame.

**`frontend/`** — an RL run reports a pass rate rather than a loss curve, so the
run page needs one chart and the gym run rows need to name the training run that
produced the checkpoint they measured.

## Gates — measured, 2026-08-29

The pass-rate gate was run before this record was written, against the eight
gym runs in `slate_gyms`. **It fails, and not for the reason it was written to
catch.**

85 episodes, 0 correct. But only three of the eight runs are evidence at all:

| run | what it measures |
| --- | --- |
| 21:06, 21:10 (25 eps) | two models that generated an **empty reply** every episode. Nothing was tested. |
| 21:19, 21:24, 21:30 (30 eps) | seeded to `/workspace/files` — `ls -la` in `/home/user` returned an empty directory. The environment, not the model. |
| **21:37, 21:39, 21:45 (30 eps)** | **seeded correctly. The only real reading.** |

On those 30: **0 correct, and 22 of them never produced an answer at all** —
the round budget ran out with the model still calling the shell. 19 extracted
an empty answer. Mean 5.8 steps of 8.

So the gate's real finding is not "the model tries and fails". It is:

- **The model does not terminate.** Answering is a separate act from searching
  and it never gets there. This is prior to difficulty: on a task so easy the
  first command solves it, an episode that never stops calling still scores 0.
- **The model does not use the right verbs.** Across 174 commands: 81 `ls`, 66
  `find`, **10 `grep`**, 14 `cat`. Every `bash_grep` task is one
  `grep -r NEEDLE .`, and the model reaches for `find -name` — matching "find a
  file" to the wrong tool and never reading a file at all.
- **It repeats.** `ls -la` at steps 3, 4 and 6 of one episode, identical output
  each time, no state carried between rounds.
- The scaffold itself is **fine**: well-formed `<|tool-call|>` blocks, exit 0,
  real listings coming back. Nothing in the plumbing is what is failing.

**Consequence for this design.** GRPO's gradient is exactly zero here, and so is
rejection-sampling SFT's — there is nothing to sample from. Neither the phase-1
plan nor its fallback can start. Two things have to land first, and neither is
RL:

1. **A terminal round.** *Built.* The last round of an episode is generated with
   `max_tool_calls: 0`, which masks the tool-call marker out of the grammar, so
   the model must answer in prose. `gym.rs` already grades whatever the last
   reply says; this makes that reply an answer instead of another command. It
   converts "never answered" into "answered, probably wrong", which is the
   difference between a group with no variance and a group with signal.
2. **Behaviour cloning, from trajectories the gym can write itself.** *Built,
   end to end; the fine-tune has not been run.* The task generator holds the
   answer, so a *correct* trajectory is a template, not a dataset anyone has to
   collect: `grep -r NEEDLE .`, the real output, the answer on the last line. A
   few thousand cost nothing to emit, and they teach the two things missing —
   which verb, and when to stop. RL then has somewhere to start.

   `gyms/bash/expert.py` and `POST /gym/trajectories` produce them, verified
   against the real sandbox — every command exits 0, the evidence reaches the
   transcript and the answer grades 1.0.

   **The corpus is an ordinary dataset, and that is the correction.** An earlier
   draft of this record concluded the assembly was *engine* work, on the grounds
   that `llm.chat.tools.render_result` is "the single renderer for both the live
   path and the SFT data". That conflated rendering with storage. `ChatTurn`
   already accepts a tool turn that names its call and renders it itself, so a
   demonstration is stored **unrendered** — `{tool, argument, content}` — and
   `render_result` still runs exactly once, in the engine, at build time.

   So the demonstration goes where every corpus goes: `POST /gym/demonstrations`
   on the backend renders each observation body with `tools::observation` and
   writes an `sft` dataset to the *data engine*, which the existing SFT build
   reads with no new reader, no new key and no new corpus path. The engine owns
   no corpora — a demonstration set beside one would be the engine owning a
   corpus again — and this way the demonstrations can be **mixed** into one
   fine-tune with a chat corpus, which matters, because three task types alone
   train a model that only answers gym questions. See
   [`../gyms/demonstrations.md`](../gyms/demonstrations.md).

   One thing this surfaced that was not about gyms at all: the layout of a shell
   result is now rendered in three languages, and only two of them were pinned
   against each other. `contracts/shell-observation.json` pins the third. That
   drift has no symptom on any test — the corpus looks fine and the fine-tune
   converges — until a model meets the real thing.

### Re-running the gate

`gyms/scripts/gate.py` asks the question of a *set* of runs, because a group is
G rollouts of one seed and a group with no variance in its returns has no
gradient in it — all-fail and all-pass alike.

Eight ordinary 20-episode runs sharing one `seed_base` **are** twenty groups of
eight: episode *i* draws `seed_base + i` in every run. So the gate costs no
schema and no `group_size` field.

```sh
cd gyms && .venv/bin/python -m scripts.gate --seed-base 12345
```

Two things make the answer meaningless if got wrong, and both are properties of
how the runs are produced rather than of the script: **temperature above zero**,
or the eight rollouts are one rollout counted eight times, and **no sampler
seed** — `ChatConfig.seed` pins the draw, so eight runs would be eight copies.

The number to read is `GROUPS WITH SIGNAL`, not the mean score.

## Difficulty tiers, and what "easy" has to mean

Tiers are the right next thing and they are what makes the curriculum below
possible — but the measurement above says **easy cannot mean a smaller
haystack**. `bash_csv` is already a single file in the directory the shell opens
in, asking for one row of eighteen, and it scored 0 on all 9 valid episodes.
Shrinking the search space is not the axis that is binding.

**Easy means fewer things that must be chained.** A tier is defined by the
length of the shortest correct trajectory:

| tier | shortest solution | what it tests |
| --- | --- | --- |
| easy | one command, named in the prompt (`cat data.csv`) then answer | can it stop and answer at all |
| medium | choose the verb, then answer (today's `bash_grep`, `bash_csv`) | can it pick `grep` over `find` |
| hard | explore, then choose, then read, then answer | today's `bash_code`, and up |

A tier is a variation axis like any other, and the generators now have the pool
machinery for it — the phrasing, filename and marker already vary per seed (see
[`../gyms/demonstrations.md`](../gyms/demonstrations.md)). What a tier adds is a
*bound on the shortest solution*, which is a property of the generator's
structure rather than of its surface.

**What shipped, and where it differs.** The tiers are in
[`../gyms/README.md`](../gyms/README.md) as three *families* of three rungs
rather than one global ladder, so a run says which skill a model stops at and
not only how far it got. The bound above is the one that was implemented:
`bash_read_easy` is this section's own easy example — one file named by the
prompt, one `cat`, then the answer.

One nuance the table above gets slightly wrong in the light of that. "Easy
cannot mean a smaller haystack" is right about `bash_csv`, and `bash_find_easy`
still ships three files rather than fifty — not to shrink the search, but
because at three files `cat *` is a correct one-command solution and at fifty it
is not. The haystack is the lever; the *shortest correct trajectory* is still
what defines the rung.

That ordering is a **curriculum**, and it is why tiers and RL belong in the same
plan rather than in sequence: a group of rollouts on an easy task is where a
non-zero pass rate first appears, and GRPO promotes a run to the next tier when
its pass rate crosses a threshold. A tier whose pass rate is 0 teaches nothing
and a tier whose pass rate is 1 teaches nothing either — the curriculum exists
to keep the batch in between, which is exactly where group variance is largest.

## What can be farmed, and what stops it

A reward is a specification, and this one has holes worth naming before a model
finds them.

- **`ON_TARGET` pays the model for typing the answer.** `rewards.py`'s docstring
  says `target` is the needle and not the answer, because "rewarding the answer
  appearing in a command would pay a model for typing something it could only
  know by having already found it". **For all 29 `bash_csv` episodes in the
  database, `target` and `answer` are the same string** — the generator sets
  both to `VAL_nnnn`. Under evaluation that is harmless; under RL it is a term a
  model can farm by echoing its guess into a command. Fix the generator, and pay
  `ON_TARGET` **once per episode** regardless.
- **`RAN_CLEANLY` pays for a command that ran.** `true` runs. `DAWDLE` and the
  round budget already bound the total, and the clip bounds it again.
- **Guessing is free.** The grader reads the last line, and a wrong answer and no
  answer both score zero — so answering immediately has non-negative expected
  value and costs no rounds. This is the one hole with no clean patch, and it is
  deliberately left open in phase 1 and *watched*: if mean episode length
  collapses toward one round while pass rate stalls near chance, that is the
  model having found it, and the fix is a small penalty on answering before any
  command rather than a change to the grader.
- **The score is the grader.** Every property of `gyms/bash/grade.py` — the last
  line, the normalization, the forgiveness — is now an optimization target rather
  than a measurement convenience. `tests/test_grade.py` stops being a test of a
  helper and becomes the specification of what the model is being paid for.

## What this costs

- **Two rollout loops, permanently.** The browser's and the trainer's, in two
  languages, sharing tasks, rewards, grader and renderer — everything that
  defines what the gym *measures* — and not the forty lines of ask-run-ask.
  Accepted knowingly, again.
- **A run needs both models resident**, policy and frozen reference, plus a
  batch of rollouts in flight. On one device, against a training run's usual
  ceiling.
- **A rollout batch is bounded by HTTP, not the GPU.** `G x T` sandboxed commands
  per round, and the card idles through all of them. Lockstep batching is what
  makes that one wait instead of `G x T` waits, and it is why the batched sampler
  is not optional.
- **The engine calls the backend while the backend is calling the engine.** The
  rollout's step request is in flight from a training thread while nothing else
  in that process blocks; the pipeline must reach the backend the way
  `remote.py` does — off the event loop — or a run deadlocks against its own
  service.
- **Reproducibility weakens.** A gym run is reproducible from its row. An RL run
  is reproducible only given the same weights, seeds, sampler *and* sandbox
  behaviour, and the last of those is a container.

## Rejected

- **The backend drives rollouts headlessly and posts trajectories to the
  engine.** The natural reading of the existing driver, and it fails on the first
  fact in this record: what arrives is text, and a policy gradient needs the ids
  and log-probabilities the sampler drew. It also needs a weight-sync story —
  the policy changes every iteration — and would grow a second run queue outside
  the module that owns runs.
- **The engine renders observations in Python.** Rejected once already, and now
  worse: the gym would be teaching a layout serving never shows, in the one place
  the model is being *paid* to fit it.
- **Serving each iteration's policy from disk so the browser loop can generate
  it.** Four times a model's weights per iteration, a second resident copy, and
  the rollouts still come back as text.
- **PPO with a value head.** A critic the size of the policy, on one device, to
  predict a binary outcome the group baseline already estimates for free.
- **Rejection-sampling SFT as the destination.** It is the fallback if the pass
  rate gate fails, and it is a good one — but it can only ever reinforce what the
  model already does sometimes, and it discards the negative half of every batch.
- **Training on the shaped reward.** The gym's own doc says why: a step reward
  that can compete with the answer is a reward a model farms by running commands
  forever. The clip is that sentence made enforceable.
- **A `closed` episode as a zero.** An episode the driver abandoned is absent
  from a run's mean, and it stays absent from a batch: it measures this side
  giving up, not the model. An episode that exhausted its rounds is a different
  thing — `gym.rs` grades what it last said, and that zero is earned.
