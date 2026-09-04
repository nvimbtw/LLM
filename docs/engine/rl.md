# Reinforcement learning — practising against a gym

A run that improves a model by *doing* rather than by reading. It rolls out a
batch of gym episodes, grades them, and takes one optimizer step on the tokens
the model itself drew. Every iteration writes a checkpoint, and a checkpoint is
a model, so measuring progress needs nothing new.

Why it lives here rather than beside the gym, and what was weighed:
[`../decisions/reinforcement-learning-in-the-engine.md`](../decisions/reinforcement-learning-in-the-engine.md).
The one-sentence version: a policy gradient is computed from token ids and
log-probabilities, and no HTTP frame carries them.

## The shape of a run

```
iteration  =  one gym run per lane          one optimizer step, one checkpoint
 └ batch      T tasks x G rollouts          lanes concatenated
    └ group   G rollouts of ONE seed        the advantage baseline
       └ episode  rounds of (generate, run, observe)   one return
```

**One iteration is one gym run**, created through `/internal/gym/runs` with
`kind: "rl"`. Not a convenience: it makes an iteration reproducible from a row,
and it means the trajectory record, the step rewards and the grading are the
gym's, exactly as they are for a run a person watched. `llm/orchestrate/rl.py`
holds a batch in memory, takes its step, and drops it — **the engine writes down
no trajectory**, the same rule `llm/corpus/remote.py` states for corpora.

## Where the pieces are

| | |
| --- | --- |
| `llm/backend.py` | The hop toward the Rust backend: the token, the base URL, the failure. `corpus/remote.py` and `gym/client.py` both read it, so there is one token minter. |
| `llm/gym/client.py` | Creating a run, stepping an episode, grading it. **Nothing here retries** — see below. |
| `llm/train/rollout.py` | A batch of episodes in lockstep: one batched forward per round, then every command at once. |
| `llm/train/policy.py` | The return, the group advantage, the masked loss, the KL. |
| `llm/orchestrate/rl.py` | Load, roll out, score, update, checkpoint, repeat. |
| `llm/config.py::RlConfig` | What a run is asked for. States no architecture, like `SftConfig`. |

## A read may be retried and a write may not

`corpus.remote.Client` retries a dropped page, because the alternative is losing
the pack around it and asking twice costs a page. A gym step is a command that
ran in a real filesystem: asking again after a lost response runs it twice,
records two steps, and pays the shaping reward twice for one action. So
`backend.Hop` makes exactly one attempt and the retry lives with the caller that
can afford it.

## The mask is written down as it is drawn

Each rollout holds one growing list of ids and one growing list of flags saying
which of them the model produced. It is **never** recovered by re-rendering the
conversation and looking for the assistant's spans: a re-tokenized rollout is a
guess at what the sampler drew, and under a masked grammar not even a
well-defined one.

Appending is exact for one reason, and `tests/test_rollout.py` pins it: every
seam between what the model drew and what was appended around it is a marker,
and a marker is an added token the BPE never merges across. That is what makes
appending equivalent to the single-pass encode `build_prompt` deliberately does.

Only drawn tokens carry loss. The prompt, the system block and every observation
are context — a `<|tool-result|>` block is not a training target under
`template.training_segments` and does not become one here, because training it
teaches a model to invent the shell's output.

## What a ragged batch needs from the model

`Transformer.forward` takes two optional arguments that exist for this and are
passed by nothing else, so every other path is the arithmetic it was:

- **`positions` [batch, seq]** — per-row RoPE positions. `pos_offset` is one
  number for the whole batch and cannot describe rows at different points in
  their own episodes.
- **`key_mask` [batch, keys]** — True attends, matching
  `scaled_dot_product_attention`'s own convention. Without `key_mask` the causal
  case stays `is_causal=True`, which is the fused kernel's fast path.

Rows are **left-padded**, so every row's next token sits at the same index and
the decode step is one gather. `transformer.attention_mask` allows a padding
query to attend to *itself*: a row masked everywhere softmaxes to NaN, and that
NaN reaches every later query through the cache.

`sample.pick_rows` is the row-wise twin of `sample.pick`. The grammar objects are
**not** row-wise — `OpenBlocks` and `_ToolCalls` already mask a 1-D logit vector,
so the rollout holds one per row and loops over them, which is less machinery
than a row-wise rewrite and no slower at these sizes.

**A round re-prefills the whole sequence.** Within a round the KV cache carries
the decode; between rounds it is dropped, because rows append different numbers
of observation tokens and a ragged cache is a class of bug rather than a saving
at these lengths.

## The return, and what cannot outrank an answer

```
R = score + clip(sum of step rewards, -C, +C)      C = policy.CLIP = 0.2
```

The invariant is `2C < 1`, asserted at import: **no incorrect episode can outrank
a correct one in its own group**, whatever it did with the shell. What the
shaping can still do is break ties among rollouts that all failed and among
rollouts that all succeeded — precisely the two places a binary reward is silent.

`A_i = (R_i - mean) / (std + eps)` within a group. A group whose returns are
identical contributes **nothing**, all-failed and all-passed alike, and is
skipped — with the skipped fraction reported, because that is the number saying
whether a run is learning or merely spending.

One update per batch, so the importance ratio is 1 by construction: no ratio, no
clipping, no stored old log-probabilities. The loss is REINFORCE with a group
baseline, plus `beta * KL(policy || base)` against the frozen model the run
started from — k3, so a single rollout cannot contribute a negative divergence
and pay the policy for moving away.

**A run needs both models resident.** That is the cost `SftConfig.measure_base_kl`
documents for the diagnostic version of the same idea, made mandatory.

## An episode that is absent, and an episode that is a zero

Different things, and confusing them corrupts a baseline.

- **Abandoned** — this side could not reach the sandbox. Absent from the batch,
  exactly as a closed episode is absent from a run's mean. It measures a
  transport fault, and counting it as a failure would hand the group a spread the
  model did not produce.
- **Out of rounds, or out of window** — graded on what it last said. That zero is
  earned, and it is in the batch.

The engine renders no caveat for the first case. The browser's driver hands the
model "that did not work" and carries on; doing that here would mean rendering an
observation in Python, which is the single thing this path may not do.

## Frames

An RL iteration publishes an ordinary `Progress` frame. `loss` carries the
policy-gradient loss for the chart's sake — it is an advantage-weighted sum whose
sign and size say nothing on their own, and it is not a quantity that goes down —
and everything a person reads is in `diag`: `pass_rate`, `signal`, `groups`,
`skipped_groups`, `kl`, `mean_return`, `episodes`, `abandoned`, `episode_rounds`,
`policy_tokens`. Field names travel to the chart unchanged, as every frame's do.

**`signal` is the diagnostic, not `pass_rate`.** A run whose signal has collapsed
is spending without learning however good its pass rate looks.

## Running one

```sh
POST /rl   # RlConfig; 202, queued like every other run
```

From the app it is `/jobs/new/rl` — a composer, not the gym page. Starting a run
from `/gyms/<id>` is one evaluation and always was; the loop is a queued job, and
the two are separate doors on purpose. See the frontend doc's *Two doors to a
gym*. The composer refuses `tasks_per_batch * group_size` above the gyms
service's `MAX_EPISODES`, because that product is one gym run and a run over the
ceiling fails on iteration one.

Before it is worth starting, ask whether the model is in the band where a group
can differ at all:

```sh
cd gyms && GYMS_DATABASE_URL=... .venv/bin/python -m scripts.gate --model <id>
```

Two things make that answer meaningless: **temperature above zero** is required,
or G rollouts are one rollout counted G times, and **no sampler seed** —
`ChatConfig.seed` pins the draw. `RlConfig` has no seed field for that reason.

Measuring a run is an ordinary gym run against `<the run's name> #n` at
temperature 0, on seeds from `holdout_base`. `RlConfig` refuses a `holdout_base`
inside the range the run will train on: a seed that trained is a seed the model
may have memorised the answer to.

## More gyms, and running them combined

`RlConfig.lanes()` returns one `(gym_id, task_types)` pair today, and **the whole
of what a combined run needs is for it to return more than one** — the pipeline
already creates a gym run per lane and pools their groups before the step.

`POST /gym/runs` refuses task types from two gyms and should keep refusing them:
a gym that has to know another exists is a gym nobody can add. So a run spanning
gyms is several gym runs pooled, not one run spanning gyms.

Pooling needs no cross-gym normalization, and that is what makes it cheap: an
advantage is standardized **inside a group** of rollouts of one seed, so two gyms
whose rewards live on different scales still contribute advantages of the same
size. `tests/test_policy.py` pins that groups from different runs are not one
group — pooled on the episode index, two gyms' episode 0 would be each other's
siblings, which is a baseline over two different questions.

What is still owed when that lands: `TaskWeight.weight` actually consulted across
lanes (a flat list is dominated by whichever gym registers the most task types),
frames grouped by `(gym, family, level)` so a combined run cannot hide one gym
sitting at zero, and the sandbox budget — just-bash holds 64 filesystems, so
`sum over lanes of tasks_per_batch * group_size` is bounded well under that.
