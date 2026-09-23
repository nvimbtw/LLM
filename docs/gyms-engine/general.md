# General — mixing source gyms across episodes

`general` selects questions from the registered source gyms. Each episode is
one source question with that source's instructions, tools and grading. It
does not combine questions within an episode, invent a common answer format,
or give tool-free tasks access to another gym's tools.

## Ownership and provider contract

`gyms/contracts.py` defines the boundary shared by every source:

- A task type accepts `Generation(seed, split, version)` and returns a `Task`.
- The task envelope carries source gym, task type, actual seed, generator
  version, prompt, system instructions, tools, effective split and an optional
  problem key. Source-specific state lives in `private`.
- `Task.public()` is the allowlist served while an episode is open. Answers,
  solutions, workspace files and grading expectations never enter it.
- `stored()` preserves the historical flat `task_config` layout. `restore()`
  reads old rows too: absent version means version 1, and absent tools use the
  source's declared tools. New rows snapshot both.
- A provider declares tools, supported splits, current generator version and
  whether it offers demonstrations. It implements `prepare`, `step`, `finish`
  and `demonstrate`. Unsupported actions are refused before executing anything.

The bash provider owns sandbox seeding, step rewards, expert moves and the
post-answer mutation check. Math prepares no environment, rejects steps, and
grades the final number. State tracking and constraint logic also have no tools;
they share a bounded final-name grader, while keeping separate generators and
private answers. Complete grading is always `provider.finish`; there is no
registry-level answer-only shortcut that could bypass file checks.

The current action wire and step storage still describe shell commands and
their raw output. A future gym using a different kind of tool will also need
that tool's wire/storage shape and a renderer in the backend. Another gym
using bash, or no tools, needs neither change. Observation rendering stays in
Rust; the Python services never call each other.

`gyms/registry.py` assigns each task type exactly one owner and rejects duplicate
gym/task ids. The general catalogue references the source rows with `gym_id`
on each task type; it registers no copies. `composite: true` distinguishes its
empty gym-level tool list from a source gym with no tools. Drivers always use
the episode's tools.

## Selection and weights

`POST /gym/runs` accepts the existing fields plus `gym_weights` and
`task_weights` for `gym_id: general`:

```json
{
  "gym_id": "general",
  "model_name": "<model-id>",
  "task_types": ["bash_read_easy", "math_addition_easy", "math_word_easy"],
  "gym_weights": {"bash": 1, "math": 2},
  "task_weights": {"math_word_easy": 3},
  "count": 32,
  "group_size": 8,
  "seed_base": 100,
  "kind": "rl"
}
```

Task types may be omitted for `general`, meaning all currently registered
source types. Source gyms still require a nonempty task list and preserve
their existing cycling behavior, including repeated ids used by the trainer.
An explicit general list must contain unique ids. The UI starts at the easy
rungs and lets the caller select other tasks.

Selection happens in two draws: first a source gym, then a task type within
that gym. Both default to equal weights. Adding more task types to one gym
does not increase that gym's share. Weights are relative, finite, positive and
at most 1,000,000; omitted entries weigh 1. Keys must refer to selected sources
or task types. Exclude a source by deselecting its tasks, rather than giving it
a zero weight. Frequencies are probabilities, not per-batch quotas; a small
batch can contain only one source.

The selection stream is deterministic, versioned, and separate from task
generation. Source and type ordering are canonical, so merely reordering the
request does not change a general run. The resolved catalogue and weights are
saved once on `gym_runs.selection` by migration `0005`. Historical runs keep
that column null. Adding a source later affects new all-source requests, not
the saved plan or episodes of an existing run.

## Groups, replay and splits

The selector draws once per group. With `group_size: 8`, a selected question
is generated once and used for eight attempts, with eight independent
workspace keys where needed. The group index determines selection and the
initial task seed (`seed_base + group`); model sampling remains stochastic.

`gym_runs.gym_id` is `general`; `gym_episodes.gym_id` is the actual source.
The existing episode column stores this distinction without another table or
another episode loop. Every episode stores the generated task itself, its
actual seed, version and effective split. Replay should use those recorded
values, not today's registry defaults. Generator version 1 retains the existing
bash and math questions; new reasoning sources start at version 1. Unknown
requested versions are rejected.

The default split is `train` for RL and `test` for evaluation. Math, state
tracking and constraint logic support train/validation/test streams and
deduplicate problem keys within a task type on single-attempt runs, including
mixed runs. This can advance the recorded seed beyond the initial seed.
Grouped attempts deliberately repeat questions.
Bash declares no split support and records `split: null`; a mixed evaluation
does not imply that bash has math's wording-based split separation.

## Demonstrations, training and results

`POST /gym/trajectories` accepts the same source/type weights and optional
general task list. Every provider returns the same envelope:
`{count, selection, trajectories}`; each trajectory includes source, seed,
version, effective split, instructions, raw moves and the expert answer.
Demonstrations always request the training split. Math, state tracking and
constraint logic have no moves and never touch the sandbox; bash reuses one
cleared workspace for a batch.
The backend stores the selection policy as dataset provenance and the source
gym on each conversation. These requests create SFT data, not scored runs.

RL configuration uses `gym_id: general`, explicit `task_types` (strings or
`{id, weight}` entries) and optional `gym_weights`. The LLM engine forwards
the whole candidate pool and weights to the gyms engine. Pre-sampling task
types in the trainer would change the source mixture, so only ordinary source
gyms retain that older path. Automatic mastery progression remains math-only.

Run responses include `source_scores`, keyed by source gym, with
`episode_count`, `graded`, `correct` and `score` alongside the overall score.
Only graded episodes contribute to either mean; unmeasured sources have null
scores. The general evaluation view also shows live results by source. RL progress
frames and the saved training result retain source scores for the latest batch,
shown separately on the training page.

## Adding a source

1. Implement seeded generators and the provider contract in the new gym's
   module. Declare split and demonstration support explicitly.
2. Register its provider and unique task types. The general catalogue includes
   them automatically; shared HTTP routes need no source-specific branch.
3. Run the shared contract tests in `tests/test_general.py` and add tests for
   that source's environment, grading and demonstrations. The shared suite
   enumerates registered source types rather than maintaining a second roster.
4. Keep old generator versions replayable when introducing new semantics.
