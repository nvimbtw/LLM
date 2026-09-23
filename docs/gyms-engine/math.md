# Math — exact answers, no tools

The math gym generates arithmetic, expressions and short word problems. It
owns the problem, the private solution and the grader; the LLM engine owns the
adaptive training schedule. No Python service calls another Python service.

## Tasks and difficulty

`gyms/math/tasks.py` provides six families with easy, medium and hard rungs:

| Family | Easy | Medium | Hard |
| --- | --- | --- | --- |
| Addition | operands 0–19 | two digits, guaranteed carry | up to three digits |
| Subtraction | nonnegative, operands 0–19 | guaranteed borrowing | up to three digits, possibly negative |
| Multiplication | small facts | two digits times a small factor | up to three digits times two digits |
| Division | exact small division | larger dividends | larger exact division |
| Expressions | three-term addition | multiplication precedence | parentheses before multiplication |
| Word problems | one-step addition | give away, then receive | multiply then add, with irrelevant colour information |

The runtime implements the [shared provider contract](general.md#ownership-and-provider-contract).
Math keeps these same semantics when selected by the general gym.

Every task has a numeric integer answer. Fractions, remainders, equations and
proofs are not task families yet. The grader accepts exact numeric fractions
as alternate spellings, which is different from generating fraction problems.

Task types advertise prerequisite ids. Each rung requires its previous rung;
expressions also require basic addition and multiplication, and word problems
require basic addition and subtraction. The catalogue advertises `tools: []`.
The service creates no filesystem and rejects `/step` for a math episode.
Both generation drivers mask tools out from the first token.

## Grading

One final numeric answer is required, preferably `Answer: 5`. Bare numbers,
decimal equivalents, fractions, and a final `\boxed{5}` are accepted. Working
may precede it. The last nonempty line after removing model markup is graded;
thinking blocks do not count as answers. A correct number earlier in a wrong
solution earns nothing. Multiple alternatives, executable expressions,
division by zero and oversized answers are refused. Comparisons use exact
rational arithmetic, with no floating-point tolerance. Rewards are 0 or 1;
there is no reward for length, formatting, closeness or shell actions.

The numeric grader cannot certify the reasoning. A correct final answer with
bad working still scores 1. Validation reports invalid answers and generation
truncation separately from correctness.

## Reproduction and splits

`POST /gym/runs` accepts `split: train | validation | test`; absent means
`train` for RL batches and `test` for evaluations. Math stores the split,
generator version and actual seed in each episode's existing `task_config`.
The answer and solution remain private until grading; only the answer is
revealed afterwards. No database migration is required.

Each split has a separate deterministic random stream and wording pool.
Single-attempt runs deduplicate problem keys within each task type, using a
bounded seed search. Consequently a math episode's actual seed can differ
from `seed_base + index`; use its recorded seed to reproduce it. Grouped
training attempts retain identical problems within each group.

Elementary facts can recur across splits and checks because the fact space
is small. These scores measure fluency and robustness to wording, not proof
of unseen-fact or general reasoning ability. Test wording is never used for
SFT demonstrations. Keep generator version 1 stable for historical runs;
changing its semantics requires a new version and compatible replay.

## Demonstrations and practice

`POST /gym/trajectories` accepts `gym_id: math` and math task ids. It returns
`{count, selection, trajectories}` with checked, tool-free demonstrations and
an empty moves list on each. Basic arithmetic
ends immediately with the answer; expressions and word problems include short
computed steps. The backend's existing `/gym/demonstrations` route stores them
in the data engine as ordinary SFT conversations. Naming an existing dataset
appends; this does not queue training.

The practice composer exposes this action, manual task selection, and automatic
progression. See [the LLM engine's curriculum](../llm-engine/math-practice.md) for
placement, promotion, review, fixed tests and checkpoint continuation.
