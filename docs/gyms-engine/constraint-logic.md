# Constraint logic — line order

`constraint_logic` generates ordering puzzles from a hidden permutation. It
derives true clues from that order, then narrows the set of possible orders
until exactly one remains. The answer is the person at a queried position.
Generation, verification and grading are programmatic; no LLM or tool is used.

| Rung | People in the line |
| --- | --- |
| `constraint_logic_easy` | Two. |
| `constraint_logic_medium` | Three. |
| `constraint_logic_hard` | Four. |

Clues state that one person is before or immediately before another, or that
someone is not first or last. The generator enumerates every permutation to
ensure the published clues admit exactly one full order. It never gives the
hidden order in the open episode. The demonstration reveals that order and ends
with `Answer: <name>`; the shared text grader checks only the final name.

Generation is deterministic for `(version, split, seed)`. Splits use separate
name pools and random streams but the same logical relations, so scores on the
test split measure new instances and names rather than a new kind of logic.
Version 1 must remain replayable. Single-attempt runs deduplicate by problem
key; grouped rollouts use one puzzle per group. The gym has no tools, supports
demonstrations and enters `general` through the source registry. Automatic
mastery progression remains math-only.
