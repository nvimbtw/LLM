# State tracking — object transfers and swaps

`state_tracking` generates short histories of people passing and swapping named
objects. The answer is the final holder of the object named in the question,
calculated by applying every event to a private state map. No LLM creates tasks or grades
answers, and no tool or sandbox is available to the episode.

| Rung | Generated history |
| --- | --- |
| `state_tracking_easy` | One object and one transfer. |
| `state_tracking_medium` | A transfer of each object, then their holders swap objects. |
| `state_tracking_hard` | Two transfers of each object, with two swaps interleaved. |

The prompt states each object's initial holder, the events in order, and the
queried object. Each transfer names the current holder. Each swap exchanges
the objects without naming their holders, so the model must track both states.
The final holder and a short state trace stay private until the episode is
graded. Demonstrations use the trace, ending with
`Answer: <name>`. The grader accepts only that name on the final line, with an
optional `Answer:` prefix; it does not search earlier reasoning for a match.

Generation is deterministic for `(version, split, seed)`. Train, validation and
test use different name and object pools. This separates surface wording, while
the same transfer rule is shared across splits. Version 1 must retain its
semantics for saved episode replay. Single-attempt runs use a problem key to
deduplicate histories; grouped rollouts repeat the identical task.

The provider uses `gyms/text_runtime.py`, shared with constraint logic. It
declares no tools and offers demonstrations through the existing general-gym
contract. Automatic mastery progression remains math-only; select this gym's
rungs explicitly for RL.
