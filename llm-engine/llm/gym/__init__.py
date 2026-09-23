"""Driving gym episodes from the LLM engine, for a batch nobody is watching.

A second rollout loop, deliberately. The browser's lives in
`backend/crates/api/src/gym.rs` and this one lives here, and they share the
tasks, the sandbox, the rewards, the grader and — the part that matters — the
renderer, which neither of them contains. What is duplicated is the forty lines
of ask-run-ask; what is not duplicated is anything that decides what the gym
*measures*.

The reason a policy gradient cannot use the other loop is one sentence: it is
computed from token ids and log-probabilities, and no HTTP frame carries them. A
trajectory that arrives as text has to be re-tokenized to be trained on, and a
re-tokenized rollout is a guess at what the sampler drew — under a masked grammar
not even a well-defined one. See
`docs/decisions/reinforcement-learning-in-the-llm-engine.md`.

Nothing here names a gym. A task type is a string the registry resolves, and the
one place `bash` is spelled is the backend, which is what lets a second gym be a
row in a table rather than a branch in a rollout.
"""
