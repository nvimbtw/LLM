# A gym episode is driven by the backend

- **Date**: 2026-08-29
- **Status**: Implemented

## Context

A gym episode is the same shape as a chat turn: generate a reply, run whatever
it asked the shell for, hand the output back, generate again. That loop already
exists — `backend/crates/api/src/turn.rs` — and the question was whether the
gym should reuse it or grow its own, and if its own, in which process.

The obvious place is the **LLM engine**. It owns the models, it is where the
sampler is, and reinforcement learning — the reason the module is called a gym
at all — needs the rollout in the same process as the gradient, because a
policy gradient is computed from token ids and log-probabilities that no HTTP
frame carries. A driver written in `llm-engine/llm/gym/` today is the driver RL
wants tomorrow. That argument is real and it is why a copy of the gym library
was first written in the LLM engine.

It is not enough, for one reason.

**What the model reads after a command is rendered in Rust.** A shell's answer
is not `{stdout, stderr, exit_code}` — it is
`tools::markup::neutralize`d output inside a `<|codeblock|>`, preceded by a
`<|warn|>` when the exit code is news or the output had to be cut, all of it
clamped to `bash_budget_chars`. That is `crates/tools/src/bash.rs`, it is what
every chat round has shown the model, and it is what SFT taught.

An LLM engine driver would have to render that in Python. Two renderers of one
layout is the exact failure `llm/chat/tools.py` already warns about for tool
calls, and it is worse here than there: the gym's entire purpose is to be the
place a model practices what it does in production. A gym whose observations
are a near-copy of the real ones trains a model for a shell it will never be
served.

## Decision

**The backend drives an episode**, over a WebSocket at `/ws/gyms/runs/{id}`,
and the gyms engine serves one move of that loop per request.

- `backend/crates/api/src/gym.rs` holds the loop. It asks the LLM engine's
  **stateless** `/chat/stream` for each round — an episode is not a conversation
  anybody should find in their chat list — running the shell through the gyms
  engine and rendering what comes back with `tools::observation`, which is
  `bash.rs`'s own renderer, exported for exactly this.
- `gyms-engine/` keeps what it is uniquely able to keep: the tasks, the sandbox and its
  seeding, the grading, and the record of every run, episode and step.
- The LLM engine gained one thing: a `ChatTurn` may name the `tool` and `argument`
  its text answers, so the *call* block is rendered by
  `llm.chat.tools.render_result` and not by the backend. The two renderers meet
  at the boundary and neither crosses it — Rust renders the tool's output, Python
  renders the block that output sits in.

## What this costs

- **The RL driver will be a second loop.** When training arrives it will run in
  the LLM engine, reaching `/gym/*` through the backend the way `llm/corpus/remote.py`
  already reaches the data engine. It will share the tasks, the rewards and the
  grader — every part that defines what the gym *measures* — and not the loop,
  which is forty lines of "ask, run, ask again". That is the trade: a duplicated
  loop is cheap and a duplicated observation format is not. How that driver
  reaches the shell without rendering anything itself — and why the "calling the
  backend to render" rejection below does not bind it — is
  [`reinforcement-learning-in-the-llm-engine.md`](reinforcement-learning-in-the-llm-engine.md).
- **`turn.rs` and `gym.rs` are two loops in one crate.** They share their frame
  vocabulary and their tool rendering, not their body. A chat turn is stored by
  the LLM engine and comes back as history, so it sends one `tool_result` per round;
  an episode holds its own conversation and sends the whole thing every round.
  Threading a flag through one function to serve both would have been more
  machinery than the duplication it removed.
- **A dropped socket abandons the run.** Episodes already graded are stored and
  the rest are never run. Reattaching to a run in flight is not built; a reload
  reads the record. This is the same bargain a chat turn makes, and it is the
  first thing to revisit if runs get long enough that losing one hurts.

## Rejected

- **The LLM engine drives, rendering observations in Python.** The whole argument
  above.
- **The LLM engine drives, calling the backend to render.** A round trip per command
  purely to format a string, and it leaves the loop split across two processes
  anyway.
- **The gyms engine drives, calling the backend for generation.** Adds a
  service-to-service hop the architecture does not otherwise have in that
  direction, and still leaves the rendering in Python.
- **Reusing `turn.rs` verbatim with a "this is a gym" flag.** See above.
