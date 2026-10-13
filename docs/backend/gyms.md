# Browser-driven gym runs

`/ws/gyms/runs/{id}` (`routes/stream.rs::gym_socket`, admin only) drives a whole
gym run: every episode in it, in order, each one generate → run the command →
generate again until the model stops calling the shell.

`api/src/gym.rs` coordinates the run; `gym/episode.rs` owns generation and
ordered command execution, `gym/protocol.rs` interprets episode records and builds
requests, and `gym/steps.rs` is the shared shell/observation adapter. It differs
from `turn.rs` in the two places a chat turn and a gym episode genuinely differ, and is a separate function for
exactly that reason:

- **Where an answer comes from.** A chat call goes to `tools::Registry`; a gym
  call goes to `POST /gym/episodes/{id}/step` on the gyms proxy, which runs it
  in *that episode's* filesystem and records it with its reward. What comes back
  is rendered by `tools::observation`, so the two are the same text.
- **How the next round is asked.** Rust stores a chat turn and loads authorized
  history, so a round sends one `tool_result`. An episode has no chat:
  the conversation is held in the loop and handed whole to the LLM engine's
  stateless `/chat/stream` every round, with tool turns naming their call so the
  LLM engine renders the `<|tool-result|>` block (`ChatTurn.tool`/`argument`).

Frames are the chat vocabulary plus four: `episode_start`, `episode_done`,
`run_done`, and the LLM engine's `done` relayed as **`round_done`** — an episode
takes a round per command, and a client counting `done` would see an episode end
every time the model ran `ls`. Every frame carries the `episode_id` it belongs
to, so a run of twenty tasks is one socket.

Bounded by `MAX_GYM_ROUNDS` (env, default 8). A model still calling when the
budget runs out has its last reply graded anyway rather than the episode being
dropped: an unscored episode would let a model that never stops looking around
be judged on the one task it happened to finish.

A dropped socket abandons the run — graded episodes stand, the rest are never
run — which is the bargain a chat turn already makes. Why the loop is here at
all rather than in the LLM engine, where RL will want it, is
[`../decisions/gyms-are-driven-by-the-backend.md`](../decisions/gyms-are-driven-by-the-backend.md).

**`POST /gym/demonstrations` is the other route that is not a pass-through**, and
for the same reason as the socket. It asks the gyms engine for expert
trajectories, renders each observation with `tools::observation` — the same call
a live round makes — and writes the result to the *data engine* as an ordinary
`sft` dataset, which the LLM engine then fine-tunes on with no reader it does not
already have. Three services and one job, and the assembly is here because this
is the only process where all three are reachable: the Python services never
speak.

The dataset description is optional; when omitted, the backend sends an empty
string to the data engine, whose import endpoint requires a string if supplied.

`api/src/demonstrations.rs` is that conversion, and the line it holds is worth
stating. The observation **body** is rendered here, because that is what serving
renders it with. The `<|tool-call|>` header around it is **not** — the stored
turn carries `tool` and `argument` beside the body, and the LLM engine's
`render_result` adds the header at build time. Spelling it here would be a
fourth copy of a layout this side has no reason to own. See
[`../gyms-engine/demonstrations.md`](../gyms-engine/demonstrations.md).

```sh
cargo run -p tools --example bash_once -- 'echo hello > f.txt; ls -l'
```
