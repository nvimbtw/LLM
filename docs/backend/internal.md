# Service calls into Rust

`api/routes/internal.rs`, behind issuer-scoped service extractors, admits calls
from the LLM and benchmark services. The LLM engine reads documents there that the data
engine owns — it cannot ask the data engine itself, because the Python
services never speak, and it cannot be handed them in advance, because a queued
run needs them at a moment nothing on this side knows ahead of time. It drives
gym episodes there for the same reason, and one more of its own: see
*The gym family* below.

Three things make that safe to have, each enforced somewhere different:

- **The audience claim, in the new direction.** An inbound token says
  `aud: backend`, so a token this server minted *for* a service cannot be
  replayed at it. `contracts/backend-token.json` is the shared vector — minted by
  PyJWT, verified by `jsonwebtoken`, the only token in this project that Python
  signs and Rust checks.
- **The issuer list, named at the call site.** `InboundSecret::caller` takes the
  issuers it will accept. The dataset/gym extractor accepts
  `llm_engine::AUDIENCE` alone; benchmark measurement routes use `BenchCaller`
  and accept `bench_engine::AUDIENCE` alone. The data engine holds a valid secret and still cannot read through the backend,
  because it has no business asking.
- **`act` survives the hop.** The caller the LLM engine names is re-minted into the
  token going out to the data engine, so a run reads exactly the datasets the
  person who queued it could have read. `crates/service` names no service, so
  the crate that knows both is `api` and only `api`.

**The dataset routes are reads only.** A run consumes datasets; creating,
cleaning and deleting them stay a person's, under `/data`.

### The gym family

`GET /internal/gyms` forwards the task catalogue for adaptive practice.
Tool-free episodes (`tools: []`) receive a single terminal response with tool
calls masked from the outset. The gyms engine also refuses steps for them.
Demonstration requests forward `gym_id`, `gym_weights` and `task_weights`;
math trajectories contain no moves,
so their checked answer and short working pass through as one assistant turn.
Source gym, generator version and split travel in conversation metadata; the
resolved selection policy is retained in dataset provenance. A general run
uses each episode's tool policy, just as an ordinary source run does. See
[math gym](../gyms-engine/math.md).

`/internal/gym/*` is not read-only and cannot be — a rollout runs commands and
grades answers — but every one of its writes lands in the gyms engine's own
record, which is the module that owns them. Nothing new is stored here.

**The step route answers with the rendered observation, and that is the whole
reason it exists.** `POST /internal/gym/episodes/{id}/step` takes
`{tool, argument, reply}` and comes back with what the model should read,
laid out by `tools::observation` — the same function the browser's gym socket
and the live chat `bash` tool render with. `gym::play` is the shared body; the
socket loop and this route differ only in what they do with the result.

So the LLM engine drives a rollout without ever laying an observation out, which is
what reconciles two things that read like a contradiction:
[`../decisions/gyms-are-driven-by-the-backend.md`](../decisions/gyms-are-driven-by-the-backend.md)
rejected *the LLM engine drives, calling the backend to render* — priced as a round
trip taken purely to format a string — and
[`../decisions/reinforcement-learning-in-the-llm-engine.md`](../decisions/reinforcement-learning-in-the-llm-engine.md)
requires the LLM engine to drive, because a policy gradient needs the token ids and
log-probabilities the sampler drew and no HTTP frame carries them. The rejection
priced a round trip the LLM engine would otherwise not make. It makes this one
anyway — the sandbox is behind this server — so the rendering rides a hop that
already had to happen and costs nothing.

The tool name is not decoration. `play` routes on it, and a call to a tool the
gym does not implement comes back as a caveat rather than as a failed episode.
That is where a second gym's renderer goes: a row beside `bash`, not a branch.

`BACKEND_SERVICE_SECRET` unset leaves `/internal` open and the server says so at
startup — the same bargain each Python service strikes with its own secret unset,
and the same trust boundary, since this server binds loopback by default.

**That default is what makes the bargain safe, so a container cannot strike it.**
`BIND_ADDR` overrides the binding and `docker-compose.yml` sets it to `0.0.0.0`,
because a published port cannot reach a loopback binding. Which is why the same
file *requires* `BACKEND_SERVICE_SECRET` and all service secrets rather than
letting them default to unset — see
[`../decisions/one-command-development.md`](../decisions/one-command-development.md).
