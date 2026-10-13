# Routes — the contract the backend codes against

The LLM engine's public surface. Split out of [`README.md`](README.md) when that
file passed the ~250-line rule (see [`../README.md`](../README.md)); the README
keeps the map and this file keeps the routes.

Every route is behind `Depends(auth.require_service)` — declared once on the
`FastAPI` app rather than per route, so a route is authenticated because it
exists and not because someone remembered to decorate it. The backend signs a
short-lived token naming who it acts for (`LLM_ENGINE_SERVICE_SECRET` on both
sides, unset leaves the LLM engine open and it says so at startup). A WebSocket
upgrade is an HTTP request, so the socket routes are covered by the same check.
FastAPI's own `/docs`, `/redoc` and `/openapi.json` bypass an app-level
dependency, so all three are turned off.

Conversation/file persistence routes now live in Rust. Python exposes stateless
`POST /chat` and `/chat/stream`, plus `/chat/prepare` (validate/render/count new
turns and snapshot model attribution) and `/chat/context-floor` (return a history
index). The stream emits `token`, `tool_call`, `done` with generation summary,
or `error`. Rust commits the assistant message before sending its client-facing
`done`; Python writes no transcript.

`POST /measure/{id}` returns model likelihood, unconstrained prompt samples and
ablation measurements. `/measure/{id}/cancel` signals a cooperative stop. These
are model primitives used by bench-engine through Rust, with no scoring policy
or benchmark table here. The queued `/eval` route accepts only `language` and
dispatches through Rust. [Execution admission](execution.md) covers busy responses.

## Refusing the shell for a turn: `max_tool_calls`

`ChatConfig.max_tool_calls` is the ceiling on how many tool calls one turn may
open; absent leaves `MAX_TOOL_CALLS_PER_TURN`. **`0` is a real value and not
"unset"** — it masks the tool-call opener out of the grammar before the first
token, so the turn can only answer in prose.

That zero is the whole of a gym episode's terminal round. The alternative was
telling the model to stop searching and answer, which a model this size does not
reliably do: 22 of 30 episodes in the first measured gym runs ended with the
model still calling the shell. The mask is enforcement where an instruction was
a suggestion, and it costs nothing at any other budget — the check
(`_ToolCalls.mask`) was already there for the per-turn budget and now reads a
field instead of a constant. See
[`../decisions/reinforcement-learning-in-the-llm-engine.md`](../decisions/reinforcement-learning-in-the-llm-engine.md).

## Listing models, and keeping a run's intermediates out of the way

`GET /models` is the join of two authorities — a directory under
`artifacts/models/` is what can be loaded, a `models` row is what a person calls
it — so a directory with no row is a model trained outside the queue and a row
with no directory is one whose weights were removed.

**`checkpoints: bool = False` filters both halves of that join.** A run may retain explicit snapshots alongside
its latest recovery point and each one is a model in every respect, directory
included, so filtering only the rows would let every one of them back in through
the disk half — reported as an unregistered model, which is the one shape this
listing treats as interesting. `_joinable_dirs` therefore asks the database for
`checkpoint_ids` and hides those directories too. The guarantee the frontend
rests on is this one: **a default `GET /models` never grows because a run was
long**, so no grid, picker or benchmark list has to learn what a checkpoint is.

`GET /models/{id}/checkpoints` answers with that model's intermediates, ordered
by `checkpoint_step` — not by `created_at`, which two checkpoints of a fast run
can share to the resolution the column stores. It is a route of its own rather
than a flag on the model detail because the detail is loaded on every chat and
the ten extra rows are wanted on exactly one page.

`DELETE /models/{id}` takes the intermediates with it, rows and directories
both, and answers with `checkpoints_removed`. Nothing else can reach them once
the parent is gone: they are absent from every default listing and the only path
to one is `checkpoints_of` on the row being deleted. The route reads that list
*before* the delete, because afterwards nothing on disk says which directories
were intermediates.

See [`README.md`](README.md#checkpoints) for what a checkpoint is and the
cadence a run takes them at.

## Context extension (NTK-aware RoPE)

`ChatConfig.context_multiplier` (default `1.0`) stretches a turn's context past
the model's trained `ctx_window`. `Transformer.extended_rope` builds a fresh
`(cos, sin)` table at `round(ctx_window * context_multiplier)`, scaling the RoPE
base by the length ratio raised to `head_dim / (head_dim - 2)` — the tables the
model trained on stay exact, and only the extrapolated tail stretches. `orchestrate/chat.py`
passes that table into every `Transformer.forward` call for the turn via the
`rope=` override, and uses the stretched length everywhere `ctx_window` used to
bound the prompt and the generation loop.

Two things keep this from touching what SFT and pretrain already guard:

- **Nothing is written into the model's own `rope_cos`/`rope_sin` buffers.** They
  are shared by every request against a cached model (see `orchestrate/chat.py`'s
  `_CACHE`), and mutating them would race two concurrent turns extending to
  different lengths against each other.
- **It never reaches what is saved.** `ctx_window` and `rope_base` still come
  from `arch`, exactly as `SftConfig`'s docstring insists — this is a per-turn
  inference parameter, not an architecture change, and a model file written
  during that turn records the length the weights were trained at. It buys
  headroom against silent truncation (`build_prompt` in `llm/chat/template.py`
  evicts oldest turns first), not the quality of a model actually trained at the
  longer length.

**The last resort is a hard cut, and it is a cut rather than a crash.** Evicting
whole turns cannot save a conversation whose *latest* turn alone busts the
window, because that turn is the one that must be kept — so `_hard_cut` brings
the rendered ids under the allowance directly. The trailing `<|assistant|>` is
never cut (a prompt missing it has not asked the model to speak), the system
block goes only once the body has nothing left to give, and the freshest end of
the turn is what survives. Until this existed the oversized prompt was handed to
`Transformer.forward` as it stood and came back as
`position N exceeds context window M` — a crash where the docstring promised a
truncation, reachable by any model with a small window and a long question.

## `/prune`'s floor: `context_floor`, and why nothing is written down

Rust’s `GET /chats/{id}/context-floor`, using Python’s `POST /chat/context-floor`, answers "where would this chat's history be cut
if it were fitting the model's own `ctx_window` rather than whatever
`context_multiplier` extends it to" — `llm.chat.template.context_floor` runs the
identical newest-to-oldest walk `build_prompt` already does (`_fit_history`,
factored out so the two cannot disagree), just with the un-multiplied allowance.
It is read-only: nothing is stored, on either side of the call.

The frontend is what remembers the answer — a message id, kept in
`localStorage` per chat (`lib/context-floor.ts`) — and resends it as
`history_floor` on every later turn. Rust slices authorized stored messages
against it before `ChatConfig.turns` is built, so
history before the floor never reaches `build_prompt` at all, regardless of how
wide a raised `context_multiplier` would otherwise let it reach. A floor that
doesn't match any stored message — a stale id, another chat's — falls back to
the full history rather than erroring: a client's memory of a prune is a
convenience the turn it's answering should not fail over.

This buys the whole feature without a migration: no chat gains a column, no
message is ever deleted, and a lost `localStorage` (a cleared browser, a second
device) just means the next turn falls back to full history — the same
durability every other chat setting in this app already has. If a prune ever
needs to survive that, this is the point to reconsider: persist the floor
against the chat instead of trusting the client to keep resending it.

**Snapped past a lone `Role.TOOL` turn.** A naive cut can land exactly on a
tool's answer, and dropping the assistant turn that called it while keeping the
answer would leave a `<|tool-result|>` naming no call — see the tools section of
[`README.md`](README.md) for why that shape is never allowed to exist.
`context_floor` shifts forward past it rather than trying to pull the call back
in.

## Reading a head's attention

`POST /models/{model_name}/attention` takes `{text, layer, head}` and answers
with that one head's attention matrix and the token strings on both axes — the
labels are most of the value; an unlabelled matrix is a grey square.
`api/routes/attention.py` is the route, `llm/orchestrate/attention.py` the work.

**Nothing is recorded by default and this endpoint is the only thing that ever
changes that.** `Attention.record_attention` — the same flag the diagnostics
collector uses, not a second mechanism — is flipped on for the asked-for block
alone, for one `no_grad` forward, and off again in a `finally` so a pass that
raises cannot leave the process off the fused kernel. Every other layer keeps
`scaled_dot_product_attention` for that forward, and so does the whole model
after it.

Three constraints shape the rest of it:

- **On demand, never streamed.** At 512d/8h/6L/ctx512 the full tensor is ~8 MiB
  a layer and ~50 MiB a sample, so recording a run would carry ~1.5 GiB. One
  `(layer, head)` matrix at ctx 512 is 512×512 fp32 — exactly 1 MiB — and is
  affordable precisely because a request is what brings it into existence.
- **The sequence is bounded**, at `min(MAX_SEQUENCE, model.ctx_window)` with
  `MAX_SEQUENCE = 512`. The matrix is quadratic in that number, so it is the
  number that has to be capped rather than a byte size: 512 tokens is 1 MiB of
  fp32, 1024 is 4, 4096 would be 64 out of a route any signed-in person can
  call. Longer text is a 400 naming both the count and the ceiling, not a
  truncation — a silently shortened text answers a question nobody asked.
- **The matrix travels as base64 fp32, row-major**, not as JSON numbers. 1 MiB
  of weights is 1.33 MiB base64 against 2.5–3 MiB printed to six decimals and
  ~5 MiB unrounded, and it needs no decision about how much precision to discard.
  fp16 would halve it again and was not taken: no browser has a native half
  decoder, and it would put "the rows sum to 1" inside a tolerance instead of at
  the exactness the model computed it to.

The model is the one `orchestrate/chat.py` caches, so a chat generating in
another thread during the window takes the explicit-softmax path for one block.
That is the same arithmetic (`bench.py` pins the two paths to 1e-5) and it is
slower; the alternative is a second resident copy of the weights, which is the
cost that cache exists to avoid.
# Queue pause

`POST /runs/pause?paused=true` stops admission of new queued work for the current
engine process. It does not cancel the active job. `paused=false` wakes the
runner; `GET /runs` reports that same process-local flag. Every restart starts
unpaused. Job records, ordering, dependencies and interrupted-job handling stay
durable; restarting does not itself resume interrupted training. Migration
`d8c10f6a21b4` drops the old `queue_state` table without altering job history.
