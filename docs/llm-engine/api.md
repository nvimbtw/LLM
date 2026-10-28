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

Generic run/model reads and run SSE derive visibility from the service JWT's
exact singleton `scope: ["administration"]` or public fallback. Rust mints that
claim from verified browser access; browser query fields never grant visibility.
Legacy, missing and malformed scopes use public projection, including open
development requests. Validation jobs are excluded in SQL before active/history
pagination and counts, and detail/events reject public access before replay.
Public run config/results/stage/logs and raw errors, plus opaque model metadata,
are omitted; measured model facts and finite known progress metrics remain.
Administrators retain complete metadata. A durable observer disconnect does not
cancel the job. Health hides an active validation's identity from public callers.

## Architecture catalogs (partial ISS-56 delivery)

`/blocks` and `/architectures` currently expose administrator-only collection,
detail, draft, metadata/archive and immutable revision reads. An exact
administrative scope and actor UUID are required, even in open-development
mode. Configured hop authentication verifies the token signature; open development
retains the existing optional unsigned-claim policy. Browser visibility queries
cannot authorize these routes. Listing and
saving never import custom Python or parse editable YAML. Invalid drafts remain
readable/editable, with independent draft/metadata versions. Stale mutations
return 409; builtin mutation returns 403; unknown identities return 404.

Collection reads bound search/filter/page inputs and use UUID keyset cursors
scoped to their original family/filters/administrative visibility. Revision
pages bind their cursor to the original resource and use immutable revision
numbers. Immutable source reads verify both resource and revision identities;
retained acceptance evidence survives original job pruning. Sources and graph
bodies do not appear in catalog summaries. Source UTF-8 byte overruns return
413; other interpreted input refusals return bounded structured 422 errors.

Literal `GET /architectures/schema` precedes UUID routes and exposes the actual
recipe schema plus source/YAML/work limits and binding syntax. Its typed
`ArchitectureSchema` also supplies the installed runtime's prospective default
profile and reported CPU/CUDA devices; these are editable validation inputs,
not capability evidence or device admission.

Explicit `POST /architectures/{id}/declaration` accepts the exact `DraftSubject`
and returns `RecipeDraftInspection` for those saved YAML bytes. It safely parses
declared parameters, ports and pinned references without importing dependencies,
resolving a model, constructing tensors or enqueueing work. Changed version/hash
returns 409; invalid YAML returns a bounded inspection-phase 422. The result is
always `inspection_only` and cannot authorize revision acceptance or training.
Ordinary draft reads/saves retain their no-parsing behavior. Block declaration
inspection executes Python and continues to use the supervised validation queue.
The versioned
[architecture contracts](architecture-contracts.md) define these interfaces.

`POST /blocks/{id}/validations` and `/architectures/{id}/validations` return 202
normal validation run envelopes. Submission locks the unarchived resource and
freezes exact saved bytes, immutable dependency sources, builtin compatibility
pins and the requested profile before committing one actor/request-idempotent
queue row. Reusing the same actor/request with different arguments returns 409;
replaying a pruned operation returns 404 and requires a fresh request UUID.
Inspection needs no constructor/profile values and is block-only. Full requests
may use pending tokenizer bindings, which return provisional metadata with
construction/capture/numerical stages skipped. They authorize neither acceptance
nor training. Serialized snapshots cap at 256 KiB for inspection and 8 MiB for
full validation; the snapshot ceiling includes JSON encoding and dependency bytes.

Final tokenizer facts must match a complete owned tokenizer build. The adapter
reads bounded data-only tokenizer bytes and checks their full SHA-256, actual
vocabulary size and requested reserved marker IDs. The legacy shortened corpus
fingerprint is not a version-one binding hash. Invented facts or client-supplied
derived provenance return 422. Admin-only
`GET /tokenizers/{id}/architecture-binding` returns the authoritative `FinalTokenizer`;
optional repeated `reserved` query names request engine marker IDs. Names use an
`_id` suffix; `pad_id` refers to the existing end-of-text padding token. Other
supported names identify their `<|name|>` marker. Missing builds return 404;
inconsistent bytes/sidecars or unavailable markers return 422. Tokenizer bytes
cap at 64 MiB; this operation loads no model weights.

`GET /validations/{id}` serves a completed, verified immutable report.
`GET /validations/{id}/graph` pages previously published receipt-bound scopes;
it never constructs a model or starts capture. Page sizes, scope/cursor lengths
and view/mode/signature are checked before artifact reads. Incomplete/failed jobs,
invalid/provisional evidence, unavailable artifacts and stale graph identities
return structured 409 errors. A completed-invalid validation is a done job with
diagnostics; a worker/cleanup failure is a failed job without acceptance evidence.

`POST /blocks/{id}/revisions` and `/architectures/{id}/revisions` accept only
the exact current draft/version/hash and dependency digest with fully bound,
valid retained evidence. Changed builtin implementations invalidate copied-source
evidence before acceptance. Identical source acceptance returns its existing
revision. Evidence, reports and graph bytes for retained receipts survive deletion
of the original run. Cancellation discards a completed report from the cancelled
job result and cannot retain acceptance evidence. Retry copies the frozen bytes,
minting fresh job/request UUIDs without adding `model_name`. Durable observer
disconnects do not cancel the worker.

Startup seeds nine read-only builtin block drafts and ordinary CPU validation
jobs. Their manifest-derived revisions are accepted only after the same verified
publication/evidence transaction. The immutable source view shows exact builtin
implementation Python; its revision hash covers the complete implementation
manifest. `GET /blocks/{id}/revisions/{revision_id}/copy-source` separately returns
eligible custom wrapper bytes as `BlockCreate`, without creating a resource.
Unavailable historical builtin implementations cannot be replaced with current
copy source. See [builtin identities](architecture-builtins.md).

Four builtin architecture drafts become eligible for their own ordinary checks
only after their required block revisions exist. Those checks use a real
published byte-alphabet tokenizer, with zero learned merges, for one concrete
acceptance profile. No receipt is manufactured at startup. Live completion and
startup adoption queue newly ready checks transactionally; retries retain exact
bytes and repeated startup is idempotent. Training with pending bindings still
finalizes its own tokenizer before optimization.

The existing queue owns pause, admission, cancellation, retry and stages/events.
The supervised handler verifies child cleanup before publication, then the runner
commits done and retained evidence in one transaction. Restart adopts only a
complete matching parent publication after exclusive process reconciliation;
corrupt or mismatched publications remain unsuccessful. See
[validation](architecture-validation.md). Actual supervised bootstrap acceptance and the full
authoring-to-training flow remain incomplete. Current in-process adapter checks
do not establish real worker/network or PostgreSQL concurrency acceptance.

## Recipe training submission (partial ISS-58 delivery)

`POST /train?request_id=<uuid>` accepts `architecture: {revision_id, parameters,
validation_run_id?}`. Selected recipes require administrative visibility, an
actor UUID and the request UUID. Explicit legacy structural fields cannot
accompany a selection, including values equal to defaults. Resume/SFT/RL inherit
their saved definition and reject new structural selections/overrides. Ordinary
flat requests remain supported through the deliberate legacy adapter.

A pretrain `resume_from` referencing a modular model also requires administrative
actor attribution and a request UUID. It inherits retained immutable definition,
source and final-tokenizer facts even after original training history is pruned,
the catalog root is archived or the named tokenizer build is deleted. The actual
saved package is verified by the execution adapter before dataset acquisition;
model/source loading stays in the worker. Replays retain their access check even
if the original saved input has since been deleted.

`POST /sft?request_id=<uuid>` uses the same administrative actor/request contract
when `base_model` or `resume_from` references a modular model. It inherits that
registered definition's exact source/tokenizer pin, with sparse original SFT
config and actor-scoped idempotency. Legacy SFT remains supported. Private
tokenizer-label adoption and recovery retain immutable enqueue pins and require
the [parent-prepared SFT authority](modular-sft.md) before publication.

`POST /rl?request_id=<uuid>` applies that same administrative actor/request
contract to modular saved policies. It retains the exact original reference and
source definition through new attempts and completed-iteration recovery; no
corpus grant is used. The [RL adapter](rl.md) owns bounded actor/resource effects,
durable task-seed position and fresh sandbox creation after recovery. Rust's
`/runs/rl` proxy preserves query, scope and upstream status.

Submission resolves public parameter defaults/types/bounds without importing
source, reads actual final tokenizer facts or pins this job's pending tokenizer
operation, and freezes the accepted root, full dependency source/declaration/
compatibility closure, values and evidence alongside the reserved model output
UUID. Serialized source snapshots cap at 8 MiB. Explicit prior evidence must
match source/dependencies/values/tokenizer/context and effective input/runtime
profile. Final in-job preflight remains required under actual worker admission,
including when prior matching evidence is retained.

`POST /runs/chain?request_id=<uuid>` prepares all source/evidence pins before
inserting any job and commits the complete chain together. Waiting SFT/RL steps
inherit the preceding model's frozen recipe. Actor/request replay returns every
original chain UUID; changed arguments return 409, and missing pruned history
returns 404 without recreating work. Retry and recovery copy existing immutable
pins after root archive/draft edits. Optional `request_id` also deduplicates
those operations. Recovery reserves a new output and uses the latest registered
generation; it does not select the current recipe draft.

Pinned pretraining now dispatches through a parent acquisition/admission handler
and the actual child training backend using the existing queue/device lease.
Verified package/evidence/definition/model registration precedes acknowledgement;
terminal success follows worker/session cleanup. Pinned SFT now uses the same
session/publication owner with private mask/probe grants and saved-definition
preflight. Pinned RL now dispatches its saved policy/original-reference packages
through the same queue/lease/publication boundary, with a separate parent effect
owner and finite journal admission. Actual supervised pretraining/SFT/RL and saved/process recovery remain open acceptance
gates in this sandbox; trusted adapter/numerical checks do not establish them.

## Existing execution routes

Saved modular models now route `/chat`, `/chat/stream` and `/measure/{request_id}`
through the reusable supervised [serving adapter](modular-serving.md). Public
published-model chat remains usable with the normal service scope. Serving
requires retained exact full-prefix evidence for batch one and the complete
trained context; unsupported declared context extension refuses. Interactive
disconnect signals cancellation before waiting for verified execution cleanup.

Administrator-only `/models/{model_id}/definition` returns `ModelDefinitionSummary`
with retained `validation_receipts`; `/models/{model_id}/graph` uses the same
bounded `GraphQuery` as validation graphs. These read immutable package metadata
without tensor/source loading or recapture. Legacy models without a modular
definition return 404 on these two administrative readers. Generic summaries
retain measured context/counts and null incompatible legacy five-field shapes.

Conversation/file persistence routes now live in Rust. Python exposes stateless
`POST /chat` and `/chat/stream`, plus `/chat/prepare` (validate/render/count new
turns and snapshot model attribution) and `/chat/context-floor` (return a history
index). The stream emits `token`, `tool_call`, `done` with generation summary,
or `error`. Rust commits the assistant message before sending its client-facing
`done`; Python writes no transcript.

Chat requests default to `max_tokens=2048`, shared by input preparation and
generation. An explicit `max_tokens` overrides that default. Generation can
finish earlier on an end-of-assistant/end-of-text marker or when the effective
context window is full. The limit includes reasoning and answer tokens together;
closing open markup blocks can append additional tokens after the limit.

These chat requests also accept authorized `project` facts and up to eight
retrieved `memories`, supplied by Rust. `/chat/prepare` validates their bounded
schemas and checks memory-token adoption before Rust persists new inputs.
`/chat/context-floor` accounts for their actual bounded fitting. No response or
stream-event fields change. See [project-memory.md](project-memory.md) for the
wire shapes, token ceilings and legacy tokenizer behavior.

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

Model listings, details and checkpoint responses also expose resolved
`architecture` (the five shape fields or null), `vocab_size` and
`parameter_count` (positive measured counts or null). `api/model_metadata.py`
prefers saved results, then requested shape fields, with metadata on disk and
base/resume ancestry filling legacy gaps. An ancestry walk stops at 25 models,
cycles, invalid IDs or missing ancestors. It reads no weights and mutates no
training provenance. The requested `max_vocab` never substitutes for a measured
vocabulary. Clients may estimate parameters from a resolved shape and vocabulary
when a measured count is unavailable.

`PATCH /models/{id}` accepts an optional `size`: `fast`, `balanced`, `flagship`
or null to clear. Omission preserves the category. It is an admin-authored display
choice stored as `parameters.display_size`, returned as `size`; it is neither
computed from parameters nor inherited. Edits preserve config, results and recovery
flags. Names and families reject blank labels; duplicate names return 409 without
partially saving other fields. Rust authorizes these shared-model writes as admin
operations and forwards them as the service's system caller.

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
`api/routes/attention.py` is the route. Legacy work remains in
`llm/orchestrate/attention.py`; saved modular models use the resident supervised
[attention owner](modular-serving.md#occurrence-aware-attention).

`GET /models/{model_name}/attention-points` returns `InspectionCatalog` on explicit
request. It discovers actual module/loop occurrences, optionally loading the
resident worker. `POST` also accepts `{text, point_id, head}`; supplying both a
point ID and a layer is invalid. Declared candidates carry no numerical proof
until their actual view/domain passes strict capture and numerical checks.
Successful modular matrices bind point, definition, profile and worker UUID plus
the actual validated sequence bound. Legacy numeric selection remains deliberate
and generic recipes never invent layer counts. Neither response exposes source
or captured graph data. Unknown/unsupported attention remains an explicit refusal.
The following recording details describe the legacy adapter.

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

Legacy inspection uses the existing cached model under device admission and
resets its recording flag on every exit. Modular inspection uses stateless
canonical probability operations and leaves normal recording state unchanged.
Interactive HTTP disconnect signals the request and drains its owned operation;
live browser-through-Rust disconnect behavior remains an integration gate.
# Queue pause

`POST /runs/pause?paused=true` stops admission of new queued work for the current
engine process. It does not cancel the active job. `paused=false` wakes the
runner; `GET /runs` reports that same process-local flag. Every restart starts
unpaused. Job records, ordering, dependencies and interrupted-job handling stay
durable; restarting does not itself resume interrupted training. Migration
`d8c10f6a21b4` drops the old `queue_state` table without altering job history.

## Vocabulary metadata

`GET /tokenizers/{id}` joins the index and complete build on disk and includes
recorded model references, following up to 25 base/resume ancestors. Missing
provenance stays unavailable. `PATCH /tokenizers/{id}` requires the system caller
after Rust admin authorization. `display_name` is required, trimmed, nonblank
and limited to 120 characters; optional `description` is nullable and limited
to 2,000 characters. Omission preserves it; null clears it. Duplicate names
return 409 and roll back every field. Unknown IDs return 404. A disk-only build
gains an index row on its first edit. Display changes never write artifacts,
change vocabulary fingerprints or rewrite historical training configurations.
