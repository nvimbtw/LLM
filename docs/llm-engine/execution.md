# Durable execution and device admission

`execution_jobs` is the sole queue authority: stable UUID, kind, status, rank,
label, owner, dependency, progress/stage/error and timestamps. `training_runs`
holds pretrain/SFT/RL type, configuration, result, reserved target model UUID and
produced model reference, including a partial model saved on cancellation. `tokenizer_builds` has the corresponding vocabulary
contract. `benchmark_dispatches` contains the replayable dispatch request and
logical remote report ID; full reports and scoring live in bench-engine.

The internal `Run` facade delegates configuration/results to its domain record;
there are no second copies of execution status. The public `/runs` routes retain
IDs and queue/event contracts. Responses add `output: {type, id}`; `ref_id` remains
a compatibility projection, not an untyped database column. The unused chat job
kind is retired. Ordinary inference is a request/stream, with its transcript in
Rust, and creates no job row.

Cancellation/completion serialize on `execution_jobs` using `FOR UPDATE OF
execution_jobs`. Eager domain snapshots remain outer-joined reads; PostgreSQL
must not attempt to lock their nullable sides. No additional queue/lifecycle
owner or schema migration is involved.

The LLM process has one device gate, including cached model changes. Training
waits for the gate, evicts the chat cache only after admission, and releases
allocator memory before releasing the gate. Benchmark dispatch holds no device:
it waits for bench-engine through Rust, while the returning measurement request
holds the same gate. Interactive generation, attention and context-floor work
are refused as busy when admitted work or queued admission holds priority. HTTP
returns 409; an already-open SSE stream emits `error`.

A disconnected stream waits for its in-flight token step before closing its
generator and releasing admission. Token counting uses no weights and bypasses
the gate. Run a single LLM worker; multiple uvicorn workers would have independent
locks and caches and are unsupported.

Pause is process memory, initialized false on every restart. Resume wakes the
runner through its event loop. Existing active runs finish/fail at their ordinary
boundaries; pause never changes durable jobs or dependencies. Orphaned executing
jobs become failed after restart, while queued/waiting work survives. Cancelled jobs keep any produced output link but do not release dependent jobs.
Recovery creates a new training job using the previous job's latest checkpoint.

`POST /runs/chain` accepts at most 100 steps. It validates every effective
pretrain/SFT/RL/benchmark config and the initial pretrain tokenizer before any
queue insert, retaining only stated config values. Model-producing steps reserve
their output UUIDs; waiting consumers receive the actual model on promotion.
All queue/domain rows commit in one transaction before the runner is woken once.
A later configuration or insertion failure leaves no jobs from that submission.
Tokenization and validation cannot be chain steps, and a benchmark can only be
the final step because its output is a report. Immutable recipe/evidence pins
join that transaction, including waiting consumers and saved-definition inputs.

Pinned pretraining uses `api/training_handler.py` with the existing Runner lease.
Authorized acquisition stays in the parent; source, models and optimizer work
stay in the admitted child. Package adoption owns registration before ACK, and
`RegisteredCheckpoint` carries its committed live event without asking the legacy
progress sink to register it again. Terminal success follows session cleanup and
parent temporary-input cleanup. A cancellation winning the durable terminal row
lock remains cancelled even if its in-memory signal arrives after the last check.
Completed model bytes remain valid; cancelled jobs do not release successors.
Actual supervised saved/process acceptance remains unverified under the sandbox
socket restriction. Pinned SFT/RL still refuse their legacy numerical path.

Migrations keep all historical job UUIDs and dependencies. An unexpected legacy
`chat` row blocks migration rather than silently losing history. See the dated
[boundary decision](../decisions/execution-and-ownership.md) and
[cutover instructions](../backend/ownership-cutover.md).
