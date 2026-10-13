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

Migrations keep all historical job UUIDs and dependencies. An unexpected legacy
`chat` row blocks migration rather than silently losing history. See the dated
[boundary decision](../decisions/execution-and-ownership.md) and
[cutover instructions](../backend/ownership-cutover.md).
