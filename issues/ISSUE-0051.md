---
id: ISSUE-0051
title: "Design the epic: a resilient GPU job queue (train/inference) with a /queue API + live position"
scope: backend
priority: P1
status: todo
rule: none
location: backend/src/jobs.rs backend/src/routes/
blocked-by: []
blocks: []
---

## Description

**Meta / design issue.** The deliverable of THIS issue is a *designed epic* (parent
tracking issue + `blocked-by` chain of small, independently-verifiable children),
in the same shape as ISSUE-0030 / ISSUE-0040. Do not build the queue under this id.

### Why we need it

Today there is no queue. `JobRegistry::start` (`jobs.rs:102`) registers a run and
**immediately `std::thread::spawn`s** a worker that does blocking GPU/FFI training.
There is nothing serialising GPU access:

- Two training runs (or a training run and inference) can hit the GPU at once and
  contend for VRAM / CUDA context — undefined behaviour, OOM, or a crash.
- Inference (chat) and training are entirely separate paths; chat has no notion of
  "the GPU is busy training, wait your turn."
- There's no way to ask the server *what it's working on*, how deep the backlog is,
  or where a given request sits.

The user wants a **single resilient queue** in front of the GPU: heterogeneous job
types (train, inference, tokenize, eval, …) that wait on each other for exclusive
(or capacity-limited) GPU use, plus visibility over REST and a live socket so a
waiting client (e.g. chat) can show "you're 2nd in line".

### What the epic must design

A **GPU work queue** — one conceptual scheduler that owns GPU admission:

- **Job model.** Generalise the current `Job`/`JobStatus`/`JobState`
  (`jobs.rs:31-70`) to a typed job with a `kind` (train | sft | inference | …), a
  priority, a queued/running/finished lifecycle, and an enqueue time. Existing
  training jobs become one kind on the queue rather than an immediate thread spawn.
- **Scheduler / admission.** A component that decides what runs next and enforces
  GPU exclusivity (start with **strict serialisation** — one GPU job at a time — and
  design so a future capacity model, e.g. small inference jobs sharing, is
  possible). Ordering policy (FIFO vs. priority; can interactive inference preempt
  or jump ahead of a long training run?) is an explicit design question for the
  epic to resolve.
- **`GET /queue` API.** Reports the current running job, the ordered backlog
  (kind, position, enqueued-at, owner), and counts. Consider a per-job
  `GET /queue/{id}` for a single request's position/ETA.
- **Live position socket.** A subscription (reuse the existing per-job socket
  pattern in `jobs.rs`, which already snapshots state for attach/re-attach) so a
  queued client gets pushed position updates and a "now running / done" signal —
  the mechanism chat waits on before its request executes.
- **Resilience.** Define behaviour on: worker panic / CUDA fault (already partly
  handled — a spawned worker catches and marks `Failed`, `jobs.rs:124`), cancel
  while queued vs. while running, server restart (queue is in-memory today — does it
  need persistence, or is a lost-on-restart queue acceptable?), and starvation
  (a flood of one kind blocking another).

### Design questions the epic must answer (not this issue)

- **Ordering/priority:** FIFO, or priority tiers? Can interactive inference jump the
  queue ahead of training, and if so, does a running train job get preempted
  (checkpoint + yield) or only jumped once it finishes? Preemption is a big lever —
  decide in/out of scope.
- **Concurrency model:** strict one-at-a-time (simplest, safe) vs. a capacity/VRAM
  budget allowing small jobs to co-run. Start strict.
- **Persistence:** in-memory (lost on restart, matches today) vs. DB-backed durable
  queue (survives restart, re-drives on boot). Trade resilience vs. complexity.
- **How chat integrates:** does the chat route enqueue an inference job and await
  the socket, or is there a fast path? Where's the boundary between "queued request"
  and the current synchronous inference path?
- **Backpressure / limits:** max queue depth, per-user limits, timeouts for stale
  queued jobs.

### What to reuse

- The existing `JobRegistry` + per-job `Mutex<JobState>` + snapshot-on-attach
  socket pattern (`jobs.rs`) is the seed — the queue generalises it, it isn't a
  greenfield rewrite.
- The socket/reattach machinery already exists for training; the queue-position
  socket should follow it rather than invent a new transport.

### Non-goals

- Not designing the mixed-precision training change (that's ISSUE-0050).
- Not multi-GPU / distributed scheduling — single GPU, single host.
- Not a general background-job framework beyond GPU admission (no cron, no retries
  policy engine) unless a child explicitly scopes it.

### Definition of done (for THIS meta issue)

A parent epic issue exists with a `blocked-by` chain of small children covering: the
generalised typed job/queue model, the scheduler/admission with GPU serialisation,
`GET /queue` (+ per-job position), the live position socket, chat integration, and
resilience/cancel/restart behaviour — each child single-location with its own DoD,
mirroring ISSUE-0040's decomposition. No production queue code changes under this id.

## Comments
### Auditor · 2026-07-12
Filed as a meta/design issue at the user's request. Captures the real current
state — `JobRegistry::start` spawns a GPU worker thread immediately with nothing
serialising GPU access (`jobs.rs:102-129`) — and the target: one resilient queue
fronting the GPU for heterogeneous job kinds, a `/queue` API, and a live
position socket chat can wait on. The hard product decisions (priority vs. FIFO,
preemption, persistence, chat integration) are listed as open questions for the
epic to resolve rather than pre-decided here. P1: concurrent GPU access is a
latent crash/corruption risk, not just polish. Pairs with [[ISSUE-0050]] as the
two training-infra epics to design next.
</content>
