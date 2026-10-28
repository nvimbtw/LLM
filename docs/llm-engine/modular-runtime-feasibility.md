# Supervised runtime feasibility — ISS-48

This is a **2026-10-20 prototype record**, not a production execution contract.
The worker experiment is deliberately unwired from the application. The current
durable runner and device gate still follow [execution.md](execution.md).
The complementary tensor/capture evidence lives in
[architecture-capture.md](architecture-capture.md).

## Result and reproduction

The CPU process-lifecycle experiment passes on Linux 7.2.9-arch1-1,
Python 3.14.8, PyTorch **2.13.0+cu130**, with one torch CPU thread.
`torch.cuda.is_available()` is false; neither `/dev/nvidia*` nor `nvidia-smi`
is available. No CUDA allocation, device-memory release or GPU scheduler
restart claim follows from these results.

From the repository root, using the already installed development environment:

```sh
engine/.venv/bin/python -m unittest discover -s llm-engine/tests/feasibility -v
engine/.venv/bin/python -m compileall -q llm-engine/tests/feasibility
```

The experiment and tests are `llm-engine/tests/feasibility/runtime_worker.py`
and `llm-engine/tests/feasibility/test_runtime_worker.py`. They use unittest
discovery independently of application pytest fixtures, avoiding storage,
service startup and fixture model writes. They launch actual fresh interpreters
and torch workers, rather than mocking cancellation or subprocess state.
The final confirmation completed **13 tests in 44.102 seconds, all passing**;
compileall and whitespace checks also passed. No configured Python formatter,
linter or typechecker executable was available in the installed environment.

## Ownership and supported code

The engine stand-in owns the single admission lease, the initiating actor,
service adapter, deadlines and report retention. The lease is never passed to
the worker. The worker keeps the model, AdamW state and request-local decode
cache; two training requests followed by several generations reuse its PID.
Optimizer step counters reach three across requests. Checkpoints are bounded
files, not tensor payloads transferred over IPC.

A separate trusted guardian owns the worker process group, acts as a Linux
child subreaper and watches the engine's sole liveness socket. The worker gets
only its own IPC socket and an explicit disposable artifact directory. Fresh
interpreters use `-I`, an allowlisted environment, closed unrelated descriptors
and `/dev/null` for standard streams. The inheritance test injects dummy browser
and service credentials into the engine environment and proves they are absent
from the child's inherited environment. Torch may subsequently add its own
cache environment entries.

This supports trusted administrator model code under the proposed SDK contract:
immutable configuration/parameters/buffers, explicit tensor state and no custom
external I/O, arbitrary subprocess/session spawning or hidden persistent state.
It is **not a hostile-code sandbox**. Process groups cover workers and ordinary
helpers actually created by this experiment. A custom descendant deliberately
calling `setsid`, credential changes, native code changing limits, or the
simultaneous death of engine and guardian are outside the proved contract.

## Failure evidence

| Actual experiment | Observed contract |
| --- | --- |
| Source import and constructor never return | Parent deadline expires; guardian kills and reaps the worker after cancellation grace. |
| Successful finite eager fixture validation, then hung torch forward | Admission stays held until cleanup acknowledgement. |
| Successful fixture validation, then hang inside actual autograd backward hook | No incomplete optimizer step/checkpoint is reported; cleanup releases admission. |
| Two completed training steps, then decode hangs after three tokens | Last complete checkpoint bytes survive unchanged; request-local unfinished decode is discarded. |
| Healthy resident worker idles longer than one second | Subsequent generation reuses the same PID. |
| Worker creates an ordinary helper process | Cleanup kills and fully reaps both processes. |
| Actual engine stand-in receives SIGKILL | Guardian observes socket EOF, kills/reaps worker and helper, exits, and a replacement engine can start work after verified disappearance. |
| Guardian receives SIGKILL | Missing cleanup acknowledgement quarantines the lease; new admission fails. The test performs explicit emergency worker cleanup, so this is not a no-orphan proof for guardian failure. |
| Worker requests a 32 GiB host allocation | Per-process address limit produces `MemoryError`. |
| Worker writes a file beyond 1 MiB | File limit produces `OSError`; output remains at most 1 MiB. |
| Worker floods progress | Parent refuses more than 64 reports for one command and retains at most 64 reports across its lifetime. |
| Excessive, malformed, duplicate-field or truncated datagram | Typed boundary rejects the record. |
| Parent-issued gym write commits but acknowledgement is lost | Issue ledger retains the operation identity and refuses retry. |

The fixture validation above is a finite eager forward, not the production
capture validator. Actual custom-source capture acceptance and production
runner integration are separate gates.

## Measured budgets and proposed production policy

The values below are explicit experiment settings, not promises that a full
training job fits these demonstration limits.

| Resource | Experiment bound |
| --- | --- |
| IPC | Unix `SOCK_SEQPACKET`, 4096 bytes per datagram, four typed fields, strings at most 64/1024 characters; kernel truncation detected. |
| Reports/effects | 64 reports per command, rolling retention of 64 reports, at most 64 issued effect identities. Socket send waits are bounded. |
| Work | At most eight optimizer steps or eight decode tokens per command; no per-tensor IPC. |
| Startup | Absolute 30-second import/construction deadline; progress does not renew it. |
| Command deadlines | Tests use 10 seconds for tiny legitimate work and 0.1–0.2 seconds for deliberate hangs. Reports never reset an absolute command deadline. |
| Cancellation | Cooperative command, 150 ms grace, SIGKILL to the owned group, at most two seconds to reap and verify. Missing proof keeps admission closed. |
| Host address space | `RLIMIT_AS` 16 GiB per worker, including mapped torch/CUDA libraries; not aggregate tree RSS. |
| CPU | `RLIMIT_CPU` 30 CPU seconds per demonstration worker; enforcement configured, CPU exhaustion not exercised. |
| Files/handles | `RLIMIT_FSIZE` 1 MiB per file, `RLIMIT_NOFILE` 64, core dumps disabled; checkpoint file bytes separately checked. |
| Source | A 1 MiB check exists around the fixed import-hang fixture. General source ingestion is not implemented here. |

One independent three-step measurement recorded 2.2557 seconds for startup,
0.0478 seconds for normal cooperative cleanup, a 3725-byte checkpoint,
3,432,748 KiB peak virtual size and 681,840 KiB peak resident size. These are
single-run observations of a tiny `Linear(4,4)` model and AdamW, not workload
estimates. The test checks hung-phase cleanup finishes within three seconds.

Production must derive admission estimates from the bound definition and input
profile before allocation, retain the existing allocator/device margin policy
from [memory.md](memory.md), and supervise the complete training lifecycle in
the owning worker. Import/validation deadlines must be distinct from training
progress/step watchdogs and generation limits; a legitimate hours-long run must
not inherit this demonstration's short whole-command or CPU deadline. Serving
needs an explicit cache eviction/idle policy, not implicit socket read expiry.

## Integration still required

The parent-side effect broker is an in-memory adapter seam. Its tests preserve
the actor and prove no replay on uncertain completion; they do **not** contact
Rust, data-engine or gyms-engine. Production must bind that seam to the existing
Rust hop, journal/reconcile effect issuance with durable job identity, and use
fresh gym sandboxes after an interrupted iteration. A worker must never receive
browser tokens, service clients or direct service-storage access.

The checkpoint test verifies atomic replacement and retention of a complete
file; durable model registration, recovery reconciliation and restart identity
remain governed by [checkpoints.md](checkpoints.md) and require their integration
tests. Admission in this prototype is an engine-owner experiment; production
must use the actual `llm.model.device` owner without importing/reacquiring its
process-local lock in workers.

The host exposes cgroup v2 without a writable delegated subtree. The experiment
therefore does not claim aggregate descendant memory/process-count limits.
GPU allocation caps, CPU/GPU resource estimates for real architectures,
CUDA-context teardown after cancellation/parent death, production restart
exclusion before old workers are verified dead, and live Rust effect/recovery
integration remain unverified here. Later worker/integration slices must supply
that evidence before claiming those production guarantees.
