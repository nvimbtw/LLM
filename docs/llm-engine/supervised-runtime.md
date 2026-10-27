# Supervised worker implementation and consumer handoff

ISS-52 supplies usable parent-owned sessions and the runner/lifecycle foundation.
It does not by itself convert every existing caller to custom-model isolation.
The frozen messages and resource semantics are in
[architecture-runtime.md](architecture-runtime.md).

## Parent and worker API

`llm.runtime.supervisor.WorkerManager(root, gate)` owns the engine's runtime
lock and session registry. `start()` reconciles old worker ownership before
the application marks orphan jobs failed and admits queued work. `close()`
stops/drains its sessions before releasing the lock. The application uses
`storage.runtime_root()`; tests inject a disposable directory and `DeviceGate`.

```python
with WorkerManager(runtime_root, gate) as manager:
    with gate.admit(wait=True) as lease:
        with manager.session(lease, policy, "llm.architecture.worker:Backend") as session:
            snapshot = session.artifacts.stage_file(snapshot_file, "validation_snapshot_v1")
            result = session.run(Validate(type="validate", snapshot=snapshot),
                                 request_id=job_id, on_event=report,
                                 effects=actor_broker, adopt=publish)
            report_bytes = session.artifacts.read_bytes(result.result)
```

This is an adapter example; the backend entrypoint belongs to its implementing
slice and must exist before use. The parent passes a trusted engine module and
factory name, never a browser-selected import path. That factory is imported
and called **only in the child** with `WorkerContext` and returns a backend
implementing `handle(payload, context) -> Completed`, `finish_request()`,
and `close()`. One instance owns model/optimizer state across complete commands.
`finish_request()` clears request caches before success is emitted; `close()`
releases all resident state. Construction and both cleanup hooks are supervised.

`WorkerContext` exposes `device`, `progress(Progress, force=False)`,
`effect(EffectOperation)`, `publish(artifact, completed_step, final=False)` and
`artifacts`. It contains no actor, HTTP connection, storage session or device
gate. `SourceLoader` must stay alive alongside the built model throughout
capture/compilation/serving: pinned torch reimports revision module names.
Closing it immediately after construction is incorrect. The backend closes
the loader on unload, after model execution has stopped.

Training failures raised as `ArchitectureError` retain their stable domain code
and at most 512 characters of explanation, including unsupported captured
operation names. Plain `ValueError` contract refusals also retain a bounded
explanation for administrative training; arbitrary exception subclasses do not.
Ordinary run projections still replace private error text.
Generation and other runtime failures retain the exception-class-only response;
training diagnostics are not copied into public serving errors. No traceback or
custom exception object crosses IPC.

`manager.session(None, ..., metadata_only=True)` permits CPU inspection only;
it rejects numerical commands. `manager.resident_session()` owns a parent
lease for a cached serving lifetime. Queue admission calls `evict_for_queue()`
to evict idle caches or request eviction when the current operation finishes.
The worker never acquires an independent process-local device lock.

An owned parent monitoring thread drains idle resident cleanup certificates and
releases their leases even when no later request arrives. It polls healthy
workers without taking their request lock; only a ready guardian channel can
reserve that lock for retirement. This prevents an idle health check from
rejecting a valid load or generation as an overlapping request, while keeping
retirement separate from active execution. Startup reconciliation
failure before the application yields also drains the manager and releases its
cross-process lock; a partial startup cannot strand admission.

Socket option setup failures close both newly created endpoints. Session setup
owns its directory and channels until the worker binding/launch handoff; failure
before binding removes those resources and leaves admission reusable. Once a
launch handoff is uncertain, local handles close but ownership records remain,
quarantine is persisted and the manager refuses numerical and metadata sessions.
Restart cannot treat that uncertainty as verified cleanup.

`session.run()` is blocking and belongs on the existing runner thread or another
explicitly owned adapter thread, not the async executor. It accepts a
`threading.Event` for cancellation. The guardian watches cancellation,
deadlines and parent liveness independently of parent callback execution.
Disconnecting a durable job observer must not set this event. Interactive
disconnect must cancel/close its request and release cache state.

## Enforcement and restart

Each guardian retains the engine's advisory lock descriptor. The lock file is
never unlinked: a new engine cannot accidentally lock a different inode while
an old guardian is still draining. Parent/guardian and guardian/worker startup
barriers require fsynced PID, kernel birth tick and boot identity records before
releasing executable startup. Restart verifies those identities before any
termination; a recycled unrelated PID cannot authorize killing a process.

The guardian observes ordinary owned process groups, helpers, aggregate RSS,
session/cache disk usage and work progress at the configured monitoring interval.
It caps its forwarding queues and keeps control/liveness separate from progress.
Host tree monitoring has bounded **detection latency**, with possible overshoot;
it is not an instantaneous hard RSS limit. Per-process address/file/descriptor
limits and the worker allocator cap serve different purposes. Unsupported hard
cgroup policy fails explicitly when no delegated controller was configured.

Engine-controlled helpers must remain registered under the owning process
group/session and fit `owned_processes`. Compiler parallelism starts at one;
consumers must budget additional compiler/helper processes rather than treating
`RLIMIT_NPROC` as a per-worker tree limit. The supported trusted administrator
SDK forbids arbitrary process/session detachment, service access and hidden
external state. These mechanisms are supervision, **not a security sandbox**.

Unknown ownership, missing cleanup acknowledgement, unreaped descendants or
accounting failure persist quarantine and prevent new admission. Guardian death
also kills the direct worker through Linux parent-death signaling, but restart
still checks for surviving helpers before it may admit anything. Session success
and process-tree cleanup are separate: no durable terminal success may be written
until output adoption and required worker cleanup have completed.

## Obligations for later slices

| Owner | Required integration |
| --- | --- |
| ISS-54 capture/validation | Implement child-only inspect/validate backend, preserve the live `SourceLoader`, emit real bounded phases and return versioned result grants. Numerical/source probes never execute in the parent. |
| ISS-55 saved packages | Parent verifies byte size/hash/manifest and publishes immutable artifacts; actual custom source import, alias reconstruction and weight/optimizer load stay in the worker. Parent registration cannot unpickle custom model objects. |
| ISS-56 durable APIs | Register a supervised runner handler. Complete cleanup and artifact publication before marking the validation job done and retaining evidence in the same caller transaction. Registry observer disconnect is not cancellation. |
| ISS-57 memory/optimizer | Respect the worker's actual admitted device allocator ceiling when planning/guarding; a later training `memory.guard` call must not increase the configured cap. Include the frozen reference and trusted compiler helpers in resource estimates. |
| ISS-58 pretraining/recovery | Split parent corpus/tokenizer acquisition from child model/optimizer lifetime. Do not spawn legacy `orchestrate.train.run`, which constructs a parent-authenticated `Hop` client. Retain exact corpus/schedule/RNG identities and publish/adopt checkpoints before acknowledging success. |
| ISS-59 SFT | Existing `_finetune` loads model weights before corpus preparation. Reorder the adapter boundary: parent obtains/prepares authorized corpus, child imports/loads/trains the saved definition. Do not spawn legacy `sft.run` with `remote.Client(owner)`. Keep optional token-row initialization and private reserved-marker adoption in their proper model/tokenizer owners. |
| ISS-60 RL | Keep policy/reference/optimizer, generated tensors and updates resident. Catalogue/create/step/finish/abandon/status go through `EffectBroker` and the initiating actor's Rust hop. Never construct `gym.Client(owner)` in the child or replay issued writes after uncertain completion; discard interrupted iterations and create fresh recovery sandboxes. |
| ISS-61 serving/scoring | Route custom `chat.load` uses through a reusable session, including `history_floor`, attention and measurement paths, not just token streaming. Metadata/token counting stay parent-safe. Clear request caches on every outcome and evict the resident worker before priority queue work takes the device. |

`api/runner.py` ownership transfers **ISS-52 → ISS-56 → ISS-58**. Runtime protocol
changes remain coordinated across all consumers. Existing builtin handlers keep
their established path until their adapters intentionally opt into supervision;
that compatibility is not a claim that those paths have already been converted.

## Effects and adoption callbacks

`EffectBroker(actor, dispatch, EffectJournal(path))` keeps authorization and
service handles in the parent. `dispatch(actor, request, artifacts)` receives the
full typed `EffectRequest`, including its operation UUID, validates domain data
and uses the existing finite-timeout Rust hop. The RL adapter uses that same UUID
for durable job/resource ownership before returning a result grant. Issued
write UUIDs are fsynced before the remote call; partial/corrupt or duplicate
issuance is not retryable. Existing safe read retries remain adapter-owned.
Durable journal retention/recovery belongs to the queued operation owner.

Parent adoption callbacks verify regular files, exact byte length and digest,
publish/register atomically under the existing checkpoint contract, and return
only after durable completion. They must bound database/storage waits. A callback
exception or timeout stops the worker; no acknowledgement permits another
unregistered checkpoint. Late remote replies/adoption completion do not turn an
interrupted request into terminal success and cannot trigger write replay.

Actual GPU hardware verification remains explicitly user-deferred. CUDA support
and opt-in runnable checks remain present; CPU tests do not certify GPU cleanup.
