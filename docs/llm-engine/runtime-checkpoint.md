# ISS-52 paused implementation checkpoint

**Paused by the user on 2026-10-20. ISS-52 is unfinished.** No commit or push.
Resume from the existing shared checkout; preserve all other sprint agents' work.
Do not represent the runtime foundation as completed or all custom entrypoints
as isolated. Later integration remains separately owned.

## Implemented partial surface

- `llm/runtime/supervisor.py`: `WorkerManager`, `WorkerSession`, cleanup and
  cancellation exceptions, parent sessions, stable engine lock and restart
  reconciliation, bounded progress retention, asynchronous effect/adoption waits,
  resident serving cache ownership/eviction hooks.
- `llm/runtime/guardian.py`, `processes.py`, `entry.py`: independent guardian,
  parent/worker start barriers, fsynced PID/birth/boot identity records,
  parent-death worker signaling, process-group ownership/reaping, aggregate
  RSS/helper/output monitoring, cancellation channel and cleanup certificates.
- `llm/runtime/watchdog.py`, `bootstrap.py`, `transport.py`: phase/operation and
  training-progress watchdogs, sanitized launch data, bounded packet channels.
- `llm/runtime/worker.py`: child-only backend import/construction, resident
  model/optimizer backend protocol, request cache cleanup, typed effects and
  artifact adoption, host limits and CUDA allocator-margin setup.
- `llm/runtime/artifacts.py`, `effects.py`: immutable byte grants and size/hash/
  regular-file verification, actor-owned parent effect broker, fsynced
  issued-before-write operation journal, no write replay on uncertain completion.
- `llm/model/device.py`: explicit `DeviceGate`/`DeviceLease`, parent-only
  admission, one bound worker per lease and permanent uncertain-cleanup quarantine.
- `api/worker.py`: explicit `SupervisedRun`/`SupervisedHandler` adapter seam.
- `api/runner.py`, `main.py`: initial supervised dispatch, shielded active thread
  ownership and runtime shutdown/startup barrier. **These edits were added last
  and have not been verified.**

The ISS-49 protocol/fixtures remain frozen. Storage runtime path helpers were
added by the persistence owner in `llm/storage.py`; no runtime store/migration
changes were made by this lane.

## Stable consumer API already communicated

```python
with WorkerManager(root, gate) as manager:
    with gate.admit(wait=True) as lease:
        with manager.session(lease, policy, "trusted.engine.module:Factory") as session:
            grant = session.artifacts.write_bytes(snapshot_bytes, "validation_snapshot_v1")
            completed = session.run(Validate(type="validate", snapshot=grant),
                                    request_id=job_id, on_event=report,
                                    effects=actor_broker, adopt=publish,
                                    cancel=cancel_event)
```

Factories execute only in the child and return a backend implementing
`handle(payload, context) -> Completed`, `finish_request()`, `close()`.
`WorkerContext` exposes `device`, `progress`, `effect`, `publish`, `artifacts`.
It receives no actor, service client, credential or parent device gate.
Keep `SourceLoader` alive alongside the model through capture/compilation/serving;
close it only on backend unload. Source/optimizer/tensor state stays in the child.

`manager.session(None, ..., metadata_only=True)` permits CPU inspection only.
`manager.resident_session()` owns its parent serving-cache lease; queued work
uses `evict_for_queue()` before admission. Runner ownership transfers
**ISS-52 → ISS-56 → ISS-58** after this foundation is completed and verified.

## Exact evidence so far

1. Manual real production-path check passed: manager/guardian/fresh torch worker,
   pinned `Load`, three AdamW steps through `Train`, result-grant read and verified
   worker cleanup/lease release.
2. First production process suite:

   ```sh
   engine/.venv/bin/python -m pytest --noconftest llm-engine/tests/test_runtime_supervision.py -x
   ```

   **11 passed, 1 failed in 37.44s.** The failure was fixture construction of a
   `GymFinishEffect`: a base `RuntimeArtifact` instance was passed where the
   narrowed `EffectData` type is required, so the parent dispatch was never
   reached. The fixture now explicitly validates `EffectData` from the grant.
3. Affected and remaining focused checks after that correction:

   ```sh
   engine/.venv/bin/python -m pytest --noconftest llm-engine/tests/test_runtime_supervision.py -x -k 'parent_actor or parent_adoption or owned_helper or pid_reuse'
   ```

   **5 passed, 12 deselected in 16.25s.** This covers actor-bound uncertain writes,
   journal no-replay after reload, adoption exception/timeout, ordinary helper
   cleanup/count bounds and refusing PID-reuse termination.
4. No final full process-suite, legacy regression, formatter/typecheck or app
   lifecycle integration run was performed after the latest runner/main edits.
5. GPU check is present in `test_runtime_supervision.py`, opt-in with
   `LLM_TEST_CUDA=1`; actual hardware verification is explicitly user-deferred.
6. The prior completed ISS-49 runtime suite passed 129 checks/119 shared cases;
   it is contract evidence, not verification of this new implementation.

At pause, the running focused test command had completed successfully. A process
check found **no remaining runtime guardians, worker entrypoints, runtime tests
or sleeping helper subprocesses** owned by this task.

## Resume work and known review obligations

- Verify and finish the new runner/main adapter first. In particular, startup
  failures after acquiring the manager lock must close the manager; draining,
  pause/reset and repeated application-lifespan behavior need explicit checks.
- Complete the process-failure matrix: parent SIGKILL/restart lock handoff,
  guardian SIGKILL with surviving helper/quarantine, all crash windows around
  fsynced ownership/start release, flood/backpressure, aggregate RSS/output
  enforcement, metadata-only restrictions and concurrent shutdown.
- Check launch-failure cleanup and certificate/read/close races. Unknown process
  ownership must fail closed, and PID reuse must never authorize killing an
  unrelated process. The lock file must never be unlinked/recreated.
- Finish real supervised runner tests: no premature durable terminal success,
  cancellation status, checkpoint-adoption failure, cleanup uncertainty and
  preserving existing builtin handler behavior. Existing builtin math paths
  remain in-process until their explicit later adapters are converted.
- Parent effect/adoption callbacks must use bounded trusted HTTP/database/storage
  waits. A worker watchdog can stop model execution independently; a late parent
  callback cannot turn an interrupted job into success or trigger write replay.
- Review immutable artifact verification/load races and cumulative resource
  accounting. The current runtime rejects unconfigured hard-cgroup policy;
  `preflight_and_monitor` is bounded-latency supervision with possible overshoot,
  not a hard RSS/process security sandbox.
- Preserve the configured CUDA allocator ceiling in later training planners:
  existing `memory.guard` calls must not increase it. Helpers need explicit
  ownership/count budget; `RLIMIT_NPROC` is not a per-worker tree cap.
- Run appropriate compile/lint/documentation/diff checks and focused legacy
  runner/chat/checkpoint regressions after the implementation is stable.
- Obtain independent review before closing ISS-52; notify coordination when
  runner ownership can transfer.

Detailed later-slice obligations for ISS-54–61 are recorded in
[supervised-runtime.md](supervised-runtime.md): parent corpus/Hop splitting,
SFT's early checkpoint load, RL catalogue/create/step/finish/abandon/status actor
effects, exact recovery, and serving `history_floor`/attention/measurement paths.
