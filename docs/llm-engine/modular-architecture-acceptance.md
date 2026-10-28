# Modular architecture acceptance

FEAT-4 remains **in progress**. The authoritative scope is the reviewed
[Discord feature](https://discord.com/channels/1538659898383011961/1562025656110284921)
and ISS-48–ISS-68. Eight issues are accepted, twelve are in progress and ISS-68
is open. Implemented paths and isolated numerical/adapter evidence do not prove
the remaining live contracts. This matrix is a verification map, not acceptance.

## Automated entry point

From the repository root:

```sh
bash scripts/verify_modular_architecture.sh
```

The script runs schema/type freshness, the complete engine suite, Rust workspace
build/Clippy/tests plus the explicit Rust/Python privacy seam, frontend
type/lint/tests/build and the existing browser fixture scripts. It records one
log per command and exact statuses in a fresh `/tmp/feat4-acceptance.*` directory.
Failures do not hide later checks. `FEAT4_REPORT_DIR` selects an unused report
directory; existing status files refuse overwrite. Module groups can be supplied
explicitly, for example `contracts frontend`. Running a subset is not full
acceptance. Review test skips and JUnit results even when a command exits zero.

Prerequisites:

- Install dependencies using the [native setup](../../README.md). Set
  `SLATE_ENGINE_PYTHON` to an absolute engine interpreter path when necessary;
  otherwise the script prefers `llm-engine/.venv`, then `engine/.venv`.
- Permit local TCP/Unix sockets and supervised subprocesses. The real tests
  must retain worker, cancellation, restart and disconnect checks.
- Set `LLM_ENGINE_TEST_ADMIN_URL` to a disposable PostgreSQL administration
  service with database creation permission. Engine fixtures create unique
  ownership-marked databases from template0 and remove only their own databases.
  Configure backend `DATABASE_URL` for disposable PostgreSQL tests as documented
  in the [backend map](../backend/README.md). Do not use production databases.
- Start the frontend for fixture browsers, normally on port 3011. Set
  `AUTHORING_URL` for another origin and `CHROMIUM_PATH` if Playwright has no
  installed browser. These scripts exercise explicitly synthetic fixture pages.
- Production builds need access to the existing Google font downloads. Hardware
  CUDA/fp16/bf16/allocator checks remain user-deferred to the GPU machine; CPU
  passes or skipped CUDA tests cannot certify them.

This entry point does not start Docker, migrate application databases, install
dependencies, commit, publish or mark issues complete.

## Requirements and evidence still required

Every row maps a formal acceptance criterion to the relevant current tests and
the additional proof needed. Existing detailed reports are in the
[evidence ledger](modular-architecture-evidence.md). Test names identify coverage,
not a blanket claim that the complete test file currently passes.

| Requirement | Automated evidence surface | Remaining acceptance proof |
| --- | --- | --- |
| Decorated custom block, declared controls, captured internal diagram | `test_architecture_core.py`, `test_architecture_validation.py`; shared frontend decoders | Actual administrator create→inspect→configure→validate→accept and browser diagram |
| YAML branches/merges, multiple ports, subarchitectures and nested bounded loops | `test_architecture_core.py`, `test_architecture_validation.py`, `test_builtin_recipes.py` | Same constructs through Rust/UI and real supervised execution |
| Parameter changes alter construction and stale old validation | `test_authoring_metadata.py`; frontend validation form/submission/draft tests | Edit source/parameters during delayed validation and verify acceptance stays disabled |
| Traceable model trains/serves; untraceable draft remains saveable but cannot train | `test_training_submission.py`, `test_validation_queue.py`, `test_supervised_pretraining.py`, `test_serving_parent.py` | Real process admission, publication, ordinary chat and rejected capture |
| Standard recipe preserves pinned trace and legacy artifacts | `test_dataflow_trace.py`, `test_builtin_recipes.py`, `test_train_state.py` | Full current legacy suite and actual published legacy chat |
| Looped recipe shares weights/gradients and retains serving iterations | `test_modular_training_loop.py`, `test_modular_worker.py`, `test_inference_worker.py` | Train→saved checkpoint→restart/recovery→serve without structural override |
| Unique counts, shared serialization and optimizer ownership | `test_architecture_packages.py`, `test_modular_training_support.py`, `test_training_publications_postgres.py` | PostgreSQL publication/rollback plus actual process recovery |
| Enabled caches equal full-prefix and isolate occurrence/request state | `test_architecture_core.py`, runtime interface tests | Declared production caches remain disabled; any enabled adapter needs exact signature/capability capture and numerical proof. Legacy cache regression remains required |
| Edits/archive after enqueue cannot change execution after restart | `test_training_submission.py`, publication/recovery tests | Real restart with renamed/archived catalog and pinned custom revisions |
| Resume/SFT/RL/checkpoint/scoring/chat consume saved definitions | `test_supervised_pretraining.py`, `test_supervised_sft.py`, `test_supervised_rl.py`, `test_serving_parent.py` | Real queue/worker chains and live Rust/data/gym/bench seams |
| Invalid syntax/ports/code/budgets/cancellation/crashes remain bounded | architecture/runtime/admission/handler tests | Real import/load/forward/backward/decode/measurement termination, no orphan or held admission |
| Administrative privacy and ordinary published chat | `architecture_proxy.rs`, `test_architecture_visibility.py`, `test_saved_model_metadata.py` | Genuine Rust/Python HTTP/WebSocket plus browser admin/non-admin transitions |
| Shared decoders and stale editor/results races | shared contracts; frontend source/draft/validation/submission/read-owner tests | Browser focus, edits, delayed responses, actual session replacement |
| Causality, padding/prefix, finite gradients, RNG/replay/dropout RL | capture/numerical/modular training/SFT/RL tests | Full current CPU suite; user-deferred GPU arithmetic and allocator evidence |
| Hangs/crashes cannot orphan workers/leases/partial models | runtime supervisor, worker, publication and recovery tests | Real forced termination at every custom execution boundary and subsequent device use |
| Retention/concurrent acceptance/same Python names/pending tokenizer/SFT without token rows | store/package/submission/SFT tests | Real PostgreSQL concurrency and supervised pending-tokenizer/SFT paths |
| Required migrations, module checks, Rust/frontend/browser gates pass | Automated entry point above | Every required gate passes; blocked or skipped required checks keep FEAT-4 incomplete |

## Connected scenarios

Use the `agents` development account. Keep scratch resources/datasets/models
separate from human application data. Record actual run/revision/definition/
receipt/graph identities, numerical settings and screenshots for each scenario.

1. Author a tiny custom block and recipe through the UI, inspect before required
   parameters are known, exercise invalid Python/YAML, branches/merges/multiple
   ports, independent stack versus shared repeat and nested subarchitectures.
   Copy a builtin and explicitly pin dependency revisions. Check optimistic-save
   conflicts, typing during save, archive rules and stale validation.
2. Submit standard/shared/custom runs and a pretrain→SFT→RL chain. Include
   pending tokenization, a custom model without token-row initialization and
   dropout RL. Edit/archive catalog resources after enqueue. Check every job's
   frozen pins, final preflight, optimizer ownership and checkpoint counts.
3. Cancel and restart actual execution; recover a complete checkpoint and compare
   weights/optimizer/RNG with the uninterrupted trajectory. Prune predecessor and
   validation history, then inspect saved graphs and source. An imported model
   with no catalog rows must show retained custom source and exact hashes.
4. Publish and chat as an ordinary user, measure through bench-engine, exercise
   gym rollout effects through Rust and request supported occurrence attention.
   Verify normal users cannot fetch administrative source/graphs or private
   validation events. Disconnect generation/measurement and confirm drain before
   the next device owner enters.
5. Force custom import/construction/load/forward/backward/decode/measurement
   failures and hangs. Verify cleanup, bounded actionable errors, orphan-package
   adoption/rollback, no replayed uncertain gym writes and successful subsequent
   execution. Preserve legacy APIs/presets/artifacts throughout.

Browser fixture scripts alone do not exercise these connected scenarios. No
fixture graph or successful build substitutes for their actual service evidence.
