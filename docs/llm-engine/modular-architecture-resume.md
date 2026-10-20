# Modular architecture resume handoff

Checkpoint: 2026-10-20, branch `modular-arch`, HEAD `49c14b1`. All feature work
is uncommitted in the shared worktree. No commit or push is authorized. Preserve
tracked modifications and untracked files; `git diff --stat` excludes the many
new contract/source/test/doc files. Read `git status --short` before resuming.
The user confirmed pause. Root marked the goal PAUSED and posted the Discord
pause;50/51/52 remain incomplete pending final review. No further implementation,
review or tests may run until the user resumes.

## Authoritative requirements and state

Discord FEAT-4 revision4, its amendments and ISS-48 through ISS-68 remain the
requirements authority. Local navigation copies are `/tmp/modular-arch-created.json`,
`/tmp/feat4-full-4.md` through `9.md`, `/tmp/feat4-current-amendments.md` and
`/tmp/feat4-current-index.md`. Root manages Discord states. The detailed local
acceptance ledger is `/tmp/modular-arch-evidence-ledger.json`; the persistent
[implementation evidence](modular-architecture-evidence.md) records current gates.
Read full AGENTS.md, conventions and affected READMEs again before resumed edits.

The user explicitly deferred actual GPU verification to their other machine.
GPU support remains required; keep runnable hardware checks and record them as
user-deferred rather than passing on CPU. No GPU hardware is available here.

## Current owners and handoffs

| Agent | Files and current slice |
| --- | --- |
| Root | Orchestration, Discord states, final gate acceptance. |
| `sol_coordination` | ISS-49 authoritative `spec.py`/`wire.py`/schema harness and API fixtures; ISS-51 `store/architecture_schema.py`, `architecture_queries.py`, sole Alembic migration/schema ownership, `store/schema.py`, `llm/storage.py`, persistence/route tests and documentation maps. |
| `sol_capture` | ISS-50 SDK/source/parser/resolver/build/execute/cache/plan files and core tests; ISS-49 interfaces and SDK/YAML fixtures/docs. Authorized minimal `contracts.decode_fixture_yaml` delegation to production `parse.decode_yaml`; confirm its final result before resuming shared harness work. |
| `sol_runtime` | ISS-52 runtime supervisor/guardian/worker/artifact/effects modules, `llm/model/device.py`, initial `api/runner.py` and `main.py` integration and runtime tests/docs; sole current runner owner until handoff56 then58. |

No competing Alembic heads, schema copies, store edits or runner edits. ISS-55
will consume storage path helpers; ISS-56 and58 consume explicit Session-owned
catalog transactions. No further subagents or new slices were started at this
checkpoint. Root must accept current gates and assign the next work explicitly.

## ISS-51 completed implementation and checks

Forward-only migration `b8e246d913af` follows `a4e79c1026bf`, adds11 catalog,
revision/dependency, retained receipt, definition, validation-domain, immutable
run-pin and actor/request tables, and nullable model/training definition refs.
It adds `model_validation` RunKind/output without a second lifecycle authority.
PostgreSQL triggers reject immutable-row updates; restrictive FKs preserve pinned
source/evidence. Legacy IDs and null definition semantics are retained.

[Persistence documentation](architecture-storage.md) gives exact transaction APIs
and helper paths. The caller owns commit/rollback of the complete chain. Drafts
save exact invalid Python/YAML without imports; acceptance locks current draft
and evidence; enqueue snapshots immutable transitive sources; definitions retain
profile receipts outside semantic identity; actor/request retries serialize.
New `record_evidence` requires the validation job already `done`;56 must finish
job and retain evidence in one transaction after verified worker cleanup and
artifact publication. Validation cannot be a model-producing chain step.

Executed from `llm-engine/` with installed `engine/.venv` Python3.14.8,
torch2.13.0+cu130 CPU and local PostgreSQL18.6:

- `PYTHONPYCACHEPREFIX=/tmp/iss51-pycache ../engine/.venv/bin/python -m pytest tests/test_architecture_store.py tests/test_migrations.py tests/test_store.py tests/test_run_routes.py tests/test_chain.py --log-level=WARNING --junitxml=/tmp/iss51-persistence.xml`:62 passed9.78s.
- After final completion/chain guard changes, `PYTHONPYCACHEPREFIX=/tmp/iss51-pycache ../engine/.venv/bin/python -m pytest tests/test_architecture_store.py tests/test_run_routes.py --log-level=WARNING --junitxml=/tmp/iss51-final-store-routes.xml`:40 passed8.32s.
- Coordinated49 fractional queue-rank/blank-label correction:355 affected contract tests passed3.28s;67schemas regenerated and checked.
- Targeted compileall, function-doc AST audit, local documentation links and `git diff --check` passed.

All migration/concurrency databases are unique `slate_arch_<pid>_<uuid>` names,
created from template0 with verified ownership markers before forced cleanup.
Existing app DBs were untouched. Template0 avoids a pre-existing template1
collation mismatch. No SQLx metadata regeneration applies to51: no Rust SQL/schema
changed. Storage fixtures supply typed receipts, not numerical execution proof.
HTTP/worker/publication integration remains52/55/56/58. Root/peer review was
pending at the budget checkpoint; do not claim those downstream slices complete.

## ISS-50 brief independent review

`PYTHONPYCACHEPREFIX=/tmp/iss51-pycache ../engine/.venv/bin/python -m pytest tests/test_architecture_core.py --log-level=WARNING --junitxml=/tmp/iss50-core-independent.xml`:
40 passed9.67s. Read-only review covered parser, source eligibility/loader,
resolver, construction/alias placement, eager/strict invocation and occurrence
cache routing. No confirmed blocker was found in those boundaries.

One narrow policy caveat remains for54/runtime integration review: a custom
forward containing `torch.get_default_dtype() == torch.float32` passes source
eligibility and strict export. The worker/profile must pin this ambient setting
or the supported-source policy must reject the read. This is a recorded concern,
not a proven eager/export mismatch. `torch.initial_seed()` and
`torch.get_num_threads()` were separately observed to fail strict export. Do not
expand the loader into a hostile-Python sandbox; the supported policy is trusted
admin source, with bounded process ownership and explicit capture checks.

## Next resume steps

1. Read peer final reviews and root gate decisions for50/51; inspect their last
   files and targeted reports. Resolve only concrete recorded defects.
2. Confirm52 final runner/main lifecycle and real-process evidence. Keep its
   runtime/HTTP wiring obligations explicit.
3. Root assigns53 to capture after50 and Rust62 to coordination after51;62 may
   start against49 fixtures but cannot close before live56/58/61 contracts.
4. Continue54/55/57 and then56/58 in dependency order. Keep one migration owner
   and serial runner ownership; preserve actor privacy and atomic chain pins.
5. Maintain per-issue evidence and actual user-deferred GPU commands through68.
