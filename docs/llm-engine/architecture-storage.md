# Modular architecture persistence

The engine owns catalog drafts, immutable revisions, validation evidence,
resolved definitions and enqueue source snapshots. Rust owns users and forwards
actor UUIDs; these are opaque attribution, never cross-database foreign keys.
Production numerical validation and worker/publication wiring remain separate
from this persistence layer. See [contracts](architecture-contracts.md) and
[durable execution](execution.md).

## Schema and migration

Forward-only migration `b8e246d913af` follows `a4e79c1026bf`. It preserves all
legacy model/job/training IDs, adds `model_validation` to `run_kind` and leaves
old `models.definition_id`/`training_runs.definition_id` null. Existing v1 model
loading remains unchanged.

| Table | Owned fact |
| --- | --- |
| `model_blocks`, `architectures` | UUID, display metadata, origin/archive, exact editable bytes/hash, independent draft/metadata versions and actor/timestamps. |
| `block_revisions` | Immutable source/declaration/compatibility and concrete acceptance receipt; unique resource/number and resource/content. |
| `architecture_revisions` | Immutable YAML/canonical recipe, dependency digest and acceptance receipt. |
| `architecture_dependencies` | Exactly one pinned block or child-architecture revision per declared dependency path; FK/type constraint rejects both/neither. |
| `validation_evidence` | Immutable typed receipt/hash and logical source-run UUID, independently retained after history pruning. |
| `model_validations` | One-to-one execution domain config/report/receipt link, never another status/queue authority. |
| `model_definitions` | SHA-256 semantic identity and immutable resolved manifest. |
| `definition_evidence` | Profile receipts retained by saved definitions/packages without including profiles in semantic identity. |
| `run_architecture_pins` | Accepted root, retained evidence, submitted parameters and complete immutable source snapshot for queued/waiting training. |
| `architecture_mutation_requests` | Unique actor/request identity, argument digest and logical operation outcome; no prunable job FK. |

PostgreSQL triggers prohibit updates to revisions, dependency pins, receipts,
definitions, retained package-evidence links and enqueue snapshots. Required
source/evidence references use restrictive FKs. Deleting execution history
cascades only its domain/pin rows; accepted revisions and definition-evidence
links retain their receipts and source closure. No hard-delete catalog API exists.

## Transaction APIs

`store.architecture_queries` accepts explicit Sessions. The caller owns one
transaction and commits after the complete operation/chain succeeds. Never catch
an operation error and commit its earlier queue rows.

- `create_block`/`create_architecture` and `save_*_draft` store exact bounded bytes
  without parsing/importing. Root row locks plus expected draft versions prevent
  concurrent overwrite. Invalid Python/YAML remains editable.
- `update_metadata` locks the root and checks its independent metadata version.
  Supplied builtins are read-only; the internal seeding choice is not a public
  request field.
- `record_evidence` first requires an existing `model_validation` job in `done`,
  then inserts its typed immutable receipt and domain link. The validation adapter
  finishes the job and records evidence in one transaction after worker cleanup
  and artifact publication. Job deletion never owns receipt deletion.
- `accept_block`/`accept_architecture` lock the root, compare current version/hash,
  require exact successful retained draft evidence and freeze source/dependencies.
  Identical validated source acceptance returns the existing revision. The caller
  supplies the validated declaration/recipe from its captured snapshot.
- `require_revision`, `recipe_dependencies`, `dependency_digest` and
  `dependency_closure` read immutable pins with bounded nesting/source cardinality.
  Reused subarchitecture source is visited once; recursive/corrupt pins fail.
- `enqueue_pinned_training`/`pin_training_run` check root archive, acceptance,
  dependency/parameter/evidence identity and persist the source closure in the
  same queue transaction. A pinned child may be archived; new root selection may
  not. Structural/source errors roll back queue and snapshot together.
- `record_definition` verifies the root/recipe/source closure and final tokenizer,
  parameters/context evidence, atomically deduplicates semantic identity and
  retains package receipts. Training/model rows reference it after finalization.
- `idempotent_mutation` uses a unique actor/request insert and row lock. Identical
  retries return one outcome, changed arguments fail, and another actor can use
  the same request UUID. The logical outcome survives job pruning.
- `prune_unreferenced_evidence` visits at most100 unused receipts, skips locked
  evidence and relies on all restrictive reference FKs. Acceptance reads hold key
  share until commit, so pruning cannot race away evidence being accepted.

`CatalogError.code` distinguishes missing/read-only/stale/archive/invalid pin
refusals. API adapters choose HTTP statuses and bounded diagnostics; storage owns
no HTTP, source execution, service clients or filesystem publication.

`Run` delegates validation config/result to `ModelValidation`; its typed output
is always `{type: validation, id: same job UUID}`. Lifecycle remains exclusively
in `execution_jobs`. Generic chain submission rejects validation before creating
any rows because a completed check may be invalid and never produces a model.

## Artifact paths and handoff

`llm/storage.py` is the only path authority. These helpers validate identities
and return paths without creating directories:

| Helper | Path beneath DATA_DIR |
| --- | --- |
| `validation_dir(run_uuid)` | `artifacts/validations/<uuid>/` |
| `definition_dir(sha256)` | `artifacts/definitions/<digest>/` |
| `runtime_root()` | `artifacts/.runtime/` |
| `runtime_session_dir(worker_uuid)` | `artifacts/.runtime/<uuid>/` |
| `runtime_lock_path()` | `artifacts/.runtime/engine.lock` |

ISS-55 owns staged/fsynced filesystem publication/reconciliation and checkpoint
integration after this path handoff. ISS-56 consumes catalog/evidence/validation
operations. ISS-58 consumes atomic pinning/definition references and must validate
all chain steps before its single transaction. No production optimizer step or
artifact publication is introduced by persistence tests. All later engine schema
changes return to the migration owner; do not create competing Alembic heads.

## Verification

From `llm-engine/`:

```sh
PYTHONPYCACHEPREFIX=/tmp/slate-check ../engine/.venv/bin/python -m pytest tests/test_architecture_store.py tests/test_migrations.py tests/test_store.py tests/test_run_routes.py tests/test_chain.py
```

The PostgreSQL suites use `LLM_ENGINE_TEST_ADMIN_URL` (default local peer socket
`postgresql+psycopg:///postgres`) and create UUID-named databases from template0.
Each has an ownership marker checked before cleanup. Existing application DBs
are untouched; a missing PostgreSQL server fails the required check instead of
silently skipping it. Template0 also avoids requiring modification of a system
whose template1 has a pre-existing collation-version mismatch.

Coverage includes populated legacy migration, ORM/schema parity, actual locking
races for drafts/acceptance/archive/enqueue, actor/request dedupe and conflicts,
atomic late-chain rollback, nested source snapshots, SQL immutable/FK failures,
waiting/package/history-pruning retention and legacy store/route/chain behavior.
Receipt payloads in storage tests are fixture evidence, not new claims of graph
capture or numerical execution. Live HTTP/worker/training/browser integration
belongs to the later gates. GPU hardware checks remain deferred by the user.
