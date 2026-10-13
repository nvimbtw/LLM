# Conversation and benchmark ownership cutover

Stop backend and Python workers before copying. Back up all databases and the
model artifact directory. Keep the old binaries available until deployment has
been accepted. The scripts are offline administrative transfers, not service
calls; running them while writers are active is unsupported.

For the repository's `slate` Compose installation,
`bash scripts/prepare_sprint_cutover.sh` stops running application writers, dumps
all PostgreSQL databases/roles and archives current and legacy artifact trees
under `~/llm-sprint-backups/`. It records schema inventory and checksums, then
leaves writers stopped. It uses sudo for Docker access and artifact reads;
run it as the ordinary repository user. It does not run migrations or tests.
Inspect its printed inventory before proceeding, especially if an older bench
database already exists. Dumps contain private data and role password hashes;
share only the inventory when diagnosing the cutover.

For that Compose installation, `bash scripts/apply_sprint_cutover.sh` uses the
latest backup directory (or an explicit first argument), checks its checksums,
and refuses to proceed while application writers are running. It uses the
existing LLM image's Python dependencies to upgrade bench-engine, run the exact
copies, upgrade the LLM schema, and copy legacy benchmark job history. It leaves
writers stopped and preserves source archives. The backend conversation migration
must already be applied. Unknown benchmark revisions and unfinished legacy jobs
stop the operation; it never stamps an incompatible schema as current.


1. Create `slate_bench` for existing installations. Apply backend migrations and
   bench-engine Alembic migrations. Do not start request-serving processes yet.
2. Set `DATABASE_URL`, `LLM_ENGINE_DATABASE_URL`, `BENCH_ENGINE_DATABASE_URL` to
   their respective databases. From the repository root, using the LLM virtualenv:
   `llm-engine/.venv/bin/python scripts/migrate_conversations.py --apply`, then
   `llm-engine/.venv/bin/python scripts/migrate_benchmarks.py --apply`.
3. Each script locks source writers, inserts missing target UUIDs, compares every
   copied column, and refuses a differing existing target row. Attachment bytes
   and benchmark JSON are compared exactly. Unknown conversation owners fail the
   local user FK rather than silently dropping private data. Resolve the owner
   mapping before retrying. No existing target rows are overwritten.
4. Target commit precedes source archival. A crash between them is retryable:
   the next copy compares existing target rows. Successful scripts rename source
   tables to `migrated_*`; they do not erase them. Do not rerun a completed script
   after source archival. The benchmark script also copies reference `.txt` files;
   use `LEGACY_BENCH_CORPUS_DIR` and `BENCH_CORPUS_DIR` for custom locations.
5. Apply LLM Alembic migrations, then start the new services. Retirement migrations
   refuse nonempty untransferred source tables. Fresh empty databases need no copy.
   `runs` becomes `execution_jobs` with unchanged UUIDs, timestamps, queue order,
   dependencies and statuses. Domain configuration/results move into training and
   tokenizer records. Existing benchmark dispatch references remain unchanged.
6. If upgrading an earlier bench-engine database, run
   `scripts/migrate_legacy_bench_jobs.py --apply` with the same service database
   variables. It compares all scheduling fields and dispatch configuration before
   committing the preserved IDs to the LLM queue; the original result JSON remains
   archived and reports stay accessible through bench-engine.

Message positions use the prior creation-time ordering, with UUID as a stable
secondary order for timestamp ties (the old query specified no tie-breaker).
Model name snapshots are taken at migration; deleted legacy models already
nullified by the old FK cannot have their lost names reconstructed.

For rollback before accepting new writes, stop services, restore all database
backups and old binaries together. After accepting new writes, restoring only
source tables would lose new conversations/reports: export and reconcile those
first. Archives can be removed only after deployment validation and backup
retention requirements are satisfied. This sprint does not automatically erase
them or execute the transfer against live data.

Validation is a separate phase: user isolation, attachment bytes/limits, feedback,
retry/disconnect behavior, migration continuity, queue dependencies and device
admission. Measure cold/prefetched history and file latency on the same dataset
before/after; these storage changes make no measured generation-speed claim.
