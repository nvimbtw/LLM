# Sprint Cleanup validation

Implementation covers ISS-20–24. Test execution remains deferred by the user;
this document records the acceptance work still required before closing ISS-19.
No live ownership transfer has been performed. Changes are uncommitted.

## Checks completed

- Rust workspace build: passed with zero warnings, using SQLx offline metadata.
- SQLx workspace metadata: regenerated against an isolated PostgreSQL database
  with backend migrations applied, without application data.
- Frontend TypeScript and ESLint: passed.
- Changed Python sources: syntax parsed. Documentation file links and diff
  whitespace checked.

These checks do not establish runtime or migration correctness.

## Deferred acceptance

| Issue | Required evidence |
| --- | --- |
| ISS-20 | Queue an English benchmark through the app; inspect its report and model attribution. Copy historical language/markup/assistant reports and compare IDs, timestamps, JSON and null measurements. Confirm service authentication and cancellation. |
| ISS-21 | Interrupt pretrain, masked SFT and RL, resume and compare optimizer/RNG/data/curriculum continuation. Fail a replacement write and registration separately; restart and recover the previous complete point. Many saves must retain one automatic generation plus explicit/reference exceptions, with bounded temporary disk use. |
| ISS-22 | Pause/resume in process; restart while paused with queued dependencies. Eligible work starts without a resume click. Validate fresh and upgraded schemas without `queue_state`. |
| ISS-23 | Copy conversations/files and compare exact bytes and all source columns. Exercise owners, cascades, binding rollback, feedback, usage, duplicate request IDs, disconnects, tool rounds and interjections. Verify history/file access while inference is unavailable. |
| ISS-24 | Upgrade populated scheduling history; compare UUIDs, dependencies and domain output references. Exercise retry/cancel/event delivery for every supported kind, including benchmark chains. Concurrent chat, training and benchmark requests must respect one device gate. |

Run the LLM, bench-engine and Rust suites against disposable databases/artifacts.
Rust database fixtures require a PostgreSQL role permitted to create temporary
test databases. Bench-engine's fixtures require `pytest` and `pytest-asyncio` in
addition to its service requirements. Migration fixtures and populated transfer
exercises are separate from ordinary SQLite fixtures.

The transfer procedure and rollback boundary are in
[ownership-cutover.md](backend/ownership-cutover.md). Preserve database/artifact
backups until deployment acceptance; archived source tables are not a substitute
for a coordinated rollback after new writes.

## Latency comparison still required

Use the checkpoint revision `129040f` and this working tree with identical copies
of the conversation/file dataset. Record history and file request latency with
the same backend/database placement, response sizes and warm-up conditions.
Report sample count, median and p95 separately for cold navigation and browser
hover/focus-prefetched navigation. Verify shared pending requests and cache
invalidation after create, send, feedback, rename, delete and logout. Retain raw
measurements; no generation-speed improvement is claimed from storage ownership.
