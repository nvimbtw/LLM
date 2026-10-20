# Modular architecture implementation evidence

This ledger tracks FEAT-4 delivery gates. A contract/build pass is distinct from
real production integration. GPU support remains in scope; the user explicitly
deferred hardware verification to their other machine. Deferred GPU checks must
remain runnable and must never be reported as CPU-tested GPU evidence.

| Issue | Owner/outcome | Evidence/status |
| --- | --- | --- |
| ISS-48 | Capture/runtime feasibility | CPU scope passed. Independent runtime13/13 in43.190s; capture plus legacy numerical suites56/56 in44.21s. Later capture refinements36/36 plus baseline22/22 passed by capture owner. |
| ISS-49 | Versioned contracts | 488 combined contract/interface/runtime tests passed in3.58s;67 schemas regenerated and drift-checked; compile/doc/link/diff checks passed. |
| ISS-50 | SDK/resolver/core | Independent current core suite40/40 passed in9.67s; critical boundary review recorded in the [resume handoff](modular-architecture-resume.md). The eight resolver-negative fixtures now exercise production resolution/building. |
| ISS-51 | Catalog persistence | Implemented: live PostgreSQL migrations/concurrency/retention and legacy checks62/62 in9.78s; final completion/chain guards40/40 in8.32s. Sole Alembic/schema/storage-path owner. |
| ISS-52 | Supervised workers | In progress; production parent admission and cleanup; actual GPU checks user-deferred. |
| ISS-53 | Builtins/recipes | Pending50; synthetic contract fixtures do not prove initialization/numerical equivalence. |
| ISS-54 | Capture/validation | Pending50/52/53. |
| ISS-55 | Packages/checkpoints | Pending50/51/53; storage paths hand off from51. |
| ISS-56 | Engine APIs/validation queue | Pending51/52/54; runner ownership after52. |
| ISS-57 | Memory/optimizer/diagnostics | Pending50/53; shared loop integration belongs58. |
| ISS-58 | Submission/pretrain/recovery | Pending52/54/55/56/57; shared config/dispatch/loop owner. |
| ISS-59 | SFT/token rows | Pending58; no shared-loop edits outside58 owner. |
| ISS-60 | RL | Pending58; real gym effects and eval-with-grad evidence required. |
| ISS-61 | Serving/scoring | Pending52/54/55. |
| ISS-62 | Rust | Contract-based work allowed; live56/58/61 gates required. |
| ISS-63 | Frontend foundation | Contract-based work allowed; shared editor/controls/dependency owner. |
| ISS-64 | Graph UI | Pending63; actual54 graph gate required. |
| ISS-65 | Authoring UI | Pending63; live56/62/64 gates required. |
| ISS-66 | Training/model UI | Pending63; sole jobs.ts owner; live execution gates required. |
| ISS-67 | Validation/navigation UI | Pending63; requests jobs.ts changes through66. |
| ISS-68 | Cross-module acceptance | Pending all delivery slices; explicit user-deferred GPU verification retained. |

ISS-49 reproduction from `llm-engine/`:

```sh
../engine/.venv/bin/python -m llm.architecture.contracts --check
../engine/.venv/bin/python -m pytest tests/test_architecture_contracts.py tests/test_architecture_interfaces.py tests/test_architecture_runtime_contract.py
```

Local detailed reports: `/tmp/iss48-runtime-independent.log`,
`/tmp/iss48-capture-independent.xml`, `/tmp/iss49-contracts.xml`.
Capture/runtime commands and numeric/resource limitations are retained in their
[records](architecture-capture.md) and [runtime evidence](modular-runtime-feasibility.md).
ISS-49 treats eight `stage: resolution` fixtures as downstream expectations;
ISS-50 adds their production resolver/builder rejection evidence. No new catalog/routes/worker integration was claimed
by the contract slice. Shared contracts and policies are mapped in
[architecture-contracts.md](architecture-contracts.md).

PostgreSQL checks must own unique disposable database names; never reuse or drop
a fixed shared test database during parallel work. No commit or push is authorized.

ISS-51 reproduction uses the command in [storage verification](architecture-storage.md#verification).
The62-test run covers the complete persistence slice and existing migration/store/chain
behavior. The later40-test run rechecks changed store/route guards, including refusal
to retain evidence before validation completion and refusal to chain validation as a
model producer. JUnit reports: `/tmp/iss51-persistence.xml`,
`/tmp/iss51-final-store-routes.xml`. The coordinated49 fractional queue-rank and
blank-label corrections passed355 affected contract cases;67schemas were regenerated.
No Rust SQL/schema changes occur in51, so SQLx regeneration is not applicable.
Production worker/HTTP/publication integration is explicitly reserved for52/55/56/58.

The user confirmed pause; root marked the goal PAUSED. ISS-50/51/52 remain
incomplete pending final review. Work stopped before starting further slices.
See the [resume handoff](modular-architecture-resume.md) for uncommitted state,
active ownership and the exact next verification/integration steps.
