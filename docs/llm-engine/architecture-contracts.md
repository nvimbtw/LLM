# Modular architecture contracts — version 1

The declarations in `llm/architecture/spec.py`, `wire.py`, `interfaces.py` and
`llm/runtime/protocol.py` are the authoritative FEAT-4 version-one contracts.
They define boundaries; production storage, construction, supervision, routes
and UI belong to subsequent issues. A schema does not introduce a running API.

- [SDK and supported source](architecture-sdk.md)
- [YAML and exact examples](architecture-yaml.md)
- [Resident worker protocol](architecture-runtime.md)
- [Catalog persistence and transaction APIs](architecture-storage.md)
- [Implementation evidence](modular-architecture-evidence.md)
- [Capture evidence](architecture-capture.md)
- [Process evidence](modular-runtime-feasibility.md)

## Shared verification

From `llm-engine/`, using the installed checkout environment:

```sh
../engine/.venv/bin/python -m llm.architecture.contracts
../engine/.venv/bin/python -m llm.architecture.contracts --check
../engine/.venv/bin/python -m pytest tests/test_architecture_contracts.py tests/test_architecture_interfaces.py tests/test_architecture_runtime_contract.py
```

The first command regenerates standalone JSON Schemas in
[`contracts/modular-architecture/v1/schemas/`](../../contracts/modular-architecture/v1/schemas).
The check fails on missing, obsolete or changed schemas. Consumers share these
schemas and fixture indexes; there are no separately maintained Rust/TypeScript
schema copies. Rust keeps engine-owned YAML/model payloads opaque.

Each indexed fixture names its model, expected validity and inline `value`, exact
`raw` JSON or relative payload `path`. Optional checks verify current receipts,
graph cursors and runtime identity/packet bounds. Duplicate/nonfinite JSON and
unsafe/duplicate/aliased YAML fail before declaration interpretation.

Eight `stage: resolution` negatives intentionally pass declaration checks. Their
actual scope/shape/alias/work/dependency rejection is an ISS-50 resolver handoff,
not a resolver test claimed by ISS-49. Catalog fixture UUIDs/hashes are synthetic
pinned snapshots. The complete planned standard/expanded/looped/custom recipes
are not deployed accepted revisions. ISS-53 must provide real source manifests,
constructor ordering and numerical/initializer compatibility evidence.

## Identity and authoritative bindings

| Identity | Authority and lifetime |
| --- | --- |
| Resource UUID | Stable catalog identity; display metadata/draft may change. |
| Revision UUID/number/hash | Immutable source/YAML/interface and dependency pins. |
| Draft version/hash | Exact saved bytes, independent from metadata version. |
| Receipt UUID/hash | Exact subject/dependencies/values/input/tokenizer/runtime/profile evidence. |
| Definition SHA-256 | Resolved semantic recipe, sources, parameters, tokenizer, context and aliases. |
| Graph SHA-256 | Immutable profile/mode/signature artifact, separate from semantic definition identity. |

`canonical_digest` hashes sorted compact UTF-8 JSON after typed normalization,
including defaults and rejecting nonfinite values. A definition excludes its own
digest, run/receipt UUIDs, device profiles, graph artifacts, actors, display names
and current archive state. Source hashes identify exact bytes. Compatibility
belongs to immutable SDK/builtin/source manifests. JSON integer parameters stay
within JavaScript's exact range. Container contents are owned snapshots; frozen
record attributes do not permit consumers to mutate shared dictionaries.

Only the tokenizer owns vocabulary size, token IDs and fingerprint. Bindings use
`${tokenizer.vocabulary_size}` or supported reserved token IDs. The recipe's
`context_window` literal/parameter binding is the sole trained-context authority;
a preview profile cannot override it. Resolved definitions check exact parameter
keys/types, context and aliases, and distinct dependency sources. Compatible
reserved-label adoption records `derived_from` and a new fingerprint without
changing IDs or base-package bytes.

## Validation and durable submission

Inspection requires request/subject identity and forbids constructor values,
tensors/profile/tokenizer/context. It inspects declarations without construction;
saving/listing never imports source. Full validation supplies explicit runtime
and input bounds. Ordinary block inputs use bounded synthetic tensor specs, not
uploaded tensor blobs. Architecture inputs follow the causal LM interface.

`execution_jobs` remains the only lifecycle authority. A completed `invalid`
validation differs from worker failure. Verdicts are `inspection_only`, `invalid`,
`provisional`, `valid`. Valid full reports require passed capture/numerical
stages, no failed/unfinished stages and a receipt naming published complete-graph
evidence. Pending tokenizer reports may skip unresolved construction/capture/
numerical stages and retain provisional metadata without graphs. Shared fixtures
accept that provisional report and reject the same skipped stages on a valid
report. Inspection/provisional results never authorize acceptance or optimizer
execution.

Receipt identity includes exact source/dependencies/parameters, named input-profile
digest, tokenizer, context and runtime. Profiles record torch/SDK/device/precision,
TF32, numerical seed, explicit strict-export policy, guards, modes and all enabled
shape/signature domains. Singleton specialization is explicit. Changed source,
tokenizer, runtime, parameters or input bounds needs new evidence. Optional
capabilities need their own evidence; unavailable metrics remain absent.

Acceptance locks the current draft/dependency snapshot, checks expected version,
content/dependency hashes and a fully bound valid receipt, then commits immutable
revision and retained evidence atomically. One example accepts a revision, not
all possible configurations. Evidence survives pruning its validation job.

Enqueue atomically pins accepted root, full dependency closure, parameters,
acceptance/current evidence and reserved output UUID. Pending tokenizer or absent
exact evidence sets `final_preflight_required`. Final binding and exact preflight
run inside the admitted training job before optimizer work, never as a child
queued behind the same device lease. Validation previews cannot override context.

Validate every chain step and pin all waiting-step dependencies/evidence before
creating any rows; commit the whole chain together. Validation cannot be a
model-producing step even when its job is `done`. Identical actor/request UUID
retries return the original operation; conflicting arguments return 409.
Validation retry creates a fresh identity over pinned bytes without `model_name`.
Training retry/recovery retains the saved definition rather than current defaults.
Archive hides new root selection; an unarchived accepted recipe still works when
only a pinned dependency is archived. Real PostgreSQL tests prove concurrency.

## HTTP families

All new authoring/source/revision/result/graph routes require Rust AdminUser and
actor-preserving audience-scoped service tokens. The browser calls Rust only.
Literal `/architectures/schema` precedes UUID routes.

| Route | Contract |
| --- | --- |
| GET/POST `/blocks`, `/architectures` | `CatalogQuery`/`CatalogPage`; create → 201 `ResourceCreated`. Invalid drafts remain saveable. |
| GET/PATCH `/{id}` | `ResourceDetail`; `MetadataPatch` uses expected metadata version. Builtins read-only. |
| GET/PUT `/{id}/draft` | `BlockDraftView`/`ArchitectureDraftView`; optimistic saves → `DraftSaved`, no validation badge. |
| GET `/{id}/revisions[/{revision_id}]` | `RevisionPage`, immutable `BlockRevisionView`/`ArchitectureRevisionView`. |
| POST `/{id}/validations` | `ValidationRequest` → 202 normal `ValidationRun` envelope. |
| POST `/{id}/revisions` | `AcceptRevision` → 201 `AcceptedRevision`. |
| GET `/architectures/schema` | `ArchitectureSchema`: real recipe schema/limits and prospective installed runtime profile/devices; no evidence. |
| POST `/architectures/{id}/declaration` | Exact `DraftSubject` → `RecipeDraftInspection`, safe YAML declarations only, never numerical authorization. |
| GET `/validations/{id}` | Durable `ValidationReport`. |
| GET `/validations/{id}/graph`, `/models/{id}/graph` | `GraphQuery` → `GraphFragment`; never recapture during reads. |
| GET `/models/{id}/definition` | `ModelDefinitionSummary`, no source import or weight loading. |
| GET `/models/{id}/sources/{revision_id}`, `/validations/{id}/sources/{revision_id}` | `RetainedSource`: exact bounded UTF-8 bytes/hash and relative display path from immutable package/publication; no imports, tensor reads or current catalog fallback. |
| GET `/architectures/revisions/{revision_id}` | Exact accepted `ArchitectureRevisionView`, without mutable head selection. |
| POST `/architectures/revisions/{revision_id}/training-preview` | `TrainingArchitectureRequest` → `TrainingArchitecturePreview`; prospective resolved metadata, no validation receipt or queue work. |
| Existing training/run/event routes | Add `ArchitectureSelection`, validation output and `ValidationEvent`; retain cancel/observer separation. |

Use 403 authorization/read-only, 404 missing, 409 stale/archive/conflict, 413 byte
limits, 422 interpreted values. `ErrorResponse` carries bounded phase/path/source
`Diagnostic`, not unrestricted logs, Python objects or tensors. `cancelling`
remains nonterminal; observer detachment does not cancel durable validation.

Rust derives `AuthorizedVisibility` from verified claims, overwriting browser
attempts to set it. Engine filters active/history/counts before pagination. Deny
validation detail/events before any bytes stream; generic model/run metadata,
errors and logs cannot expose administrative source/graphs. Ordinary published
chat and compatible existing attention access remain available.

Ordinary signed-in users can explicitly discover `/models/{id}/attention-points`
(`InspectionCatalog`) and request `/models/{id}/attention` (`AttentionSelection`
→ `AttentionMatrix`). Candidates expose actual instance/call identities separately
from numerical proof bounds. Matrix axes, base64 length, finite probabilities and
definition/profile/worker/view evidence are checked in both language decoders.
Legacy numeric layer/head selection retains its actual model dimensions; generic
views never synthesize that layout. These reads contain no custom source/graphs.

Graphs distinguish instance, execution occurrence and shared parameter owner.
Frontend lays out engine semantics. Cursors bind graph hash, receipt/profile,
view/mode/signature/scope and position; server checks both current manifest hash
and query. Scope replay fails. Read bounded lazy children, never flatten all loops.

## Policy and ownership

Initial ceilings from the measured prototype: 256 KiB source/YAML, 128 expanded
calls, eight logical nesting levels, 4096 nodes, 16384 edges, 8 MiB graph bytes,
100 catalog rows and 256 nodes/1024 edges per graph page. Synthetic tensors cap
at 16,777,216 elements and still require byte-aware admission. Worker time/memory
bounds also apply while capture discovers graph size. Unknown operation/memory
estimates fail boundedly; allowed shapes do not promise capacity. Batch-three
prototype evidence is a profile, not a universal product restriction.

ISS-49 owns coordinated contract/schema/fixture changes. ISS-50 owns supported
source, parser/resolver/builder. ISS-51 owns all migrations/storage ordering and
paths. ISS-52 owns production supervision; runner integration passes to 56 then58.
ISS-55 owns package/checkpoint publication. ISS-62 is sole Rust lane; ISS-63 owns
frontend shared adapters/controls/dependencies. ISS-66 owns `lib/jobs.ts`; 67
requests edits through it. No competing copies of domain calculations or queues.

PostgreSQL checks use unique disposable database names per run and explicit owner
markers; never drop a fixed shared scratch database under parallel agents.
Migrations remain immutable after application. Real locking/publication/restart
and all service/browser integration remain later gates.

The user deferred GPU hardware checks to their other machine. GPU support stays
in scope; CUDA/bf16/allocator/assertion/termination tests remain runnable deferred
checks. CPU passes do not certify GPU behavior. Contracts are versioned in the
branch worktree; no commit/push is implied.
