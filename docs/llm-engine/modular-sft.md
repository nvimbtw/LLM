# Saved-definition supervised fine-tuning

Queued SFT inherits the complete saved recipe, parameters, loop counts, source
closure, aliases, context and vocabulary. Structural overrides are refused.
It uses the existing runner, device lease, supervised backend, training loop and
package adopter. Source, weights, optimizer and reference remain worker-owned.
Numerical/adapter checks are distinct from the still-open process/PostgreSQL gates
in the [delivery ledger](modular-architecture-evidence.md). GPU is user-deferred.

## Prepared inputs and tokenizer authority

`api/sft_handler.py` verifies the admitted job, inherited pin and registered saved
package before acquisition. It checks optimizer/schedule/input bounds. The actual
initiating actor reaches the existing Rust-scoped dataset client. `SftConfig`
adds `precision`, `compile_model` and `allow_tf32` execution options with pretrain
semantics; CPU uses fp32 and CUDA fallback records the actual arithmetic.

Modular `POST /sft` requires an administrative actor and request UUID. Rust's
`POST /runs/sft` forwards the query and signed administrative scope through the
same bounded status-preserving adapter as pretrain. Actor/request replay retains
the original job. Recovery may copy an original-tokenizer enqueue pin while its
registered checkpoint contains derived labels; exact original-base/recovery
tokenizer bytes must prove that compatible transition before acquisition.

A waiting chain may retain its predecessor's pending-tokenizer pin after that
predecessor finishes. The registered saved model and durable parent preparation
then authorize actual final bytes and compatible label adoption. The chain pin
stays byte-for-byte unchanged; no old tokenizer build or predecessor job history
is needed for subsequent source loading/adoption.

`orchestrate/modular_sft_data.py` uses the canonical renderer in one private
temporary directory. Corpus/mask bytes, document rows and source characters are
bounded before append/encoding. Cancellation removes partial files. Identity
includes actual corpus/mask bytes, spans and ordering; reasoning and final
tokenizer participate in the private key. Preparation renders anew rather than
reusing a shared cache across vocabularies. The child verifies hashes, binary
alignment, nonzero target count, conversation count, vocabulary and fingerprint
before model construction. Admission charges all mask/probes and every copy of
both saved and separate original-reference packages.

Ordinary tokenizer bytes remain exact. Memory-bearing data can privately adopt
the fixed memory labels, preserving IDs/merges/vocabulary and reserved-ID facts.
Derived bytes receive a deterministic UUID/hash and `derived_from` source hash.
The original enqueue pin is immutable. `api/sft_authority.py` atomically publishes
parent-owned final-input authority under `storage.training_preparation_dir(job)`
before worker launch. It binds config, source model/definition, actual data,
final tokenizer and allowed rendering/warm-start provenance. Adoption checks
that record and actual tokenizer bytes before rename/registration; worker metadata
or `derived_from` alone cannot authorize a transition.

Success removes preparation after session cleanup, temporary-input cleanup and
registered-final verification. Failure retains it for post-rename orphan adoption.
Startup reconciles packages after exclusive worker cleanup, then retires
terminal/pruned-job records without unresolved publications. Active/unresolved
jobs retain authority. Unreadable package metadata prevents this cleanup and
emits a warning.

## Initialization, reference diagnostics and recovery

Complete known marker history authorizes initialization; unknown ancestry never
permits a reset. Pretrain carries known producer history and unions it on
continuation. SFT records the union and explicit completeness. The whole-model
token-row capability owns tied/untied tables. Standard/legacy share the original
mean initializer; unsupported models preserve weights and report a skipped
outcome. `warm_start` names initialized/skipped rows; `warm_started` counts only
initialized rows. Exact recovery preserves the original result without repeating
initialization. The parent checks row ownership against prepared authority.

Modular RL marks inherited marker history incomplete: rollout context/actions
may train rows absent from pretraining. A later SFT cannot use the older producer
record to reset those rows, even when the base corpus's own history was complete.

Optional pretrain probes sample bounded original-base corpus windows. Held-out
chat probes stream at most 128 files, 8 MiB total and 256 KiB per line, stopping
at the diagnostic row/context limit. Malformed rows are omitted before rendering.
Missing/over-budget optional measurements stay absent. `sft_probe_v1` grants hold
int64 NPY arrays with explicit row/context/vocabulary checks. Masks use
`loss_mask_v1`; neither can substitute for a corpus or package.

A real chat probe constructs a frozen evaluation reference. Meta admission
charges its captured weights/workspace before real allocation. Construction and
diagnostics preserve training RNG, modes, parameters, buffers and gradients.
Masked/prompt/position/conversation losses, pretrain loss and KL(base || tuned)
are bounded. Optional diagnostic OOM/budget exhaustion disables sampling;
model/training errors propagate.

Fresh SFT loads weights and starts a new optimizer. Exact recovery requires SFT
optimizer ownership/order, original request/data/mask/schedule, next-batch cursor,
RNG and retained replay selection. KL recovery grants and verifies the original
base separately. Targetless batches consume data without advancing optimizer
steps; the existing batch-based schedule remains. Final SFT metadata explicitly
accounts for skipped batches. Pretraining cannot claim that allowance.
Cancellation stops before publishing a finished output.

## Verification

From `llm-engine/`:

```sh
../engine/.venv/bin/python -m pytest tests/test_modular_sft_worker.py tests/test_modular_sft_loop.py tests/test_token_rows.py
../engine/.venv/bin/python -m pytest tests/test_sft_publications.py tests/test_training_admission.py tests/test_training_handler.py
../engine/.venv/bin/python -m pytest tests/test_sft_tokenizer.py tests/test_sft_rendering_budget.py tests/test_project_memory.py
../engine/.venv/bin/python -m pytest tests/test_supervised_sft.py tests/test_sft_publications_postgres.py
../engine/.venv/bin/python -m llm.architecture.contracts --check
```

Numerical tests train actual standard/shared/custom models and compare serialized
optimizer/RNG/cursor recovery. Their trusted tensor reader first verifies parent
refusal, then decodes its own safe tensor data; it does not spoof worker identity.
Parent tests explicitly label synthetic graph/tensor outputs. Real supervisor,
PostgreSQL and GPU checks remain separate evidence.
