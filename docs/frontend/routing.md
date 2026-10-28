# Browser routes

Public browser URLs describe the domain. Queue rows, API paths, UUIDs and socket
protocols retain their existing contracts; requests still pass through Rust.

| Destination | Canonical URL |
| --- | --- |
| Queue, directory and history | `/admin` |
| Training composer / execution | `/train/new-train` / `/train/{uuid}` |
| Benchmark history / composer / execution or report | `/bench` / `/bench/new-bench` / `/bench/{uuid}` |
| Gym catalogue / practice composer | `/gyms` / `/gyms/new-gym?gym={slug}` |
| Gym evaluation form and history | `/gyms/sources/{slug}` |
| Gym practice execution or evaluation run | `/gyms/{uuid}` |
| Vocabulary catalogue / composer / resource | `/tokenizers` / `/tokenizers/new-tokenizer` / `/tokenizers/{uuid}` |
| Vocabulary build execution | `/tokenizers/builds/{uuid}` |
| Processor directory / creation / source detail | `/processors` / `/processors/new-processor` / `/processors/{processor_id}` |
| Architecture directory / draft creation / workspace | `/arch` / `/arch/new-arch` / `/arch/{uuid}` |
| Block directory / draft creation / workspace | `/arch/blocks` / `/arch/blocks/new-block` / `/arch/blocks/{uuid}` |
| Durable architecture validation execution | `/arch/validations/{uuid}` |

`jobHref` maps each durable execution kind to its canonical URL. Gym UUID pages
check the execution row first, falling back to an evaluation only after a 404.
Benchmark UUID pages likewise support historical report identities whose
execution has a different UUID or no longer exists. Authorization, upstream
failures and malformed responses never trigger a fallback into another workflow.
Resolving a route starts no job or evaluation; the selected renderer retains its
existing cancellation, retry, socket and transcript behavior.
Completed benchmark executions show their exact output report through the same
report renderer used by history and historical report pages.

Vocabulary resource IDs and build execution IDs are distinct. Build pages link
to the output vocabulary; the vocabulary detail URL never interprets a build ID
as an artifact. Metadata edits retain the vocabulary's stable identity.

Processor IDs are their registered names, such as `strip-urls`. The Admin
Workbench Data group opens the processor directory and datasets as separate
destinations. Administrators author a new file on
the dedicated creation page; detail pages inspect/edit existing source. The
dataset Clean workbench remains available and links to these detail pages.

Workbench training entries use `/train/new-train?mode=pretrain` and
`?mode=sft`. They initialize fresh forms only; retained drafts and explicit
rerun/resume selections preserve their mode until the user changes it.
Workbench Bench links use `/bench?suite=language`, `language_v3`, `markup` or
`assistant`. Unknown suites fall back to Language. Legacy selections open their
own recorded history and offer no new-run action. Architecture and block
catalogues return to **Admin**.

## Compatibility

Architecture creation retains `/arch/new` and `/arch/blocks/new` as aliases
of the canonical creation pages, preserving copy/revision query parameters.

Other old URLs redirect rather than disappear:

- `/jobs` → `/admin`.
- `/jobs/{uuid}` → that execution's canonical URL, resolved with browser auth.
- `/jobs/new-train` and `/jobs/new/train` → `/train/new-train`.
- `/jobs/new/benchmark` → `/bench/new-bench`.
- `/jobs/new/rl` → `/gyms/new-gym`.
- `/benchmark` → `/bench`.
- `/gyms/{slug}` → `/gyms/sources/{slug}`.
- `/gyms/{slug}/runs/{uuid}` → `/gyms/{uuid}`.

Redirects retain all search parameters, including repeated values, composer
seeds (`from`, `model`, `gym`) and encoded `parent` context. Parent links resolve
recognized local destinations with their appropriate label. Execution resolution
belongs to the mounted session and UUID; navigation or session changes cancel
reads and ignore obsolete completions.
