# Modular architecture contracts

`v1/` pins FEAT-4 SDK, recipes, validation, graphs, packages and runtime messages.
Python declarations in `llm-engine/llm/architecture/` and `llm/runtime/protocol.py`
generate the standalone `v1/schemas/` files.

From `llm-engine/`, run `../engine/.venv/bin/python -m llm.architecture.contracts --check`
and the architecture contract/interface/runtime suites. Regenerate with the same
command without `--check` after coordinated declaration changes.

Indexed JSON cases name a model, expected validity and inline value, raw JSON or
relative payload path. `stage: resolution` negatives are declaration-valid
resolver handoffs, not executed resolver tests in this slice. See the
[engine contract topic](../../docs/llm-engine/architecture-contracts.md) for exact
identities, APIs, ownership, evidence and limitations.

Synthetic catalog revisions make complete planned standard/looped/custom recipe
examples self-contained. They are not deployed accepted resources. ISS-53 supplies
real sources and initializer/numerical equivalence evidence. Source/graph artifacts
remain private administrative data.
