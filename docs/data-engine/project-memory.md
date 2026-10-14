# Project and memory training context

An SFT conversation can contain input-only `project` and `memory` messages
alongside its ordinary user/assistant exchange. Their `content` is a JSON string
carrying the same object Rust supplies during serving. The shared example is
[project-memory.json](../../contracts/project-memory.json); the LLM engine owns
the validated payload and canonical rendering described in its
[corpus contract](../llm-engine/corpus.md).

The data engine acquires, stores, processes and exports these messages. It does
not retrieve a user's live project conversations, render prompt blocks or train
weights. Project conversation retrieval belongs to Rust and stays private to
that project's owner. Preparing a training dataset does not implicitly export
private projects or turn their conversations into a shared corpus.

Use the ordinary `messages: [{role, content}]` document shape:

- A `project` message carries the project ID, name, description and notes.
- A `memory` message carries remembered text, source thread/message IDs, source
  role and age in seconds.
- Both content values remain JSON strings until the LLM engine validates and
  renders them as input context. They must not become assistant training targets.

`normalise-roles` recognizes both canonical context roles. `convert-chat` keeps
their exact content and metadata while converting assistant replies; the
`markdown-emphasis` step already leaves non-assistant turns alone. A context
payload quoting an earlier tool call does not make the current example tool-using,
so `drop-tool-calls` does not interpret these payloads as live calls. Ordinary
tool turns and actual calls retain their existing filter behavior.

Malformed context is refused or skipped by the LLM engine's source parser under
its existing corpus validation policy. The data engine passes the opaque payload
through rather than duplicating its schema or marker renderer. Source-specific
converters such as Hermes and SmolTalk still apply only to their source formats.
The gym demonstration endpoint retains its gym trajectory roles.

The focused suite `tests/test_project_memory.py` verifies normalization, exact
JSON preservation, quoted-call filtering and the stored/processed/exported path.
