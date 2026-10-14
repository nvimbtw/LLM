# Project context and retrieved messages

Rust owns private project membership, source authorization and similarity
retrieval. The LLM engine receives authorized facts and source messages for one
request; it renders them, fits their tokens and performs stateless generation.
It neither retrieves messages nor stores a project or memory index.

## Request boundary

`ChatConfig` and `/chat/prepare` accept optional `project` and `memories`.
The same context travels to `/chat/context-floor`. The shared
[`project-memory.json`](../../contracts/project-memory.json) fixture pins their
cross-language shape.

| Object | Required fields and character limits |
| --- | --- |
| `project` | `project_id`: UUID; `name`: 1–120; `description`: 0–2,000; `notes`: 0–8,000 |
| Each `memories` entry | `memory`: 1–2,000; `thread`: UUID; `message`: UUID; `role`: `user` or `assistant`; `age_seconds`: nonnegative integer |

There are at most eight memory candidates. Lengths count Unicode characters;
unknown object fields, fractional ages, booleans and numeric strings for ages
are refused. Candidate ordering is Rust's retrieval priority, preserved here.
`thread` and `message` identify the source; age is the elapsed number of whole
seconds supplied by Rust, rather than a timestamp interpreted by Python.

## One renderer for serving and training

`llm/chat/template.py` owns both forms. Project facts are a separate input-only
system block, after any ordinary system block:

```text
<|system|>{"project":{"project_id":"...","name":"...","description":"...","notes":"..."}}<|/system|>
```

Every memory is an input-only block outside all conversational turns:

```text
<|memory|>{"memory":"...","speaker":"assistant","source":{"thread":"...","message":"..."},"age_seconds":3600}<|/memory|>
```

The JSON is compact, with the field order shown above. Every literal `<|` inside
JSON is written as `\u003c|`, so a quoted marker cannot end its enclosing block
or create a forged user/assistant turn. JSON parsing recovers the original
quoted text. Remembered assistant answers omit completed thinking blocks and
an unfinished thinking block's remaining suffix; user-authored memories keep
their text, with markers escaped.

The grammar forbids both memory markers and both system markers as generated
output. Neither a project's block nor any part of a memory is an SFT loss
target. An assistant source turn containing raw memory delimiters is refused,
rather than teaching the model to generate memories.

## Context budget and priority

The normal reply reserve remains unchanged. Optional project/memory blocks
can use only space remaining after the complete newest user question, its
following assistant/tool continuation turns, the ordinary system block and the
trailing assistant cue. If this protected suffix already exceeds the allowance,
the optional context is omitted and the existing ordinary hard-cut policy runs.

Within that remaining space:

1. Combined optional context uses at most one third of prompt allowance.
2. The project block uses at most 512 tokens, including its wrappers and JSON.
   Identity and name remain complete. Notes get priority over description;
   either body may be shortened to a character prefix. If identity/name alone
   cannot fit, the project block is omitted.
3. All included memories together use at most 512 tokens, and each individual
   memory uses at most 256 tokens, including wrappers and provenance JSON.
   Memories are considered in their supplied order. Text may be shortened to a
   prefix; speaker, source IDs and age remain complete. A memory whose
   provenance cannot fit is omitted.
4. Older ordinary history fits around the included context, newest first.

`/chat/context-floor` uses the same optional allocation and scaffold accounting
for requests carrying a project or memories. Ordinary requests retain their
previous conservative floor. Token budgets are ceilings, not guarantees that
all notes or all eight candidate memories are included. A tiny model may receive
no optional context. The reply cue cannot be hard-cut; a prompt allowance too
small for that cue is an actionable error.

## Legacy tokenizer compatibility

Memory delimiters consume reserved slots **47 and 48**; the special block stays
64 slots and existing marker, byte and merge IDs do not move. A memory-bearing
request adopts only these two slots on a private tokenizer copy using
`corpus.relabel`'s existing placeholder validation. The cached tokenizer, saved
model tokenizer and model weights remain unchanged. Ordinary serving requests
use their existing tokenizer directly, preserving the pinned dataflow trace.

`/chat/prepare` checks this adoption before Rust stores new input. A conflicting
named slot or a foreign vocabulary without the fixed reserved block is refused.
The backend must supply trusted context to preparation as well as generation.
The explicit `python -m scripts.relabel_reserved_slots` dry run and `--apply`
workflow remains available when an operator wants permanent vocabulary adoption.

An SFT run whose corpus actually uses memory IDs saves an adopted tokenizer
with its new output model. A run without memory examples retains the base
tokenizer byte-for-byte. Dataset `special_ids_seen` includes the new IDs, allowing
the existing warm-start policy to initialize previously unseen marker rows.
Unknown base-model marker history still follows its existing no-reset policy.
Naming a token supplies no pretrained recall ability; useful behavior requires
suitable examples and evaluation.

## SFT source messages

An ordinary dataset document's `messages` list carries source context as JSON
strings in `content`:

```json
[
  {"role":"project","content":"{\"project_id\":\"11111111-1111-4111-8111-111111111111\",\"name\":\"Notes\",\"description\":\"\",\"notes\":\"Use PostgreSQL.\"}"},
  {"role":"memory","content":"{\"memory\":\"Commit before acknowledging.\",\"thread\":\"22222222-2222-4222-8222-222222222222\",\"message\":\"33333333-3333-4333-8333-333333333333\",\"role\":\"assistant\",\"age_seconds\":3600}"},
  {"role":"user","content":"When is the write acknowledged?"},
  {"role":"assistant","content":"After the complete reply commits."}
]
```

Each example allows one project role and eight memory roles. Payloads use the
same required fields and validation as live requests, with a 65,000-character
ceiling on source JSON before parsing. Malformed context drops that source row
and contributes to the build's skipped count. Examples still end on an assistant
reply. Context roles can occur before the user turn they inform; only subsequent
assistant replies receive their information through causal attention.

These are recorded, human-curated or teacher-provided source messages. The
engine creates no summaries and introduces no separate training job type.
`training_segments` invokes the same canonical renderer as serving, masks all
context tokens and trains only assistant answer content and its closing marker.
Training renders the validated complete source payload; serving may shorten it
under the per-request token limits described above.

Packed SFT keys and sidecars include `context_format: 1`, invalidating older
packs that skipped these roles. Existing ordinary source examples still render
to the same token IDs and masks. See the data engine's
[preservation contract](../data-engine/project-memory.md) for cleaning behavior.
