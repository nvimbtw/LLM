# One Marker For Every Tool

- **Date**: 2026-08-22
- **Status**: Implemented
- **Supersedes**: the first decision in
  [`tool-calls.md`](tool-calls.md) ("A marker per tool") and the rejection of a
  generic marker recorded there. Everything else in that record still holds —
  the round loop, who executes, why a turn runs twice.

## Context

Tool calls shipped with a marker pair per tool: `<|web-search|>query<|/web-search|>`,
the stem being the tool's name on the wire. One tool existed, and for one tool
that is the better design — see the reasoning in `tool-calls.md`, which is not
wrong, only outsized by what came next.

What came next is a **sandboxed shell** (`deploy/just-bash/`), and with it the
prospect of a tool table that grows: a shell, a search, and whatever the next
one is. Three things pushed against a marker per tool at that point:

- **Two reserved slots each.** 64 slots, ~45 spent, so this was a bound on
  roughly eight more tools rather than an immediate problem — but it is a bound
  that gets paid in the one currency this project cannot inflate, since
  `SPECIAL_BLOCK` is a vocabulary change and a vocabulary change is a retrain.
- **The tool table became part of the vocabulary.** A model whose tokenizer was
  built in March could not call a tool invented in August, even though nothing
  about the model needed to change. Adopting the marker was cheap
  (`relabel_reserved_slots.py`, no ids move) but it was still a migration, run
  per checkpoint, forever.
- **The grammar's guarantee was worth less than it looked.** `Rules` could
  refuse a call to a tool that does not exist — genuinely useful — but it could
  never say anything about the *argument*, which is where a small model's calls
  actually go wrong.

## Decision

**One marker pair, `<|tool-call|>`, whose body is a JSON object naming the tool
and its argument.**

```
<|tool-call|>{"tool": "web-search", "argument": "tallest mountain in europe"}<|/tool-call|>
<|tool-call|>{"tool": "bash", "argument": "grep -rn cat . | head -3"}<|/tool-call|>
```

1. **`{"tool", "argument"}` is exactly the `tool_call` frame** the generation
   loop already emitted and the backend already consumed, so nothing between
   the marker and the tool maps one shape onto another.
2. **Adding a tool is one row in `llm/chat/tools.py`.** No slot, no relabelling,
   no retrain. A vocabulary frozen months ago can call something invented today.
3. **The result side is unchanged.** `<|tool-result|>` still holds the answer
   and still opens by repeating the call, so the model can tell which of its
   calls is being answered.
4. **`tools.parse` is deliberately forgiving.** It accepts the argument under
   `argument`, `query`, `command`, `input` or `text`, wrapped in an `args`
   object or not, because those are what a model drifts into writing and a
   refusal costs a whole round to teach nothing. It is *not* forgiving about the
   name: a body naming no tool, or a tool this build does not have, is not a
   call.
5. **The retired markers keep their slots.** `<|web-search|>` and
   `<|/web-search|>` stay in `SPECIAL_TOKENS`, emitted by nothing and forbidden
   by the grammar. Removing them would shift every id after them, and
   `corpus.relabel` would refuse a vocabulary that already named them.

## What this costs

Recorded plainly, because these are the reasons the first design was chosen and
they did not stop being true:

- **The grammar can no longer police which tool is called.** The name is
  ordinary text inside a leaf block. A call to a tool that does not exist is now
  caught after the fact — `tools.parse` drops it in the LLM engine, and
  `Registry::answer` turns an unknown name into a `<|warn|>` if one ever gets
  past. The grammar still guarantees the *shape*: one call, at the top level,
  no markup inside, closed before the turn can end.
- **The model has to keep a JSON object balanced.** ~12 tokens of scaffolding
  per call, and a mangled one is a call that does not happen. This is the real
  risk at this model size and it is mitigated in three places: SFT teaches one
  fixed layout (`tools.call` renders every training call), the parser is
  forgiving, and a body that cannot be read costs the call rather than the turn
  — the reply still stands and the person still gets an answer.
- **One embedding row per tool is gone.** The model spells `bash` and
  `web-search` out of ordinary merges now.

The mitigation that matters most is the last of the three: **a malformed call is
a sentence nobody acted on.** `_ToolCalls.observe` returns None, no frame is
emitted, and the turn ends normally. There is a test for exactly this, because
it is the difference between a bad habit and an outage.

## Rejected alternatives

- **Keeping a marker per tool and living with the slots.** The status quo, and
  defensible: 17 slots was eight more tools. Rejected because the cost of the
  generic marker is paid once, in one parser, while the cost of the dedicated
  markers is paid at every tool, in the vocabulary, forever — and because the
  moment a tool table is a *runtime* property, a model and a deployment can
  disagree about it without either being wrong.

- **A generic marker with a bare payload** — `<|tool-call|>bash ls -l<|/tool-call|>`,
  first word is the tool. Three tokens of scaffolding instead of twelve, and
  nothing to escape. Rejected because "the first word" is not a structure: a
  tool that ever takes two arguments has nowhere to put the second, and the
  format would have to change again — which is the thing this decision exists to
  stop doing.

- **Keeping `<|web-search|>` alongside the generic marker** for the tool that
  already worked. Rejected: two spellings of one idea, and the SFT corpora would
  have had to teach both. The retired markers keep their slots but nothing
  emits them.

## Consequences

- Every corpus rendered against the old markers is stale. The converters were
  updated in the same change (`data-engine/processors/smoltalk.py`,
  `hermes.py`, `llm-engine/scripts/fetch_web_search.py`), and a corpus already
  packed has to be rebuilt — the marker ids in it are the retired pair.
- `Tool::run` in the Rust `tools` crate gained a `session` argument. It is not
  about this decision but arrived with it: `bash` keeps a filesystem per chat,
  and every tool takes the key so the sandbox is not a special case.
- `frontend/lib/message-parser.ts::callOf` parses the JSON to caption a call in
  the trace. It reads the same keys as `tools.parse` and is the one other place
  that has to know the layout.
