"""Converting SmolTalk rows into this project's conversation format.

SmolTalk is chat-shaped already, but in its own dialect. System instructions
arrive as a `system` message; user and assistant turns are plain text; and
function calls are embedded in the assistant's text as
`<tool_call>[{"name": ..., "arguments": {...}}]</tool_call>`, with the
conversation ending on the call and no tool-role turn to follow it.

Every tool here is called through one marker with the tool named inside it, so
a call is rendered as
`<|tool-call|>{"tool": "web-search", "argument": …}<|/tool-call|>` — the layout
`llm-engine/llm/chat/tools.py` owns. The argument is the user's own question: the
backend searches it literally, so the source's own JSON arguments would be a
query a provider would choke on. Rephrasing the question is a real system's
job, not a cleaning script's — the honest text is what was asked.

Assistant prose is wrapped in `<|par|>` so the model learns the answer markup
from its first SFT run instead of meeting it cold at serving time. A system
message that describes the source's own tool format (`<tools>` blocks, the
`<tool_call>` output contract) is dropped by default: conditioning the model
on an instruction to emit a format we have just converted away would train a
mismatch. Kept verbatim with `keep_tool_system`.

A malformed `<tool_call>` drops the whole conversation — rendering it as
literal text would teach the model to recite JSON, and guessing at its
intent is worse than losing the row.

Markdown the source wrote (`**bold**`, `*italic*`) is *not* converted here:
that is the separate `markdown-emphasis` processor's job, so any markdown
corpus can use it and its behaviour is previewable on its own. Run
`convert-smoltalk` before it in a pipeline.

The output is `[{role, content}]`, the same shape everything else here stores
and the LLM engine's `sft._conversation` turns into `{system?, turns}` — where the
engine wraps each `tool` turn's text in `<|tool-result|>` at render time, so
the stored text is only what goes between those markers.
"""

import json
import re

from processors.spec import Document, Param, Processor, State
from store.schema import DatasetKind

SFT = (DatasetKind.sft,)

#: The one marker every tool is called through. Written out rather than
#: imported: the data engine and the LLM engine are separate services with separate
#: virtualenvs, and `llm-engine/llm/chat/tools.py` is the source of truth for the
#: layout either way.
TOOL_CALL_OPEN = "<|tool-call|>"
TOOL_CALL_CLOSE = "<|/tool-call|>"

WEB_SEARCH = "web-search"

#: SmolTalk's function calls, embedded in the assistant's text. Lazy enough to
#: tolerate the occasional unclosed tag as plain text instead of eating the
#: rest of the turn.
_TOOL_CALL = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

#: Apigen system prompts name the functions on offer and rigidly specify the
#: `<tool_call>` output contract — an instruction to emit a format this
#: converter just changed. Anything quoting `<tools>` is one of those.
_TOOL_SYSTEM = "<tools>"


def _arguments_of(call: dict) -> str:
    """The call's arguments as a bare string, when one is genuinely there."""
    for key in ("query", "question", "search", "text", "prompt"):
        value = call.get("arguments")
        if isinstance(value, dict) and isinstance(value.get(key), str):
            return value[key].strip()
    if isinstance(call.get("arguments"), str):
        return call["arguments"].strip()
    if isinstance(call.get("arguments"), dict):
        value = " ".join(
            str(v).strip()
            for v in call["arguments"].values()
            if isinstance(v, (str, int, float)) and str(v).strip()
        )
        return value.strip()
    return ""


def _render_call(raw: str, query: str = "") -> str | None:
    """One `<tool_call>` body as a web-search call.

    The body is `[{name, arguments}]` or a single such dict. The source's name
    is dropped and replaced with `web-search`: the names in this corpus are
    invented per sample (`get_role_ids`, `calculate_bmi`) and nothing in this
    project can run any of them, so keeping one would train a call to a tool
    that will never exist.

    The query is what the search backend will *literally* run against the
    provider, so it must be natural language, never a JSON blob. The most
    honest text available is the user's own question — a real system would
    rephrase it, and this processor is not in the business of inventing
    better phrasings than the model that asked. An explicit `query` wins; the
    args (a `query`/`question` key, or the values joined) are only a fallback.
    """
    try:
        spec = json.loads(raw.strip())
    except ValueError:
        return None
    if isinstance(spec, dict):
        spec = [spec]
    if not isinstance(spec, list) or not spec:
        return None

    blocks = []
    for call in spec:
        if not isinstance(call, dict):
            return None
        arguments = _arguments_of(call)
        blocks.append(_web_search(query or arguments))
    return "".join(blocks)


def _web_search(argument: str) -> str:
    """One call, rendered exactly as `llm.chat.tools.call` renders it."""
    payload = json.dumps(
        {"tool": WEB_SEARCH, "argument": argument.strip()}, ensure_ascii=False
    )
    return f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"


def _assistant(content: str, wrap: bool, query: str) -> str | None:
    """An assistant turn converted: calls become web-search, prose gets `<|par|>`.

    A call may only sit at the top level, so a turn that carries one is never
    wrapped — the prose around the call stays bare while the call is rendered
    as the leaf block the grammar allows it to be.
    """
    parts: list[str] = []
    rest = content
    while (match := _TOOL_CALL.search(rest)) is not None:
        rendered = _render_call(match.group(1), query)
        if rendered is None:
            return None
        before = rest[: match.start()].strip()
        if before:
            parts.append(before)
        parts.append(rendered)
        rest = rest[match.end() :]

    tail = rest.strip()
    if tail:
        parts.append(f"<|par|>{tail}<|/par|>" if wrap else tail)
    return "".join(parts)


def _convert_smoltalk(doc: Document, params: dict, state: State) -> Document | None:
    messages = []
    last_question: str = ""
    for turn in doc.get("messages") or []:
        role = str(turn.get("role") or "").strip().lower()
        content = turn.get("content")
        if not isinstance(content, str):
            return None

        if role == "system":
            if not params["keep_tool_system"] and _TOOL_SYSTEM in content:
                continue
            messages.append({"role": "system", "content": content})
            continue

        if role == "user":
            messages.append({"role": "user", "content": content})
            last_question = content
            continue

        if role == "assistant":
            rendered = _assistant(content, params["wrap_replies"], last_question)
            if rendered is None:
                return None
            messages.append({"role": "assistant", "content": rendered})
            continue

        if role == "tool":
            messages.append({"role": "tool", "content": content})
            continue

        return None

    if not messages:
        return None
    return {**doc, "messages": messages}


PROCESSORS = (
    Processor(
        name="convert-smoltalk",
        title="Convert SmolTalk",
        description=(
            "SmolTalk's <tool_call> blocks become <|tool-call|> web-search calls, assistant "
            "replies are wrapped in <|par|>, and tool-format system prompts are dropped."
        ),
        applies_to=SFT,
        apply=_convert_smoltalk,
        params=(
            Param(
                name="wrap_replies",
                label="Wrap replies in <|par|>",
                type="bool",
                default=True,
                description="Teach answer markup from the first SFT run.",
            ),
            Param(
                name="keep_tool_system",
                label="Keep tool-format system prompts",
                type="bool",
                default=False,
                description="Keep system messages that describe the source tool format.",
            ),
        ),
    ),
)