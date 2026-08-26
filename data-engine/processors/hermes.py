"""Converting `lambda/hermes-agent-reasoning-traces` into shell-tool SFT.

The source is ShareGPT-style (`from`/`value`, already remapped by
`hub.fetch._turn` onto `{role, content}` with the roles passed through as
written: `system`, `human`, `gpt`, `tool`). It is an agentic corpus recorded
against the Hermes harness, whose tools are `terminal`, `read_file`,
`write_file`, `search_files`, `patch`, `process`, `execute_code`, `todo`,
`memory`, `skill_*`, `session_search`, `delegate_task` and `browser_*`.

This project has two tools: `web-search` and `bash`. The corpus has no search
in it at all — across an 820-row sample of the 14,701 the only calls that reach
the network are `browser_navigate`, and 3 of 287 of those carried a search term
— so nothing here emits a web search. What it does have is a great deal of
shell: `terminal`, `read_file`, `write_file` and `search_files` are 84% of
every call made, and all four are things a shell does. Those become `bash`.

    <tool_call>{"name": "read_file", "arguments": {"path": "main.py"}}</tool_call>
    -> <|tool-call|>{"tool": "bash", "argument": "cat main.py"}<|/tool-call|>

`terminal` is already a command line and is kept verbatim. The other three are
*translated* rather than dropped, because the shell already has names for what
they do and the model has read those names a million times in pretraining —
`cat`, `find`, `rg`, a heredoc. What has no shell equivalent in the sandbox
(`patch` — just-bash implements no `patch`; `process` — no job control;
`browser_*`, `delegate_task`, `todo`, `clarify`, `memory`, `session_search`) is
removed along with its result.

## What the model is handed back

A kept call's real recorded output becomes the `tool` turn answering it,
rendered exactly the way `backend/crates/tools/src/bash.rs::render` renders a
live one: output in a `<|codeblock|>`, a non-zero exit named in a `<|warn|>`
first, silence said rather than shown as an empty block. Training on a layout
the server does not serve is the one mistake here that would be invisible until
the model met the real thing.

**One call per assistant turn**, even where the trace made four at once. The
backend answers the first call of a round and carries the rest only as a report
(`api/turn.rs::answer`), because the engine takes one `tool_result` per
request — so a conversation with two answered calls in one turn is a shape the
runtime cannot reproduce.

The recorded filesystem is not the sandbox's. A command in this corpus ran
against a real machine with `~/.hermes/skills` and a populated workspace, and
in the sandbox it will find none of that. That is a limit on what this data can
teach — the *form* of a shell session, not facts about any filesystem — and it
is why the outputs are kept exactly as recorded rather than being made to look
reproducible.

Everything the model said survives: `<think>` becomes `<|thinking|>`, prose
becomes `<|par|>`, fenced code becomes `<|codeblock|>`, nothing is truncated.
Assistant turns left adjacent by a dropped call are merged, since with the call
gone they are one uninterrupted stretch of the model talking.

The `system` prompt is dropped by default and there is nothing human-written in
it to lose: all 820 sampled rows carry the same Hermes function-calling
preamble and `<tools>` inventory, median ~10,000 characters — an instruction to
answer in a format this processor has just converted away, six times longer
than the window these models train at. `keep_tool_system` keeps it verbatim,
the same switch and the same reasoning as `convert-smoltalk`.
"""

import json
import re
import shlex

from processors.markup import (
    CODEBLOCK_CLOSE,
    CODEBLOCK_OPEN,
    blocks,
    neutralise,
    reply_markup,
    split_thinking,
    strip_control,
)
from processors.spec import Document, Param, Processor, State
from store.schema import DatasetKind

SFT = (DatasetKind.sft,)

WARN_OPEN, WARN_CLOSE = "<|warn|>", "<|/warn|>"

#: The one marker every tool is called through. Written out rather than
#: imported: the data engine and the engine are separate services with separate
#: virtualenvs. `engine/llm/chat/tools.py` owns the layout.
TOOL_CALL_OPEN, TOOL_CALL_CLOSE = "<|tool-call|>", "<|/tool-call|>"
BASH = "bash"

#: One call and one response, matched by their tags. Unclosed, either takes the
#: rest of the turn — the only reading that cannot leave half a JSON object
#: behind as prose.
_TOOL_CALL = re.compile(r"<tool_call>(.*?)(?:</tool_call>|\Z)", re.DOTALL)
_TOOL_RESPONSE = re.compile(r"<tool_response>(.*?)(?:</tool_response>|\Z)", re.DOTALL)

#: Hermes' system prompt, recognised by the tool inventory it carries.
_TOOL_SYSTEM = "<tools>"

def _quote(value: str) -> str:
    """A path or pattern as one shell word.

    `shlex.quote` and not an f-string with quotes around it: these are paths
    out of somebody else's trace, and one containing a quote of its own would
    otherwise translate into a command that means something different from the
    call it came from.
    """
    return shlex.quote(value.strip())


def _command_of(call: dict) -> str | None:
    """The shell command this Hermes call is, or None if the shell cannot do it.

    A translation and not a wrapper: `read_file` is `cat`, and what the model
    should learn is `cat`. The four translated here are the four the sandbox
    genuinely covers — `patch` has no binary in just-bash, `process` has no job
    control behind it, and a browser is not a shell at any level.
    """
    name = call.get("name")
    arguments = call.get("arguments")
    if not isinstance(arguments, dict):
        return None

    if name == "terminal":
        command = arguments.get("command")
        return command.strip() if isinstance(command, str) and command.strip() else None

    if name == "read_file":
        path = arguments.get("path")
        if not isinstance(path, str) or not path.strip():
            return None
        limit, offset = arguments.get("limit"), arguments.get("offset")
        if isinstance(offset, int) and isinstance(limit, int) and limit > 0:
            return f"sed -n {offset + 1},{offset + limit}p {_quote(path)}"
        if isinstance(limit, int) and limit > 0:
            return f"head -n {limit} {_quote(path)}"
        return f"cat {_quote(path)}"

    if name == "write_file":
        path, content = arguments.get("path"), arguments.get("content")
        if not isinstance(path, str) or not isinstance(content, str) or not path.strip():
            return None
        # A quoted heredoc, so nothing in the file body is expanded on the way
        # in. `EOF` unquoted would make `$HOME` in a config file mean something.
        return f"cat > {_quote(path)} <<'EOF'\n{content.rstrip()}\nEOF"

    if name == "search_files":
        pattern = arguments.get("pattern")
        if not isinstance(pattern, str) or not pattern.strip():
            return None
        where = arguments.get("path") if isinstance(arguments.get("path"), str) else "."
        if arguments.get("target") == "content":
            return f"rg {_quote(pattern)} {_quote(where)}"
        return f"find {_quote(where)} -name {_quote(pattern)}"

    return None


def _output_of(content: dict) -> tuple[str, int]:
    """`(output, exit code)` from a Hermes tool response.

    Each tool reports in its own shape — `terminal` has `output` and
    `exit_code`, `read_file` has `content`, `search_files` has `files` or
    `matches` — and all of them are being turned back into the one thing a
    shell produces: some text and a status.
    """
    exit_code = content.get("exit_code")
    exit_code = exit_code if isinstance(exit_code, int) else 0

    parts: list[str] = []
    for key in ("output", "content", "stdout"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value)

    for key in ("files", "matches"):
        value = content.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if str(item).strip())

    error = content.get("error")
    if isinstance(error, str) and error.strip():
        parts.append(error)
        # A tool that reported an error and no status of its own did fail, and
        # a shell that failed says so with a code. 1 is what `cat` on a missing
        # file returns, which is overwhelmingly the case being described.
        exit_code = exit_code or 1

    return "\n".join(part.rstrip() for part in parts).strip(), exit_code


def _truncated(text: str, limit: int) -> tuple[str, bool]:
    """The first `limit` characters, cut on a line boundary where there is one.

    The same rule as `bash.rs::clamp`, for the same reason: the head of a
    listing is as good as any part of it, and a model that needs the end can
    say `| tail`.
    """
    if len(text) <= limit:
        return text, False
    cut = text[:limit]
    at = cut.rfind("\n")
    if at > limit // 2:
        cut = cut[:at]
    return cut.rstrip(), True


def _result_body(output: str, exit_code: int, max_output_chars: int) -> str:
    """What a `bash` result looks like, as the live tool renders it.

    Mirrors `backend/crates/tools/src/bash.rs::render` deliberately and closely:
    this is training data for a model that will be served by that function, and
    a corpus whose results are laid out differently from the ones the server
    produces teaches a format the model then never sees.
    """
    body, cut = _truncated(output, max_output_chars)

    out = ""
    if exit_code != 0:
        out += f"{WARN_OPEN}The command exited {exit_code}.{WARN_CLOSE}"
    if not body:
        return out or f"{WARN_OPEN}The command finished with no output.{WARN_CLOSE}"

    out += f"{CODEBLOCK_OPEN}{neutralise(body)}{CODEBLOCK_CLOSE}"
    if cut:
        out += f"{WARN_OPEN}The output was longer and has been cut.{WARN_CLOSE}"
    return out


def _call_markup(command: str) -> str:
    """One call, rendered exactly as `llm.chat.tools.call` renders it.

    Control-stripped and not marker-neutralised: the argument is a command, and
    a `write_file` heredoc carries a file body that came out of the same decoded
    JSON as any result — but it is JSON-encoded into the payload here, so a
    marker inside it cannot end the call early the way one in prose could.
    """
    payload = json.dumps(
        {"tool": BASH, "argument": strip_control(command).strip()}, ensure_ascii=False
    )
    return f"{TOOL_CALL_OPEN}{payload}{TOOL_CALL_CLOSE}"


def _split_gpt(content: str) -> tuple[str, str, list[dict]]:
    """One `gpt` turn as `(reasoning, reply, calls)`, the text still raw.

    Everything the model wrote on one side and everything the harness was asked
    to do on the other: the `<think>` bodies, the turn with its thinking and
    calls cut out, and the calls themselves parsed.
    """
    reasoning, body = split_thinking(content)

    calls: list[dict] = []
    for match in _TOOL_CALL.finditer(body):
        try:
            parsed = json.loads(match.group(1).strip())
        except ValueError:
            continue
        if isinstance(parsed, dict):
            calls.append(parsed)

    return reasoning, _TOOL_CALL.sub("", body), calls


def _responses(content: str) -> list[dict]:
    """The responses in one `tool` turn, in the order the calls were made."""
    out: list[dict] = []
    for match in _TOOL_RESPONSE.finditer(content):
        try:
            parsed = json.loads(match.group(1).strip())
        except ValueError:
            continue
        if isinstance(parsed, dict):
            out.append(parsed)
    return out


def _answer_to(name: str, responses: list[dict]) -> dict | None:
    """The response answering the first call to `name`.

    Matched by tool name rather than by `tool_call_id`: the two configs number
    their ids differently (`functions.terminal:3` in one, `chatcmpl-tool-<hex>`
    in the other), and since only one call per turn is ever kept, the first
    response from that tool is the one that answers it.
    """
    for response in responses:
        if response.get("name") == name and isinstance(response.get("content"), dict):
            return response
    return None


def _assistant(run: list[str]) -> str:
    """A run of assistant turns with no surviving call between them, as one reply.

    Reasoning first, in one `<|thinking|>` block, then the prose in the order it
    was written. Not interleaved: a reply here opens with its thinking and never
    returns to it, which is what the template's `strip_thinking` and the
    grammar's budget markers are built around.
    """
    reasoning: list[str] = []
    replies: list[str] = []
    for content in run:
        thought, reply, _ = _split_gpt(content)
        if thought.strip():
            reasoning.append(thought)
        if reply.strip():
            replies.append(reply)

    return reply_markup("\n\n".join(reasoning), "\n\n".join(replies))


def _convert_hermes(doc: Document, params: dict, state: State) -> Document | None:
    turns = list(doc.get("messages") or [])
    messages: list[dict] = []
    run: list[str] = []

    def flush() -> None:
        """Close off the assistant turns waiting to be merged."""
        if not run:
            return
        rendered = _assistant(run)
        run.clear()
        if rendered:
            messages.append({"role": "assistant", "content": rendered})

    for index, turn in enumerate(turns):
        role = str(turn.get("role") or "").strip().lower()
        content = turn.get("content")
        if not isinstance(content, str):
            return None

        if role == "gpt":
            run.append(content)
            kept = _keep_one_call(content, _next_turn(turns, index), params)
            if kept is None:
                continue
            command, answer = kept
            # The call ends the run: everything before it is the reply that
            # made it, and everything after answers what came back.
            reply = _assistant(run)
            run.clear()
            messages.append({"role": "assistant", "content": reply + _call_markup(command)})
            messages.append({"role": "tool", "content": _call_markup(command) + answer})
            continue

        # A `tool` turn is dropped whatever happens: the response that mattered
        # was already taken by the call it answers, above. Dropping it without
        # ending the run is what merges the assistant turns on either side of a
        # call this project cannot make — with the call gone they are one
        # uninterrupted stretch of the model talking.
        if role == "tool":
            continue

        flush()

        if role == "system":
            if params["keep_tool_system"] or _TOOL_SYSTEM not in content:
                if content.strip():
                    messages.append({"role": "system", "content": neutralise(content)})
            continue

        if role == "human":
            if content.strip():
                messages.append({"role": "user", "content": neutralise(content)})
            continue

        # An unrecognised role drops the conversation rather than being guessed
        # at, the same as `chat._normalise_roles`.
        return None

    flush()

    # A question left hanging at the end teaches nothing and costs window.
    while messages and messages[-1]["role"] == "user":
        messages.pop()

    roles = {message["role"] for message in messages}
    if not {"user", "assistant"} <= roles:
        return None
    return {**doc, "messages": messages}


def _next_turn(turns: list[dict], index: int) -> dict | None:
    return turns[index + 1] if index + 1 < len(turns) else None


def _keep_one_call(content: str, following: dict | None, params: dict) -> tuple[str, str] | None:
    """The one call this turn keeps and the answer to it, or None for neither.

    The first call the shell can make, paired with its recorded output. A turn
    whose calls are all `browser_click` keeps nothing; a turn that made four
    searches keeps the first, because the runtime answers one call per round.
    A command longer than the budget is skipped rather than truncated — half a
    heredoc is a different command, and the file it writes is not the file the
    trace wrote.
    """
    _, _, calls = _split_gpt(content)

    responses: list[dict] = []
    if following is not None and str(following.get("role") or "").strip().lower() == "tool":
        responses = _responses(str(following.get("content") or ""))

    for call in calls:
        command = _command_of(call)
        if command is None or len(command) > params["max_command_chars"]:
            continue
        response = _answer_to(str(call.get("name") or ""), responses)
        if response is None:
            # A call nobody answered. Every kept call carries its result, and
            # one without would train the model to expect an answer that the
            # conversation then never contains.
            continue
        output, exit_code = _output_of(response["content"])
        return command, _result_body(output, exit_code, params["max_output_chars"])
    return None


PROCESSORS = (
    Processor(
        name="convert-hermes-agent",
        title="Convert Hermes agent traces",
        description=(
            "Shell-shaped calls (terminal, read_file, write_file, search_files) become "
            "<|tool-call|> bash calls with their recorded output; everything the sandbox "
            "cannot do is dropped with its result. Reasoning and prose are kept."
        ),
        applies_to=SFT,
        apply=_convert_hermes,
        params=(
            Param(
                name="keep_tool_system",
                label="Keep the tool-format system prompt",
                type="bool",
                default=False,
                description=(
                    "Keep Hermes' function-calling preamble and its <tools> inventory, "
                    "~10,000 characters describing tools this project does not have."
                ),
            ),
            Param(
                name="max_command_chars",
                label="Longest command to keep",
                type="int",
                default=400,
                minimum=40,
                maximum=4000,
                description="A longer call is dropped whole — a truncated command is a different one.",
            ),
            Param(
                name="max_output_chars",
                label="Characters of output per result",
                type="int",
                default=320,
                minimum=40,
                maximum=4000,
                description="Mirror of the backend's BASH_BUDGET_CHARS. Raise both together.",
            ),
        ),
    ),
)
