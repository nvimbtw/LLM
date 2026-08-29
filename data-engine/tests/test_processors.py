"""What each processor does to a document, and what a pipeline does to a stream."""

import json

import pytest

import processors
from store.schema import DatasetKind


def apply(name: str, doc: dict, params: dict | None = None, state: dict | None = None):
    steps = processors.resolve([{"processor": name, "params": params or {}}], processors.REGISTRY)
    return processors.run(steps, doc, state if state is not None else {}).after


def test_strip_html_removes_tags_and_script_bodies():
    doc = {"content": "<p>keep</p><script>var secret = 1;</script>"}

    after = apply("strip-html", doc)

    assert "keep" in (after or {}).get("content", "")
    assert "var secret" not in (after or {}).get("content", "")


def test_collapse_whitespace_squeezes_runs_and_trims():
    after = apply("collapse-whitespace", {"content": "a  b\r\n\n\n\nc   \n"})

    assert after["content"] == "a b\n\nc"


def test_join_broken_lines_drops_every_blank_line():
    after = apply("join-broken-lines", {"content": "line1\n\nline2\n\nline3"})

    assert after["content"] == "line1\nline2\nline3"


def test_join_broken_lines_handles_runs_of_blank_lines_and_crlf():
    after = apply("join-broken-lines", {"content": "line1\r\n\r\n\r\nline2\n   \nline3"})

    assert after["content"] == "line1\nline2\nline3"


def test_min_length_drops_below_the_bound_and_keeps_at_it():
    assert apply("min-length", {"content": "abcd"}, {"characters": 5}) is None
    assert apply("min-length", {"content": "abcde"}, {"characters": 5}) is not None


def test_max_length_truncates_prose_only_when_asked():
    long = {"content": "x" * 100}

    assert apply("max-length", long, {"characters": 10}) is None
    assert apply("max-length", long, {"characters": 10, "truncate": True})["content"] == "x" * 10


def test_dedup_keeps_the_first_copy_and_drops_the_rest():
    state: dict = {}
    first = apply("dedup-exact", {"content": "same"}, state=state)
    second = apply("dedup-exact", {"content": "same"}, state=state)

    assert first is not None
    assert second is None


def test_dedup_state_does_not_leak_between_runs():
    assert apply("dedup-exact", {"content": "same"}) is not None
    assert apply("dedup-exact", {"content": "same"}) is not None


def test_redact_replaces_identifiers_it_was_asked_for():
    doc = {"content": "write to a@b.com from 10.0.0.1"}

    after = apply("redact-pii", doc, {"emails": True, "ip_addresses": True})

    assert "a@b.com" not in after["content"]
    assert "10.0.0.1" not in after["content"]


def test_normalise_roles_maps_sharegpt_names():
    doc = {"messages": [{"role": "human", "content": "hi"}, {"role": "gpt", "content": "hello"}]}

    after = apply("normalise-roles", doc)

    assert [m["role"] for m in after["messages"]] == ["user", "assistant"]


def test_an_unrecognised_role_drops_the_conversation():
    doc = {"messages": [{"role": "narrator", "content": "hi"}]}

    assert apply("normalise-roles", doc) is None


def test_require_roles_rejects_a_one_sided_conversation():
    doc = {"messages": [{"role": "user", "content": "hi"}]}

    assert apply("require-roles", doc) is None


def test_length_of_a_conversation_counts_its_turns():
    doc = {"messages": [{"role": "user", "content": "abc"}, {"role": "assistant", "content": "de"}]}

    # "abc" + "\n" + "de" is six characters, so a bound of seven drops it.
    assert apply("min-length", doc, {"characters": 6}) is not None
    assert apply("min-length", doc, {"characters": 7}) is None


def test_params_are_clamped_rather_than_rejected():
    """A slider mid-drag must not error, so an out-of-range value is pulled to
    the nearest legal one."""
    processor = processors.REGISTRY["min-length"]

    assert processor.resolve({"characters": -5})["characters"] == 0
    assert processor.resolve({"characters": "nonsense"})["characters"] == 200


def test_an_unknown_processor_is_refused_not_skipped():
    with pytest.raises(KeyError):
        processors.resolve([{"processor": "invent-data"}], processors.REGISTRY)


def test_order_matters_stripping_before_measuring():
    """A page of markup should not pass a length bound it only met in tags."""
    doc = {"content": "<div class='a very long attribute'></div>"}

    stripped_first = processors.run(
        processors.resolve(
            [
                {"processor": "strip-html"},
                {"processor": "collapse-whitespace"},
                {"processor": "min-length", "params": {"characters": 20}},
            ],
            processors.REGISTRY,
        ),
        doc,
        {},
    )
    measured_first = processors.run(
        processors.resolve(
            [{"processor": "min-length", "params": {"characters": 20}}], processors.REGISTRY
        ),
        doc,
        {},
    )

    assert stripped_first.dropped
    assert not measured_first.dropped


def test_each_kind_is_offered_only_what_applies():
    pretrain = {p.name for p in processors.for_kind(DatasetKind.pretrain)}
    sft = {p.name for p in processors.for_kind(DatasetKind.sft)}

    assert "strip-html" in pretrain and "strip-html" not in sft
    assert "normalise-roles" in sft and "normalise-roles" not in pretrain
    assert "min-length" in pretrain and "min-length" in sft


SMOLTALK = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}


def test_convert_smoltalk_wraps_assistant_replies_in_par():
    after = apply("convert-smoltalk", SMOLTALK)

    assert after["messages"][1]["content"] == "<|par|>hello<|/par|>"
    assert after["messages"][1]["role"] == "assistant"


def test_convert_smoltalk_keeps_the_system_message():
    doc = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            *SMOLTALK["messages"],
        ]
    }

    after = apply("convert-smoltalk", doc)

    assert after["messages"][0] == {"role": "system", "content": "You are a concise assistant."}


def test_convert_smoltalk_turns_a_tool_call_into_a_web_search_call():
    doc = {
        "messages": [
            {"role": "user", "content": "When was the Eiffel Tower finished?"},
            {
                "role": "assistant",
                "content": (
                    '<tool_call>[{"name": "get_year", "arguments": {"thing": "eiffel tower"}}]'
                    "</tool_call>"
                ),
            },
        ]
    }

    after = apply("convert-smoltalk", doc)

    assert after["messages"][1]["content"] == (
        '<|tool-call|>{"tool": "web-search", "argument": '
        '"When was the Eiffel Tower finished?"}<|/tool-call|>'
    )


def test_convert_smoltalk_queries_what_the_user_asked_not_json_arguments():
    doc = {
        "messages": [
            {"role": "user", "content": "Retrieve the role IDs for document 654321."},
            {
                "role": "assistant",
                "content": (
                    '<tool_call>[{"name": "get_role_ids", "arguments": '
                    '{"document_id": "654321"}}]</tool_call>'
                ),
            },
        ]
    }

    after = apply("convert-smoltalk", doc)

    assert after["messages"][1]["content"] == (
        '<|tool-call|>{"tool": "web-search", "argument": '
        '"Retrieve the role IDs for document 654321."}<|/tool-call|>'
    )
    assert '"document_id"' not in after["messages"][1]["content"]


def test_convert_smoltalk_renders_each_call_in_a_list():
    doc = {
        "messages": [
            {"role": "user", "content": "two things"},
            {
                "role": "assistant",
                "content": (
                    '<tool_call>[{"name": "a", "arguments": {"x": 1}}, '
                    '{"name": "b", "arguments": {"y": 2}}]</tool_call>'
                ),
            },
        ]
    }

    after = apply("convert-smoltalk", doc)

    call = '<|tool-call|>{"tool": "web-search", "argument": "two things"}<|/tool-call|>'
    assert after["messages"][1]["content"] == (
        call + call
    )


def test_convert_smoltalk_leaves_prose_around_a_call_bare():
    doc = {
        "messages": [
            {"role": "user", "content": "a thing"},
            {
                "role": "assistant",
                "content": (
                    "Let me check.<tool_call>[{\"name\": \"lookup\", "
                    '"arguments": {"q": "snow"}}]</tool_call>'
                ),
            },
        ]
    }

    after = apply("convert-smoltalk", doc)

    assert after["messages"][1]["content"] == (
        'Let me check.<|tool-call|>{"tool": "web-search", "argument": "a thing"}<|/tool-call|>'
    )


def test_convert_smoltalk_drops_tool_format_system_prompts_by_default():
    doc = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are an expert in composing functions. You have access to the "
                    "following tools:\n<tools>[{\"name\": \"get_role_ids\"}]</tools>"
                ),
            },
            *SMOLTALK["messages"],
        ]
    }

    after = apply("convert-smoltalk", doc)

    assert all(m["role"] != "system" for m in after["messages"])


def test_convert_smoltalk_keeps_tool_system_prompts_when_asked():
    doc = {
        "messages": [
            {"role": "system", "content": "<tools>[{\"name\": \"x\"}]</tools>"},
            *SMOLTALK["messages"],
        ]
    }

    after = apply("convert-smoltalk", doc, {"keep_tool_system": True})

    assert after["messages"][0]["role"] == "system"


def test_convert_smoltalk_drops_on_malformed_tool_call():
    doc = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "<tool_call>[not json</tool_call>"},
        ]
    }

    assert apply("convert-smoltalk", doc) is None


def test_convert_smoltalk_does_not_wrap_when_asked():
    after = apply("convert-smoltalk", SMOLTALK, {"wrap_replies": False})

    assert after["messages"][1]["content"] == "hello"


def _md(messages):
    return apply("markdown-emphasis", {"messages": messages})


def test_markdown_emphasis_converts_double_star_bold():
    after = _md([{"role": "assistant", "content": "This is **important** text."}])

    assert after["messages"][0]["content"] == "This is <|bold|>important<|/bold|> text."


def test_markdown_emphasis_converts_single_star_italic():
    after = _md([{"role": "assistant", "content": "This is *emphasised* text."}])

    assert after["messages"][0]["content"] == "This is <|italic|>emphasised<|/italic|> text."


def test_markdown_emphasis_converts_triple_star_bold_italic():
    after = _md([{"role": "assistant", "content": "A ***very strong*** claim."}])

    assert (
        after["messages"][0]["content"]
        == "A <|bold|><|italic|>very strong<|/italic|><|/bold|> claim."
    )


def test_markdown_emphasis_handles_nesting_inside_bold():
    after = _md([{"role": "assistant", "content": "**outer *inner* text**"}])

    assert (
        after["messages"][0]["content"]
        == "<|bold|>outer <|italic|>inner<|/italic|> text<|/bold|>"
    )


def test_markdown_emphasis_ignores_underscores_inside_words_and_math():
    content = "snake_case stays, 2 * 3 stays, a * b stays, base ** 2 stays"

    after = _md([{"role": "assistant", "content": content}])

    assert after["messages"][0]["content"] == content


def test_markdown_emphasis_leaves_code_fences_and_inline_code_alone():
    content = "Run `x * y ** 2` in ```python\ndef f(): return a * b\n``` ok"

    after = _md([{"role": "assistant", "content": content}])

    assert after["messages"][0]["content"] == content


def test_markdown_emphasis_does_not_touch_existing_markers():
    content = (
        "Answer in <|par|>a **b** c<|/par|> was searched via "
        '<|tool-call|>{"tool": "web-search", "argument": "q ** 2"}<|/tool-call|>'
    )

    after = _md([{"role": "assistant", "content": content}])

    assert after["messages"][0]["content"] == (
        "Answer in <|par|>a <|bold|>b<|/bold|> c<|/par|> was searched via "
        '<|tool-call|>{"tool": "web-search", "argument": "q ** 2"}<|/tool-call|>'
    )


def test_markdown_emphasis_only_touches_assistant_turns():
    doc = {
        "messages": [
            {"role": "user", "content": "what does **bold** mean?"},
            {"role": "assistant", "content": "It means **bold**."},
        ]
    }

    after = apply("markdown-emphasis", doc)

    assert after["messages"][0]["content"] == "what does **bold** mean?"
    assert after["messages"][1]["content"] == "It means <|bold|>bold<|/bold|>."


def test_markdown_emphasis_following_convert_smoltalk_is_stable():
    doc = {
        "messages": [
            {"role": "user", "content": "Explain **this**."},
            {"role": "assistant", "content": "Here it is: **bold** and *italic*."},
        ]
    }
    sequence = processors.resolve(
        [{"processor": "convert-smoltalk"}, {"processor": "markdown-emphasis"}],
        processors.REGISTRY,
    )
    once = processors.run(sequence, doc, {}).after

    assert "**" not in once["messages"][1]["content"]
    assert "<|bold|>bold<|/bold|>" in once["messages"][1]["content"]
    # The emphasis step alone is idempotent; re-running the whole pipeline
    # re-wraps <|par|> and is not expected to be.
    again = apply("markdown-emphasis", once)

    assert again == once


HERMES_SYSTEM = (
    "You are a function calling AI model. Here are the available tools:\n"
    '<tools>[{"type": "function", "function": {"name": "terminal"}}]</tools>\n'
    "Each function call should be enclosed within <tool_call> </tool_call> XML tags."
)


def _hermes_turn(name: str, arguments: dict) -> str:
    return f'<tool_call>\n{json.dumps({"name": name, "arguments": arguments})}\n</tool_call>'


def _hermes_response(name: str, content: dict) -> str:
    body = {"tool_call_id": f"functions.{name}:0", "name": name, "content": content}
    return f"<tool_response>\n{json.dumps(body)}\n</tool_response>"


HERMES = {
    "messages": [
        {"role": "system", "content": HERMES_SYSTEM},
        {"role": "human", "content": "List the yaml files."},
        {
            "role": "gpt",
            "content": (
                "<think>\nThey want the yaml files. I should run ls.\n</think>\n"
                "Let me look:\n" + _hermes_turn("terminal", {"command": "ls *.yaml"})
            ),
        },
        {
            "role": "tool",
            "content": _hermes_response(
                "terminal", {"output": "deploy.yaml\n", "exit_code": 0, "error": ""}
            ),
        },
        {"role": "gpt", "content": "<think>\nOne file.\n</think>\nThere is one: deploy.yaml."},
    ]
}


def _bash_call(command: str) -> str:
    return f'<|tool-call|>{json.dumps({"tool": "bash", "argument": command})}<|/tool-call|>'


def test_convert_hermes_keeps_a_terminal_call_as_a_bash_call():
    after = apply("convert-hermes-agent", HERMES)

    assert [m["role"] for m in after["messages"]] == ["user", "assistant", "tool", "assistant"]
    assert after["messages"][1]["content"].endswith(_bash_call("ls *.yaml"))
    assert after["messages"][2]["content"] == (
        _bash_call("ls *.yaml") + "<|codeblock|>deploy.yaml<|/codeblock|>"
    )


def test_convert_hermes_translates_a_file_read_into_the_command_for_it():
    """`read_file` is `cat`, and `cat` is the name the model already knows."""
    doc = {
        "messages": [
            {"role": "human", "content": "what is in it?"},
            {"role": "gpt", "content": _hermes_turn("read_file", {"path": "a b.txt"})},
            {
                "role": "tool",
                "content": _hermes_response("read_file", {"content": "hello", "total_lines": 1}),
            },
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert after["messages"][1]["content"] == _bash_call("cat 'a b.txt'")
    assert after["messages"][2]["content"].endswith("<|codeblock|>hello<|/codeblock|>")


def test_convert_hermes_writes_a_file_with_a_quoted_heredoc():
    """Unquoted, `$HOME` in a config file would be expanded on the way in."""
    doc = {
        "messages": [
            {"role": "human", "content": "write it"},
            {
                "role": "gpt",
                "content": _hermes_turn("write_file", {"path": "run.sh", "content": "echo $HOME"}),
            },
            {
                "role": "tool",
                "content": _hermes_response("write_file", {"bytes_written": 11, "dirs_created": 0}),
            },
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert after["messages"][1]["content"] == _bash_call("cat > run.sh <<'EOF'\necho $HOME\nEOF")
    # Nothing on stdout, which is what a shell says when a write worked.
    assert after["messages"][2]["content"].endswith(
        "<|warn|>The command finished with no output.<|/warn|>"
    )


def test_convert_hermes_searches_for_files_or_in_them_depending_on_the_target():
    for target, command in (
        ("files", "find . -name '*.py'"),
        ("content", "rg '*.py' ."),
    ):
        doc = {
            "messages": [
                {"role": "human", "content": "find it"},
                {
                    "role": "gpt",
                    "content": _hermes_turn(
                        "search_files", {"pattern": "*.py", "target": target, "path": "."}
                    ),
                },
                {
                    "role": "tool",
                    "content": _hermes_response("search_files", {"files": ["a.py"], "total_count": 1}),
                },
            ]
        }

        after = apply("convert-hermes-agent", doc)

        assert after["messages"][1]["content"] == _bash_call(command)


def test_convert_hermes_drops_a_call_the_sandbox_cannot_make_with_its_result():
    """A browser is not a shell at any level, and just-bash has no `patch`."""
    doc = {
        "messages": [
            {"role": "human", "content": "go"},
            {
                "role": "gpt",
                "content": "<think>\nlook at the page\n</think>\nOpening it:"
                + _hermes_turn("browser_navigate", {"url": "https://github.com/trending"}),
            },
            {
                "role": "tool",
                "content": _hermes_response("browser_navigate", {"success": True, "title": "x"}),
            },
            {"role": "gpt", "content": "Done."},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert [m["role"] for m in after["messages"]] == ["user", "assistant"]
    assert "tool-call" not in after["messages"][1]["content"]
    # The turns either side of the dropped call are one stretch of the model
    # talking once the call between them is gone.
    assert after["messages"][1]["content"] == (
        "<|thinking|><|par|>look at the page<|/par|><|/thinking|>"
        "<|par|>Opening it:<|/par|><|par|>Done.<|/par|>"
    )


def test_convert_hermes_keeps_one_call_per_turn_because_the_runtime_answers_one():
    doc = {
        "messages": [
            {"role": "human", "content": "look everywhere"},
            {
                "role": "gpt",
                "content": _hermes_turn("terminal", {"command": "ls"})
                + _hermes_turn("terminal", {"command": "pwd"}),
            },
            {
                "role": "tool",
                "content": _hermes_response("terminal", {"output": "a", "exit_code": 0})
                + _hermes_response("terminal", {"output": "/tmp", "exit_code": 0}),
            },
        ]
    }

    after = apply("convert-hermes-agent", doc)

    calls = [m for m in after["messages"] if "<|tool-call|>" in m["content"]]
    assert all(m["content"].count("<|tool-call|>") == 1 for m in calls)
    assert after["messages"][1]["content"] == _bash_call("ls")
    assert "pwd" not in json.dumps(after["messages"])


def test_convert_hermes_names_a_failure_before_it_shows_the_output():
    doc = {
        "messages": [
            {"role": "human", "content": "run it"},
            {"role": "gpt", "content": _hermes_turn("terminal", {"command": "cat nope"})},
            {
                "role": "tool",
                "content": _hermes_response(
                    "terminal", {"output": "", "exit_code": 1, "error": "cat: nope: not found"}
                ),
            },
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert after["messages"][2]["content"].endswith(
        "<|warn|>The command exited 1.<|/warn|>"
        "<|codeblock|>cat: nope: not found<|/codeblock|>"
    )


def test_convert_hermes_cuts_output_the_model_has_no_room_for():
    listing = "".join(f"file-{i}.txt\n" for i in range(200))
    doc = {
        "messages": [
            {"role": "human", "content": "list"},
            {"role": "gpt", "content": _hermes_turn("terminal", {"command": "ls"})},
            {
                "role": "tool",
                "content": _hermes_response("terminal", {"output": listing, "exit_code": 0}),
            },
        ]
    }

    after = apply("convert-hermes-agent", doc, {"max_output_chars": 60})
    body = after["messages"][2]["content"]

    assert "file-0.txt" in body and "file-199.txt" not in body
    assert body.endswith("<|warn|>The output was longer and has been cut.<|/warn|>")


def test_convert_hermes_drops_a_command_too_long_to_keep_rather_than_cutting_it():
    """Half a heredoc writes a different file than the one the trace wrote."""
    doc = {
        "messages": [
            {"role": "human", "content": "write it"},
            {
                "role": "gpt",
                "content": _hermes_turn("write_file", {"path": "big.py", "content": "x" * 5000}),
            },
            {
                "role": "tool",
                "content": _hermes_response("write_file", {"bytes_written": 5000}),
            },
            {"role": "gpt", "content": "Written."},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert "<|tool-call|>" not in json.dumps(after["messages"])
    assert after["messages"][1]["content"] == "<|par|>Written.<|/par|>"


def test_convert_hermes_strips_control_bytes_a_recorded_result_carried():
    """`cat` on a binary file, which this corpus really does.

    The escape is inert in the corpus and inert in the source dataset; it only
    becomes a real NUL when `json.loads` reads the response here, and Postgres
    `jsonb` refuses to store one — the insert takes the whole batch and the run
    down with it, hundreds of documents from the row that carried it.
    """
    doc = {
        "messages": [
            {"role": "human", "content": "read the index"},
            {"role": "gpt", "content": _hermes_turn("terminal", {"command": "cat .git/index"})},
            {
                "role": "tool",
                "content": _hermes_response(
                    "terminal", {"output": "DIRC\x00\x01Test User\x07", "exit_code": 0}
                ),
            },
        ]
    }

    after = apply("convert-hermes-agent", doc)
    rendered = json.dumps(after["messages"])

    assert not any(c in rendered for c in ("\x00", "\x01", "\x07"))
    assert after["messages"][2]["content"].endswith("<|codeblock|>DIRCTest User<|/codeblock|>")


def test_convert_hermes_strips_control_bytes_from_a_command_too():
    """A `write_file` body reaches the heredoc without passing through
    `_neutralise`, so it needs the same guard on its own way out."""
    doc = {
        "messages": [
            {"role": "human", "content": "write it"},
            {
                "role": "gpt",
                "content": _hermes_turn(
                    "write_file", {"path": "blob.bin", "content": "head\x00tail"}
                ),
            },
            {"role": "tool", "content": _hermes_response("write_file", {"bytes_written": 9})},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert "\x00" not in json.dumps(after["messages"])
    assert after["messages"][1]["content"] == _bash_call("cat > blob.bin <<'EOF'\nheadtail\nEOF")


def test_convert_hermes_drops_a_call_nobody_answered():
    """A call with no result would teach the model to expect an answer that the
    conversation never contains."""
    doc = {
        "messages": [
            {"role": "human", "content": "go"},
            {"role": "gpt", "content": "Trying:" + _hermes_turn("terminal", {"command": "ls"})},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert after["messages"][1]["content"] == "<|par|>Trying:<|/par|>"


def test_convert_hermes_drops_the_tool_format_system_prompt_by_default():
    after = apply("convert-hermes-agent", HERMES)

    assert all(m["role"] != "system" for m in after["messages"])


def test_convert_hermes_keeps_the_tool_system_prompt_when_asked():
    after = apply("convert-hermes-agent", HERMES, {"keep_tool_system": True})

    assert after["messages"][0] == {"role": "system", "content": HERMES_SYSTEM}


def test_convert_hermes_keeps_a_system_prompt_that_is_not_a_tool_inventory():
    doc = {
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "human", "content": "hi"},
            {"role": "gpt", "content": "hello"},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert after["messages"][0] == {"role": "system", "content": "You are a concise assistant."}


def test_convert_hermes_keeps_fenced_code_verbatim():
    doc = {
        "messages": [
            {"role": "human", "content": "show me"},
            {
                "role": "gpt",
                "content": "Here:\n\n```python\ndef f():\n\n    return 1\n```\n\nThat is it.",
            },
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert after["messages"][1]["content"] == (
        "<|par|>Here:<|/par|>"
        "<|codeblock|>def f():\n\n    return 1<|/codeblock|>"
        "<|par|>That is it.<|/par|>"
    )


def test_convert_hermes_neutralises_markers_in_quoted_text():
    doc = {
        "messages": [
            {"role": "human", "content": "what is <|endoftext|>?"},
            {"role": "gpt", "content": "A separator: <|endoftext|>."},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert "<|endoftext|>" not in after["messages"][0]["content"]
    assert "<|endoftext|>" not in after["messages"][1]["content"]


def test_convert_hermes_closes_an_unterminated_block_at_the_turn():
    doc = {
        "messages": [
            {"role": "human", "content": "go"},
            {"role": "gpt", "content": "<think>\nstill thinking"},
            {"role": "gpt", "content": '<tool_call>{"name": "terminal"'},
            {"role": "gpt", "content": "Done."},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert after["messages"][1]["content"] == (
        "<|thinking|><|par|>still thinking<|/par|><|/thinking|><|par|>Done.<|/par|>"
    )


def test_convert_hermes_drops_a_trailing_question_nobody_answered():
    doc = {
        "messages": [
            {"role": "human", "content": "go"},
            {"role": "gpt", "content": "Done."},
            {"role": "human", "content": "and now?"},
        ]
    }

    after = apply("convert-hermes-agent", doc)

    assert [m["role"] for m in after["messages"]] == ["user", "assistant"]


def test_convert_hermes_drops_a_conversation_that_is_only_tool_noise():
    doc = {
        "messages": [
            {"role": "human", "content": "go"},
            {"role": "gpt", "content": _hermes_turn("browser_close", {})},
            {"role": "tool", "content": _hermes_response("browser_close", {"success": True})},
        ]
    }

    assert apply("convert-hermes-agent", doc) is None


def test_convert_chat_renders_a_reply_as_paragraphs_and_leaves_the_question_alone():
    doc = {
        "messages": [
            {"role": "user", "content": "What are two things to do in London?"},
            {"role": "assistant", "content": "Sure!\n\n1. The British Museum.\n\n2. Borough Market."},
        ]
    }

    after = apply("convert-chat", doc)

    assert after["messages"][0]["content"] == "What are two things to do in London?"
    assert after["messages"][1]["content"] == (
        "<|par|>Sure!<|/par|><|par|>1. The British Museum.<|/par|><|par|>2. Borough Market.<|/par|>"
    )


def test_convert_chat_turns_source_reasoning_into_a_thinking_block():
    doc = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>Add them.</think>It is 4."},
        ]
    }

    after = apply("convert-chat", doc)

    assert after["messages"][1]["content"] == (
        "<|thinking|><|par|>Add them.<|/par|><|/thinking|><|par|>It is 4.<|/par|>"
    )


def test_convert_chat_discards_reasoning_when_asked_and_keeps_the_reply():
    doc = {
        "messages": [
            {"role": "user", "content": "2+2?"},
            {"role": "assistant", "content": "<think>Add them.</think>It is 4."},
        ]
    }

    after = apply("convert-chat", doc, {"reasoning": "strip"})

    assert after["messages"][1]["content"] == "<|par|>It is 4.<|/par|>"


def test_convert_chat_keeps_fenced_code_out_of_a_paragraph():
    doc = {
        "messages": [
            {"role": "user", "content": "show me"},
            {"role": "assistant", "content": "Like so:\n\n```py\nx = 1\n\ny = 2\n```"},
        ]
    }

    after = apply("convert-chat", doc)

    assert "<|codeblock|>x = 1\n\ny = 2<|/codeblock|>" in after["messages"][1]["content"]


def test_convert_chat_neutralises_a_marker_the_source_wrote():
    doc = {
        "messages": [
            {"role": "user", "content": "what does <|endoftext|> mean?"},
            {"role": "assistant", "content": "It ends a document."},
        ]
    }

    after = apply("convert-chat", doc)

    assert "<|endoftext|>" not in after["messages"][0]["content"]
    assert "endoftext" in after["messages"][0]["content"]


def test_convert_chat_drops_a_trailing_question_nobody_answered():
    doc = {
        "messages": [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "Hi."},
            {"role": "user", "content": "and?"},
        ]
    }

    after = apply("convert-chat", doc)

    assert [m["role"] for m in after["messages"]] == ["user", "assistant"]


def test_convert_chat_refuses_a_conversation_that_uses_tools():
    doc = {
        "messages": [
            {"role": "user", "content": "search it"},
            {"role": "assistant", "content": "ok"},
            {"role": "tool", "content": "result"},
        ]
    }

    assert apply("convert-chat", doc) is None


def test_convert_chat_refuses_roles_normalise_roles_has_not_mapped_yet():
    doc = {"messages": [{"role": "human", "content": "hi"}, {"role": "gpt", "content": "hello"}]}

    assert apply("convert-chat", doc) is None


def test_drop_tool_calls_removes_a_conversation_in_either_spelling():
    source = {"messages": [{"role": "assistant", "content": "<tool_call>{}</tool_call>"}]}
    converted = {"messages": [{"role": "assistant", "content": '<|tool-call|>{"tool": "bash"}<|/tool-call|>'}]}

    assert apply("drop-tool-calls", source) is None
    assert apply("drop-tool-calls", converted) is None


def test_drop_tool_calls_keeps_a_conversation_that_only_talks_about_tools():
    doc = {"messages": [{"role": "assistant", "content": "A shell is a tool for running commands."}]}

    assert apply("drop-tool-calls", doc) is not None
