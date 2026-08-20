"""What each processor does to a document, and what a pipeline does to a stream."""

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


def test_convert_smoltalk_turns_a_tool_call_into_web_search():
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

    assert after["messages"][1]["content"] == "<|web-search|>When was the Eiffel Tower finished?<|/web-search|>"


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

    assert (
        after["messages"][1]["content"]
        == "<|web-search|>Retrieve the role IDs for document 654321.<|/web-search|>"
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

    assert after["messages"][1]["content"] == (
        "<|web-search|>two things<|/web-search|><|web-search|>two things<|/web-search|>"
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
        "Let me check.<|web-search|>a thing<|/web-search|>"
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
    content = "Answer in <|par|>a **b** c<|/par|> was searched via <|web-search|>q ** 2<|/web-search|>"

    after = _md([{"role": "assistant", "content": content}])

    assert (
        after["messages"][0]["content"]
        == "Answer in <|par|>a <|bold|>b<|/bold|> c<|/par|> was searched via "
        "<|web-search|>q ** 2<|/web-search|>"
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
