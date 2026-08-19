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
