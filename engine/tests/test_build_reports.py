"""What phase 1 says while it is running.

Packing is the longest thing a run does before it trains — hours, at the scale
`docs/engine/corpus.md` quotes — and it used to report nothing at all between
"the tokenizer is built" and "here is the corpus". These are the assertions that
keep it talking, and keep what it says honest.
"""

import pytest

from llm.corpus import pack, vocab
from llm.train import progress
from llm.train.progress import BuildReport, Tokenize
from tests.fixtures.tiny_env import TINY_DATASET_ID


class _Clock:
    """A stand-in for `time`, so the throttle can be tested without waiting for
    it. Only `monotonic` is ever read."""

    def __init__(self, now: float = 1_000.0) -> None:
        self.now = now

    def monotonic(self) -> float:
        return self.now


@pytest.fixture
def clock(monkeypatch):
    fake = _Clock()
    monkeypatch.setattr(progress, "time", fake)
    return fake


# ── the throttle ────────────────────────────────────────────────────────────


def test_a_build_reports_at_once_and_then_waits(clock):
    """The first frame is what says the step has started, so it never waits; the
    ones after it are what a `TARGET_BUILD_REPORTS` budget is for."""
    frames: list[Tokenize] = []
    report = BuildReport(frames.append)

    report.at("packing")
    assert len(frames) == 1

    assert not report.tick(documents_done=1)
    clock.now += progress.BUILD_REPORT_SECS / 2
    assert not report.tick(documents_done=2)
    clock.now += progress.BUILD_REPORT_SECS
    assert report.tick(documents_done=3)

    assert [f.documents_done for f in frames] == [None, 3]


def test_the_interval_widens_with_the_build_it_is_reporting_on(clock):
    """A run's frame history is held in memory for as long as somebody is
    attached to it, so a four-hour pack must not cost four hours of seconds."""
    report = BuildReport(lambda _: None)
    report.at("packing")

    clock.now += 60 * 60
    assert report.due, "an hour with no frame is due one whatever the interval"
    report.say()

    clock.now += progress.BUILD_REPORT_SECS * 2
    assert not report.due, "an hour in, a frame every couple of seconds is spam"
    clock.now += 3_600 / progress.TARGET_BUILD_REPORTS
    assert report.due


# ── the bar ─────────────────────────────────────────────────────────────────


def test_the_bar_never_goes_backwards(clock):
    """Phase 1's steps report their own fractions, and the bar is shared. A step
    that starts at zero must not send it back there."""
    frames: list[Tokenize] = []
    report = BuildReport(frames.append)

    report.at("merges")
    report.say(fraction=1.0)
    report.at("packing")
    report.say(fraction=0.0)

    reached = [f.progress for f in frames]
    assert reached == sorted(reached)
    assert reached[-1] >= progress.BUILD_SPANS["packing"][0]


def test_a_step_reports_its_own_name_on_every_frame(clock):
    """A frame is merged into what the panel already holds, so one that left the
    step out would read as the previous step still running."""
    frames: list[Tokenize] = []
    report = BuildReport(frames.append)

    report.at("sampling")
    clock.now += progress.BUILD_REPORT_SECS
    report.tick(sampled_bytes=10)
    report.at("packing")

    assert [f.step for f in frames] == ["sampling", "sampling", "packing"]


def test_only_the_end_of_the_phase_says_it_is_over(clock):
    """The regression this whole seam exists for: `vocab.build` closed its own
    frame with `done`, which for a pretrain run is the first tenth of phase 1 —
    so the panel left the build view for the whole of the pack that followed."""
    frames: list[Tokenize] = []
    report = BuildReport(frames.append)

    report.at("sampling")
    report.say(fraction=1.0, vocab=512)
    assert not any(f.done for f in frames)

    report.finish(documents=3)
    assert frames[-1].done
    assert frames[-1].progress == 1.0


def test_a_rate_of_nothing_is_absent_rather_than_zero(clock):
    """`Tokenize`'s convention, applied to the numbers a build projects: an ETA
    off no measurement reads as "seconds away" where it means "nobody knows"."""
    report = BuildReport(lambda _: None)
    report.at("packing")

    assert report.pace(0, 100) == {"fraction": 0.0}
    assert "eta_secs" not in report.pace(0, 100)
    assert report.pace(0, 0) == {}

    clock.now += 10
    paced = report.pace(25, 100, 5_000)
    assert paced["eta_secs"] == pytest.approx(30.0)
    assert paced["tokens_per_sec"] == pytest.approx(500.0)


def test_a_build_projects_from_text_and_not_from_document_counts(clock):
    """The number that read as fourteen hours on a pack that took four.

    A document is not an amount of text: `gutenberg_eng-clean` is 92% of this
    selection's bytes and 8.8% of its documents, so a pack a quarter of the way
    through the books had read 2% of the documents it was asked for — and
    divided the hour it had spent by that.
    """
    frames: list[Tokenize] = []
    report = BuildReport(frames.append)
    report.at("packing")
    tally = pack._Tally(
        [
            {"dataset": "books", "stage": 0, "expected": 48_284,
             "expected_chars": 17_201_587_997},
            {"dataset": "snippets", "stage": 0, "expected": 500_000,
             "expected_chars": 1_390_416_820},
        ]
    )

    book = pack.SourceDoc(dataset="books", text="x" * 356_259)
    for _ in range(12_071):  # a quarter of the books, and none of the snippets
        tally.read(book)
    clock.now += 3_600
    tally.report(report, force=True)

    # Three quarters of the books and all of the snippets, at the rate the first
    # quarter went: three and a bit hours left.
    assert frames[-1].eta_secs == pytest.approx(11_964, rel=0.01)
    counted_by_document = 3_600 / 12_071 * (548_284 - 12_071)
    assert counted_by_document > 10 * frames[-1].eta_secs


def test_a_plan_that_states_no_text_is_still_paced_by_what_it_does_state(clock):
    """A script or a test hands over document counts alone. Projecting off a
    total nobody stated would be a fraction of nothing, so the documents are
    used — which is what every build did before text was in the plan."""
    frames: list[Tokenize] = []
    report = BuildReport(frames.append)
    report.at("packing")
    tally = pack._Tally([{"dataset": "d", "stage": 0, "expected": 100}])

    for _ in range(25):
        tally.read(pack.SourceDoc(dataset="d", text="four"))
    clock.now += 10
    tally.report(report, force=True)

    assert frames[-1].eta_secs == pytest.approx(30.0)


def test_the_eta_is_projected_from_the_step_and_not_from_the_phase(clock):
    """A pack that inherited the minute the vocabulary took would read as slower
    than it is until it had caught up with the lie."""
    report = BuildReport(lambda _: None)
    report.at("sampling")
    clock.now += 100

    report.at("packing")
    clock.now += 10
    assert report.pace(50, 100)["eta_secs"] == pytest.approx(10.0)


# ── what the pack reports ───────────────────────────────────────────────────


def _frames_of_a_pack(client, plan_extra: int = 0) -> list[Tokenize]:
    selection = vocab.resolve(client, [TINY_DATASET_ID])
    tokenizer_id = vocab.auto_id(selection, 512)
    vocab.build(client, tokenizer_id, selection, 512)
    fingerprint = vocab.fingerprint(tokenizer_id)
    key = pack.corpus_key(fingerprint, [(None, selection)])

    plan = [
        {
            "dataset": one.name,
            "stage": 0,
            "expected": one.taken,
            "expected_chars": one.weight,
        }
        for one in selection
    ] + [
        {"dataset": f"unread-{i}", "stage": 1, "expected": 1, "expected_chars": 1}
        for i in range(plan_extra)
    ]
    frames: list[Tokenize] = []
    pack.pack(
        (
            pack.SourceDoc(dataset=one.name, text=row["content"])
            for one in selection
            for row in client.documents(one.id, cap=one.cap)
        ),
        vocab.load(tokenizer_id),
        key=key,
        stages=[{"name": None, "epochs": 1, "datasets": [s.record() for s in selection]}],
        tokenizer_id=tokenizer_id,
        fingerprint=fingerprint,
        report=BuildReport(frames.append),
        plan=plan,
    )
    return frames


def test_a_pack_says_where_it_is_while_it_packs(data_root, data_client):
    """The whole point: a watcher can tell a pack that is working from one that
    has stopped, without waiting for it to finish to find out."""
    frames = _frames_of_a_pack(data_client)

    assert frames, "a pack that reports nothing is the bug this replaced"
    last = frames[-1]
    assert last.documents_done and last.documents_done <= last.documents
    assert last.total_tokens and last.chars
    assert last.tokens_per_sec is not None

    named = {row["dataset"] for row in last.sources}
    assert named, "the sources are what say which dataset a stalled pack is on"
    assert all(row["tokens"] <= last.total_tokens for row in last.sources)


def test_a_planned_source_is_on_the_strip_before_it_is_reached(data_root, data_client):
    """At zero, which is a fact about the build rather than a gap in it — a
    strip that grew as sources were reached could never show what is left."""
    frames = _frames_of_a_pack(data_client, plan_extra=2)

    unread = [row for row in frames[-1].sources if row["dataset"].startswith("unread-")]
    assert len(unread) == 2
    assert all(row["documents"] == 0 and row["expected"] == 1 for row in unread)


def test_the_strip_is_bounded_by_a_constant_and_says_so(data_root, data_client):
    """Same argument as `FERTILITY_DATASETS`, except this list goes out every
    second rather than once — so the cap bounds the whole pack's reporting."""
    extra = pack.BUILD_SOURCES + 4
    frames = _frames_of_a_pack(data_client, plan_extra=extra)

    for frame in frames:
        assert len(frame.sources) == pack.BUILD_SOURCES
        assert frame.sources_total == extra + 1


def test_the_pack_counts_the_same_things_it_did_before_it_reported(packed):
    """The counters the live frame is built from are the sidecar's own, so a
    build that reports nothing still records exactly what it used to."""
    _, stats = packed

    assert stats.chars > 0 and stats.corpus_len > 0
    assert stats.datasets == len(stats.fertility)
    assert sum(one["tokens"] for one in stats.fertility) <= stats.corpus_len
