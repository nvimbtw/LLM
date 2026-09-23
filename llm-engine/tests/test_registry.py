"""The live half of a run: its frames, and who is watching them."""

import asyncio
import threading

import pytest

from api.runs import Event, Registry


@pytest.fixture
async def registry():
    reg = Registry()
    reg.bind_loop(asyncio.get_running_loop())
    return reg


async def collect(registry, run, limit=10) -> list[Event]:
    """Read `limit` frames and close the stream — the close is what runs the
    generator's cleanup, exactly as a disconnecting client's does."""
    seen = []
    stream = registry.subscribe(run)
    try:
        async for event in stream:
            seen.append(event)
            if len(seen) >= limit:
                break
    finally:
        await stream.aclose()
    return seen


async def test_a_late_subscriber_replays_the_whole_run(registry):
    """A frame is only interesting while someone is attached — but attaching
    late must not mean missing what led here."""
    run = registry.adopt("r1", "train", {})
    registry.publish(run, Event("progress", {"step": 1}))
    registry.publish(run, Event("progress", {"step": 2}))
    registry.finish(run, "completed", result={"ok": True})

    seen = await collect(registry, run)

    assert [e.kind for e in seen] == ["progress", "progress", "done"]
    assert seen[-1].data == {"result": {"ok": True}}


async def test_a_subscriber_stops_at_the_terminal_frame(registry):
    run = registry.adopt("r2", "train", {})
    registry.finish(run, "failed", error="kaboom")

    seen = [e async for e in registry.subscribe(run)]

    assert [e.kind for e in seen] == ["error"]
    assert seen[0].data == {"error": "kaboom"}


async def test_a_cancelled_run_says_so_rather_than_failing(registry):
    run = registry.adopt("r3", "train", {})
    registry.finish(run, "cancelled", result={"stopped": True})

    seen = [e async for e in registry.subscribe(run)]
    assert [e.kind for e in seen] == ["cancelled"]


async def test_a_frame_published_mid_subscribe_is_neither_lost_nor_doubled(registry):
    """The backlog is copied under the lock that registers the queue, so a frame
    emitted between the two lands in exactly one of them."""
    run = registry.adopt("r4", "train", {})
    registry.publish(run, Event("progress", {"step": 1}))

    async def publish_then_finish():
        await asyncio.sleep(0)
        registry.publish(run, Event("progress", {"step": 2}))
        registry.finish(run, "completed", result=None)

    task = asyncio.create_task(publish_then_finish())
    seen = await collect(registry, run)
    await task

    assert [e.data.get("step") for e in seen if e.kind == "progress"] == [1, 2]


async def test_a_worker_thread_reaches_the_loop_the_registry_was_bound_to(registry):
    """Publishing from a thread without the binding is the class of bug that
    shows up as a hang rather than an exception."""
    run = registry.adopt("r5", "train", {})

    def worker():
        registry.publish(run, Event("progress", {"from": "thread"}))
        registry.finish(run, "completed", result=None)

    thread = threading.Thread(target=worker)
    thread.start()
    seen = await collect(registry, run)
    thread.join()

    assert seen[0].data == {"from": "thread"}


async def test_the_active_run_is_the_one_that_has_not_finished(registry):
    first = registry.adopt("r6", "train", {})
    assert registry.active() is first

    registry.finish(first, "completed", result=None)
    assert registry.active() is None

    second = registry.adopt("r7", "benchmark", {})
    assert registry.active() is second


async def test_an_unknown_run_has_no_live_state(registry):
    assert registry.get("nobody") is None


async def test_publishing_with_no_loop_bound_still_records_the_frame():
    """A run started before the app's lifespan bound the loop keeps its history;
    only the wake-up is skipped."""
    reg = Registry()
    run = reg.adopt("r8", "train", {})
    reg.publish(run, Event("progress", {"step": 1}))

    assert [e.kind for e in run.history] == ["progress"]


async def test_a_subscriber_that_walks_away_is_forgotten(registry):
    """A run outlives the tab watching it, so a dropped viewer must not leave a
    queue behind for every frame to be pushed onto."""
    run = registry.adopt("r9", "train", {})
    registry.publish(run, Event("progress", {"step": 1}))

    await collect(registry, run, limit=1)

    assert run.subscribers == set()
