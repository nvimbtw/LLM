"""Public projections keep administrative source, graphs and diagnostics out of generic reads."""

import math
from collections.abc import AsyncIterator

from api.runs import Event, Registry, Run

PUBLIC_METRICS = frozenset({
    "fraction", "step", "steps", "total_steps", "epoch", "epochs", "loss", "lr",
    "tokens", "tokens_per_sec", "elapsed", "eta", "completed", "total", "batch",
    "batches", "accuracy", "reward", "pass_rate", "iteration", "iterations",
})


def public_run(value: dict) -> dict:
    """Return queue identity/progress without arbitrary configuration, result, stage or exception text."""
    return {
        **value, "config": {}, "result": None, "stage": None,
        "error": "Run failed" if value.get("error") else None,
    }


def public_model(value: dict) -> dict:
    """Keep resolved model facts while removing arbitrary saved configuration and provenance blobs."""
    return {
        **{key: value.get(key) for key in (
            "model_id", "name", "description", "family", "context", "architecture",
            "vocab_size", "parameter_count", "definition_id", "context_extension_supported",
            "size", "published", "created_at", "has_weights", "checkpoint_step", "checkpoint_index",
        )},
        "config": {}, "parameters": None, "run_id": None, "parent_id": None,
    }


def public_event(event: Event) -> Event | None:
    """Expose finite known telemetry and generic terminal outcomes; omit logs and unknown events."""
    if event.kind == "error":
        return Event("error", {"error": "Run failed"})
    if event.kind in {"done", "cancelled"}:
        return Event(event.kind, {"result": None})
    if event.kind not in {"train", "progress", "rl", "tokenize"}:
        return None
    metrics = {
        key: value for key, value in event.data.items()
        if key in PUBLIC_METRICS and type(value) in {int, float} and math.isfinite(value)
    }
    return Event(event.kind, metrics)


async def visible_events(registry: Registry, run: Run, administration: bool) -> AsyncIterator[Event]:
    """Apply visibility to replay and live delivery without changing durable cancellation ownership."""
    async for event in registry.subscribe(run):
        projected = event if administration else public_event(event)
        if projected is not None:
            yield projected
