"""Resolve the catalogue once; sample sources before task types, once per group."""

import hashlib
import random
from dataclasses import dataclass

from gyms.registry import GymRegistry

VERSION = 1


@dataclass(frozen=True)
class Plan:
    task_types: tuple[str, ...]
    manifest: dict

    def choose(self, seed_base: int, group: int) -> str:
        if self.manifest["version"] != VERSION:
            raise ValueError("unsupported selection version")
        if self.manifest["mode"] == "cycle":
            return self.task_types[group % len(self.task_types)]
        digest = hashlib.sha256(f"general:{self.manifest['version']}:{seed_base}:{group}".encode()).digest()
        rng = random.Random(int.from_bytes(digest, "big"))
        sources = self.manifest["sources"]
        source = rng.choices(sources, weights=[s["weight"] for s in sources], k=1)[0]
        types = source["task_types"]
        return rng.choices([t["id"] for t in types], weights=[t["weight"] for t in types], k=1)[0]


def resolve(registry: GymRegistry, gym_id: str, task_types: list[str] | None,
            gym_weights: dict[str, float], task_weights: dict[str, float], split: str) -> Plan:
    available = {t.id for t in registry.available(gym_id)}
    composite = registry.gym(gym_id).provider is None
    if task_types is None:
        if not composite:
            raise ValueError("task_types is required for a source gym")
        task_types = sorted(available)
    if not task_types or set(task_types) - available:
        raise ValueError(f"unknown task types for {gym_id!r}: {sorted(set(task_types) - available)}")
    if not composite and (gym_weights or task_weights):
        raise ValueError("sampling weights are only supported by the general gym")
    if composite and len(set(task_types)) != len(task_types):
        raise ValueError("duplicate task types; use task_weights to change their frequency")
    sources = sorted({registry.gym_of(t) for t in task_types})
    if set(gym_weights) - set(sources) or set(task_weights) - set(task_types):
        raise ValueError("weights must name selected source gyms or task types")
    manifest = {
        "version": VERSION, "mode": "mixture" if composite else "cycle", "split": split,
        "sources": [
            {"gym_id": source, "weight": gym_weights.get(source, 1.0),
             "generator_version": registry.provider(source).version,
             "tools": list(registry.provider(source).tools),
             "split": split if registry.provider(source).splits else None,
             "task_types": [{"id": task, "weight": task_weights.get(task, 1.0)}
                            for task in sorted(set(task_types)) if registry.gym_of(task) == source]}
            for source in sources
        ],
    }
    return Plan(tuple(sorted(task_types) if composite else task_types), manifest)
