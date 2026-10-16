"""Pinned outputs captured from the original text.py/filter.py implementations."""

import dataclasses
import json
from pathlib import Path

import processors

SNAPSHOT = json.loads(
    (Path(__file__).parent / "fixtures/processor_migration.json").read_text()
)


def test_migration_preserves_original_outputs_and_catalogue():
    """Preserve original metadata and outputs for five document shapes and filter options."""
    for expected in SNAPSHOT["metadata"]:
        p = processors.REGISTRY[expected["name"]]
        assert {
            "name": p.name,
            "title": p.title,
            "description": p.description,
            "applies_to": [k.value for k in p.applies_to],
            "params": [dataclasses.asdict(param) for param in p.params],
        } == expected
    for case in SNAPSHOT["cases"]:
        p = processors.REGISTRY[case["name"]]
        state = {}
        actual = [
            p.apply(doc, p.resolve(case["given"]), state)
            for doc in SNAPSHOT["documents"]
        ]
        assert actual == case["after"], case["name"]
