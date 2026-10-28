"""Developer schema publication and duplicate-safe shared-fixture verification.

This module reads versioned repository fixtures and delegates YAML syntax to the
production parser. It does not resolve models or own runtime/storage adapters.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm.architecture import spec, wire
from llm.architecture.parse import decode_yaml
from llm.runtime import protocol

CONTRACT_ROOT = Path(__file__).resolve().parents[3] / "contracts" / "modular-architecture" / "v1"
CONTRACT_MODELS = {**spec.CONTRACT_MODELS, **wire.CONTRACT_MODELS, **protocol.CONTRACT_MODELS}


def unique_json_fields(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous duplicate JSON fields before interpreting a fixture or public payload."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate_json_key")
        result[key] = value
    return result


def decode_json(text: str) -> Any:
    """Read JSON with duplicate and nonfinite-value rejection at the fixture boundary."""
    return json.loads(text, object_pairs_hook=unique_json_fields, parse_constant=invalid_constant)


def invalid_constant(value: str) -> None:
    """Refuse JavaScript-incompatible NaN/Infinity in shared protocol data."""
    raise ValueError(f"invalid JSON numeric constant: {value}")


def decode_fixture_yaml(text: str) -> object:
    """Apply the production bounded YAML syntax policy to shared contract fixtures.

    Resolution-negative fixtures remain declaration-only handoffs in this harness;
    production semantic behavior is exercised by the separate core test suite.
    """
    return decode_yaml(text)



def check_current_receipt(value: BaseModel, context: dict[str, object]) -> None:
    """Exercise exact configuration/profile receipt matching in shared stale fixtures."""
    if not isinstance(value, spec.ValidationReceipt):
        raise ValueError("receipt_check_requires_receipt")
    value.require_current(spec.ReceiptIdentity.model_validate(context["expected"]))


def check_graph_cursor(value: BaseModel, context: dict[str, object]) -> None:
    """Reject cursor replay across profile, scope, view or immutable graph artifact identity."""
    if not isinstance(value, wire.GraphCursor):
        raise ValueError("cursor_check_requires_cursor")
    value.require_query(wire.GraphQuery.model_validate(context["query"]))
    if value.graph_hash != context["graph_hash"]:
        raise ValueError("graph artifact changed")


CONTRACT_CHECKS: dict[str, Callable[[BaseModel, dict[str, object]], None]] = {
    **protocol.CONTRACT_CHECKS,
    "current_receipt": check_current_receipt,
    "graph_cursor": check_graph_cursor,
}


def fixture_cases() -> list[tuple[Path, dict[str, Any]]]:
    """Discover indexed shared examples while excluding generated schemas and unindexed data."""
    cases = []
    for folder in ("sdk", "recipes", "runtime", "api", "graphs", "definitions", "capabilities"):
        directory = CONTRACT_ROOT / folder
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json")):
            payload = decode_json(path.read_text())
            if isinstance(payload, dict) and "cases" in payload:
                if type(payload.get("schema_version")) is not int or payload["schema_version"] != 1:
                    raise ValueError("unsupported fixture version")
                cases.extend((path, case) for case in payload["cases"])
    return cases


def validate_fixture(path: Path, case: dict[str, Any]) -> BaseModel:
    """Validate one schema/syntax example and its explicitly requested identity expectation."""
    if "path" in case:
        target = path.parent / spec.checked_relative_path(case["path"])
        raw = target.read_text()
        payload = decode_fixture_yaml(raw) if target.suffix == ".yaml" else decode_json(raw)
    elif "raw" in case:
        payload = decode_json(case["raw"])
    else:
        payload = case["value"]
    model = CONTRACT_MODELS[case["model"]].model_validate(payload)
    if case.get("check"):
        CONTRACT_CHECKS[case["check"]](model, case.get("context", {}))
    return model


def schema_payloads() -> dict[str, dict[str, object]]:
    """Generate standalone language-neutral model schemas from the authoritative declarations."""
    return {name: model.model_json_schema() for name, model in sorted(CONTRACT_MODELS.items())}


def publish_schemas(check: bool) -> None:
    """Write generated schema files, or fail if checked-in schemas drift from declarations."""
    directory = CONTRACT_ROOT / "schemas"
    directory.mkdir(parents=True, exist_ok=True)
    expected = {f"{name}.schema.json" for name in CONTRACT_MODELS}
    if check and {path.name for path in directory.glob("*.schema.json")} != expected:
        raise ValueError("missing or obsolete generated contract schema")
    for name, schema in schema_payloads().items():
        path = directory / f"{name}.schema.json"
        text = json.dumps(schema, indent=2, sort_keys=True) + "\n"
        if check:
            if not path.exists() or path.read_text() != text:
                raise ValueError(f"stale generated contract schema: {name}")
        else:
            path.write_text(text)


def main() -> None:
    """Publish or check schemas through one reproducible developer entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    publish_schemas(args.check)
    print(f"{'Checked' if args.check else 'Wrote'} {len(CONTRACT_MODELS)} contract schemas")


if __name__ == "__main__":
    main()
