"""Shared SDK/YAML/runtime/wire fixtures and contract identity invariants."""

import pytest
from pydantic import ValidationError
import yaml

from llm.architecture import contracts

CASES = contracts.fixture_cases()


@pytest.mark.parametrize("path,case", CASES, ids=[f"{path.parent.name}/{case['name']}" for path, case in CASES])
def test_shared_contract_fixture(path, case):
    """Check declaration/syntax validity, with resolver-negative fixtures explicitly distinguished."""
    if case.get("stage") == "resolution":
        contracts.validate_fixture(path, case)
        assert case["valid"] is False and case.get("error_code")
        return
    if case["valid"]:
        contracts.validate_fixture(path, case)
    else:
        with pytest.raises((ValidationError, ValueError, yaml.YAMLError)):
            contracts.validate_fixture(path, case)


def test_generated_schemas_match_source():
    """Prevent Python domain changes from silently drifting Rust/TypeScript fixture schemas."""
    contracts.publish_schemas(check=True)


def test_every_public_family_has_positive_and_negative_schema_examples():
    """Require representative success and rejection for each versioned public schema family."""
    covered: dict[str, set[bool]] = {}
    for _, case in CASES:
        if case.get("stage") != "resolution":
            covered.setdefault(case["model"], set()).add(case["valid"])
    required = {
        "BlockDeclaration", "Recipe", "BlockCreate", "ArchitectureCreate",
        "MetadataPatch", "BlockDraftSave", "ArchitectureDraftSave", "CatalogQuery",
        "AcceptRevision", "ValidationRequest", "ValidationReport", "ValidationReceipt",
        "GraphQuery", "GraphCursor", "GraphFragment", "ArchitectureSelection",
        "EnqueueEvidence", "AuthorizedVisibility", "ValidationRun", "ValidationEvent",
        "ModelDefinitionSummary", "DefinitionPackage", "Capability", "RuntimeEnvelope",
    }
    assert {name: covered.get(name) for name in required if covered.get(name) != {False, True}} == {}


def test_json_duplicate_fields_are_not_silently_overwritten():
    """Reject duplicate values even where a schema validator would keep only the last value."""
    with pytest.raises(ValueError, match="duplicate_json_key"):
        contracts.decode_json('{"schema_version":1,"schema_version":1}')


def test_semantic_resolution_cases_are_handoff_evidence_only():
    """Keep downstream resolver expectations visible without claiming they ran in ISS-49."""
    semantic = [case for _, case in CASES if case.get("stage") == "resolution"]
    assert semantic
    assert all(case.get("error_code") and case["valid"] is False for case in semantic)
