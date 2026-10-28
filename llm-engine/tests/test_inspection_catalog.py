"""Actual occurrence discovery and deliberate standard-only legacy selection compatibility."""

import base64

import pytest

from llm.architecture.errors import ArchitectureError
from llm.architecture.wire import AttentionSelection
from tests.test_architecture_builtins import build_template
from tests.test_inspection_runtime import runtime


@pytest.mark.parametrize("name,count,legacy", [("standard", 2, (2, 2)), ("looped", 2, None), ("nested", 4, None)])
def test_discovery_exposes_actual_candidates_without_invented_legacy_dimensions(tmp_path, name, count, legacy):
    """Candidate metadata identifies distinct real calls; only the direct saved standard composite retains numeric layer selection."""
    active, context = runtime(build_template(name), tmp_path)
    try:
        catalog = active.catalog("fixture")
        assert len(catalog.points) == count and all(point.evidence_hash is None for point in catalog.points)
        assert catalog.runtime_id == context.bootstrap.worker_id and catalog.definition_id == active.pins.definition_id
        assert active.legacy_dimensions() == legacy
        if legacy:
            selected, layer = active.selected_binding(AttentionSelection(text="a", layer=1))
            assert layer == 1 and selected.point.id == catalog.points[1].id
        else:
            assert catalog.legacy_layers is None and catalog.legacy_heads is None
            with pytest.raises(ArchitectureError, match="point"):
                active.selected_binding(AttentionSelection(text="a", layer=0))
    finally:
        active.close()


def test_matrix_serialization_keeps_actual_proof_and_does_not_synthesize_generic_layers(tmp_path, tokenizer):
    """Shared-loop public output binds the actual worker and point, exact byte axes and validated per-view bounds."""
    active, context = runtime(build_template("looped", vocab=tokenizer.get_vocab_size()), tmp_path)
    try:
        point = active.catalog("fixture").points[1]
        matrix = active.matrix("fixture", tokenizer, AttentionSelection(text="a", point_id=point.id, head=1))
        assert matrix.layer is None and matrix.layers is None and matrix.point_id == point.id
        assert matrix.runtime_id == context.bootstrap.worker_id and matrix.validated_sequence_max >= len(matrix.tokens)
        assert len(base64.b64decode(matrix.weights)) == len(matrix.tokens) ** 2 * 4
        proved = active.catalog("fixture").points
        assert proved[0].evidence_hash is None and proved[1].evidence_hash == matrix.evidence_hash
    finally:
        active.close()
