"""Production validation workloads, complete capture domains and real worker cleanup evidence."""

import hashlib
import json
from pathlib import Path
from uuid import uuid4

import pytest
import torch
import yaml

from llm.architecture.builtins.catalog import builtin_definitions
from llm.architecture.builtins.recipes import recipe_template
from llm.architecture.dependencies import recipe_dependencies
from llm.architecture.graphs import architecture_scopes, read_fragment, verify_receipt_graphs
from llm.architecture.spec import DraftSubject, FinalTokenizer, RevisionRef, RuntimeProfile, ShapeDomain
from llm.architecture.source import imports_public_builtins
from llm.architecture.validation import validate_snapshot
from llm.architecture.validation_records import PinnedSource, ValidationOutput, ValidationSnapshot
from llm.architecture.wire import GraphQuery, TensorInputSpec, ValidationRequest
from llm.model.device import DeviceGate
from llm.runtime.protocol import Inspect, RuntimeDeadlines, RuntimePolicy, RuntimeResources, Validate
from llm.runtime.supervisor import WorkerManager, WorkerStopped
from tests.test_architecture_core import LM_SOURCE, SOURCE, causal_recipe, revision, tensor_recipe
from tests.test_architecture_builtins import build_template


def profile(**changes) -> RuntimeProfile:
    """Bind the actual CPU runtime with bounded explicit train/eval full-prefix domains."""
    values = dict(torch_version=torch.__version__, sdk_version=1, device="cpu", precision="fp32",
                  capture_policy="torch-export-strict-v1", modes=("train", "eval"),
                  domains=(ShapeDomain(batch_min=1, batch_max=3, sequence_min=1, sequence_max=4, signatures=("full_prefix",)),))
    return RuntimeProfile(**{**values, **changes})


def recipe_snapshot(name="standard", **request_changes) -> ValidationSnapshot:
    """Freeze one real builtin recipe and exact dependency pins before worker admission."""
    recipe = recipe_template(name)
    source = yaml.safe_dump(recipe.model_dump(mode="json", exclude_none=True))
    subject = DraftSubject(kind="draft", resource_id=uuid4(), draft_version=1,
                           content_hash=hashlib.sha256(source.encode()).hexdigest())
    values = dict(schema_version=1, request_id=uuid4(), mode="full", subject=subject,
                  parameters={"dimensions": 8, "heads": 2, "ff_multiplier": 2, "context_window": 8},
                  tokenizer=FinalTokenizer(kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=16),
                  profile=profile())
    values.update(request_changes)
    return ValidationSnapshot(run_id=uuid4(), request=ValidationRequest(**values), recipe=source,
                              builtins=tuple({value.revision_id: value for value in recipe_dependencies(recipe).values()}.values()))


def source_snapshot(source=SOURCE, *, inspect=False, **request_changes) -> ValidationSnapshot:
    """Freeze ordinary block source with declaration-only or exact synthetic tensor inputs."""
    values = dict(schema_version=1, request_id=uuid4(), mode="inspect" if inspect else "full",
                  subject=DraftSubject(kind="draft", resource_id=uuid4(), draft_version=1,
                                       content_hash=hashlib.sha256(source.encode()).hexdigest()))
    if not inspect:
        values.update(profile=profile(domains=(ShapeDomain(batch_min=2, batch_max=2, sequence_min=3, sequence_max=3, signatures=("full_prefix",)),)),
                      tensor_inputs={"hidden": TensorInputSpec(dtype="float32", shape=(2, 3, 8))})
    values.update(request_changes)
    builtins = tuple(value.reference for value in builtin_definitions().values()) if imports_public_builtins(source) else ()
    return ValidationSnapshot(run_id=uuid4(), request=ValidationRequest(**values), source=source, builtins=builtins)


def tensor_recipe_snapshot(*, reused: bool) -> ValidationSnapshot:
    """Freeze a two-output tensor repeat, optionally called twice through one pinned child architecture."""
    block = revision(SOURCE)
    recipe = tensor_recipe(block.reference).model_dump(mode="json", exclude_none=True)
    recipe["outputs"]["original"] = recipe["inputs"]["hidden"]
    recipe["result"]["original"] = "hidden"
    dependencies = [PinnedSource(reference=block.reference, source=SOURCE)]
    if reused:
        child_source = json.dumps(recipe)
        child = RevisionRef(kind="architecture", resource_id=uuid4(), revision_id=uuid4(),
                            content_hash=hashlib.sha256(child_source.encode()).hexdigest())
        dependencies.append(PinnedSource(reference=child, source=child_source))
        recipe["modules"] = {"branch": {"reference": child.model_dump(mode="json"), "parameters": {}}}
        recipe["flow"] = [
            {"call": "branch", "inputs": {"hidden": "hidden"},
             "outputs": {"hidden": "first", "original": "first_original"}},
            {"call": "branch", "inputs": {"hidden": "first"},
             "outputs": {"hidden": "result", "original": "original"}},
        ]
        recipe["result"]["original"] = "original"
    raw = json.dumps(recipe)
    base = source_snapshot()
    request = base.request.model_copy(update={"subject": DraftSubject(kind="draft", resource_id=uuid4(),
        draft_version=1, content_hash=hashlib.sha256(raw.encode()).hexdigest())})
    return ValidationSnapshot(run_id=uuid4(), request=request, recipe=raw, dependencies=tuple(dependencies))


@pytest.mark.parametrize("reused", [False, True])
def test_tensor_recipes_validate_without_tokenizer_and_retain_shared_child_graphs(reused):
    """Strict train/eval values and gradients authorize exact tensor profiles and publish reusable hierarchy without LM facts."""
    from llm.architecture.validation_artifacts import PublishedValidation

    snapshot = tensor_recipe_snapshot(reused=reused)
    output = validate_snapshot(snapshot, device="cpu")
    assert output.report.verdict == "valid", output.report.diagnostics
    assert output.report.unique_parameter_count == 1
    receipt = output.report.receipt
    assert receipt.identity.tokenizer is None and receipt.identity.context_window is None
    verify_receipt_graphs(receipt, output.graph_blobs)
    PublishedValidation(snapshot=snapshot, output=output)
    manifest = json.loads(output.graph_blobs[receipt.graph_manifest_hash])
    assert {(entry["view"], entry["mode"]) for entry in manifest["graphs"]} == {
        (view, mode) for view in ("architecture", "operations") for mode in ("train", "eval")}
    query = GraphQuery(identity_hash=receipt.identity_hash, view="architecture", mode="eval",
                       signature="full_prefix", scope="root/branch" if reused else "root")
    fragment = read_fragment(receipt.graph_manifest_hash, output.graph_blobs, query)
    assert {node.label for node in fragment.nodes if node.kind == "output"} == {"hidden", "original"}


@pytest.mark.parametrize("invalid", ["missing_input", "ranged_shape"])
def test_tensor_recipe_validation_rejects_incomplete_synthetic_domains(invalid):
    """Tensor recipes require every explicit input and an exact shape before any receipt or graph is issued."""
    snapshot = tensor_recipe_snapshot(reused=True)
    changes = {"tensor_inputs": {}} if invalid == "missing_input" else {"profile": profile()}
    snapshot = snapshot.model_copy(update={"request": snapshot.request.model_copy(update=changes)})
    output = validate_snapshot(snapshot, device="cpu")
    assert output.report.verdict == "invalid" and output.report.receipt is None
    assert output.report.diagnostics[0].code == ("tensor_inputs" if invalid == "missing_input" else "block_shape_domain")
    assert not output.graph_blobs


@pytest.mark.parametrize("name", ["standard", "looped", "nested"])
def test_complete_builtin_validation_domains_and_graph_scopes(name):
    """Actual strict train/eval captures cover singleton/ranged inputs and publish hierarchical metadata."""
    output = validate_snapshot(recipe_snapshot(name), device="cpu")
    assert output.report.verdict == "valid", output.report.diagnostics
    receipt = output.report.receipt
    manifest = json.loads(output.graph_blobs[receipt.graph_manifest_hash])
    operations = [entry for entry in manifest["graphs"] if entry["view"] == "operations" and entry["scope"] != "root"]
    assert len(operations) == 8
    assert {entry["mode"] for entry in operations} == {"train", "eval"}
    assert all(entry["strict"] for entry in operations)
    verify_receipt_graphs(receipt, output.graph_blobs)
    assert manifest["identity_hash"] == receipt.identity_hash
    recorded = [json.loads(value) for value in receipt.identity.profile.export_guards]
    assert recorded == [{key: entry[key] for key in ("mode", "signature", "scope", "strict", "guards", "domain")}
                        for entry in operations]
    query = GraphQuery(identity_hash=receipt.identity_hash, view="architecture", mode="eval", signature="full_prefix", limit=2)
    first = read_fragment(receipt.graph_manifest_hash, output.graph_blobs, query)
    assert len(first.nodes) == 2 and first.next_cursor
    second = read_fragment(receipt.graph_manifest_hash, output.graph_blobs, query.model_copy(update={"cursor": first.next_cursor}))
    assert not {node.id for node in first.nodes} & {node.id for node in second.nodes}
    with pytest.raises(ValueError):
        read_fragment(receipt.graph_manifest_hash, output.graph_blobs,
                      query.model_copy(update={"scope": "invalid", "cursor": first.next_cursor}))


def test_inspection_never_constructs_and_ordinary_block_full_validation_is_exact():
    """Inspection ignores missing constructor inputs while full validation captures real tensor arithmetic."""
    source = SOURCE.replace("super().__init__()", "raise RuntimeError('must not construct')")
    inspected = validate_snapshot(source_snapshot(source, inspect=True), device="cpu")
    assert inspected.report.verdict == "inspection_only" and inspected.report.receipt is None
    assert not inspected.graph_blobs
    validated = validate_snapshot(source_snapshot(), device="cpu")
    assert validated.report.verdict == "valid", validated.report.diagnostics


def test_invalid_python_retains_parser_position_under_immutable_source_identity():
    """A saved syntax error locates its exact private revision and parser line/column without exposing a host path."""
    snapshot = source_snapshot("def broken(\n", inspect=True)
    output = validate_snapshot(snapshot, device="cpu")
    assert output.report.verdict == "invalid" and output.report.receipt is None
    error = output.report.diagnostics[0]
    assert error.code == "invalid_source" and error.phase == "inspection"
    assert error.source.line == 1 and error.source.column > 0
    assert error.source.file == f"blocks/{error.revision_id}.py"


@pytest.mark.parametrize("source,code", [
    (SOURCE.replace("hidden * self.gain", "hidden[:, :, 0] * self.gain"), "port_shape"),
    (SOURCE.replace('return {"hidden": hidden * self.gain}', 'hidden.add_(1)\n        return {"hidden": hidden * self.gain}'), "input_mutation"),
    (SOURCE.replace("hidden * self.gain", "hidden * self.gain.item()"), "unsupported_source"),
    (SOURCE.replace("hidden * self.gain", "hidden * self.gain / 0"), "nonfinite_output"),
])
def test_invalid_source_shape_mutation_and_nonfinite_paths_never_receive_receipts(source, code):
    """Failed validation retains bounded diagnostics and no acceptance graph/receipt."""
    output = validate_snapshot(source_snapshot(source), device="cpu")
    assert output.report.verdict == "invalid"
    assert output.report.diagnostics[0].code == code
    if code == "nonfinite_output":
        assert output.report.diagnostics[0].phase == "numerical"
    assert output.report.receipt is None and not output.graph_blobs


def test_unknown_runtime_cache_domain_and_context_override_fail_closed():
    """Profiles cannot silently authorize another runtime, unsupported cache or larger context."""
    for snapshot in (recipe_snapshot(profile=profile(torch_version="different")),
                     recipe_snapshot(profile=profile(domains=(ShapeDomain(batch_min=1, batch_max=1, sequence_min=1, sequence_max=2, signatures=("decode",)),))),
                     recipe_snapshot(context_window=20)):
        output = validate_snapshot(snapshot, device="cpu")
        assert output.report.verdict == "invalid" and output.report.receipt is None


@pytest.mark.parametrize("expression", [
    "(hidden @ self.weight * self.scale).roll(1, dims=1)",
    "hidden.mean(dim=0,keepdim=True).expand_as(hidden) @ self.weight * self.scale",
    "hidden @ self.weight * self.scale + torch.rand_like(hidden)",
])
def test_shape_valid_future_batch_coupled_and_stochastic_eval_models_fail_numerical_preflight(expression):
    """Complete capture alone cannot authorize information leaks or nondeterministic serving behavior."""
    source = LM_SOURCE.replace("hidden @ self.weight * self.scale", expression)
    pinned = revision(source)
    raw = yaml.safe_dump(causal_recipe(pinned.reference).model_dump(mode="json", exclude_none=True))
    base = recipe_snapshot()
    request = base.request.model_copy(update={"subject": DraftSubject(kind="draft", resource_id=uuid4(), draft_version=1,
                        content_hash=hashlib.sha256(raw.encode()).hexdigest()), "parameters": {}})
    snapshot = ValidationSnapshot(run_id=uuid4(), request=request, recipe=raw,
                                  dependencies=(PinnedSource(reference=pinned.reference, source=source),))
    output = validate_snapshot(snapshot, device="cpu")
    assert output.report.verdict == "invalid", output.report.diagnostics
    assert output.report.diagnostics[0].code == "numerical_mismatch", output.report.diagnostics
    assert output.report.diagnostics[0].phase == "numerical"


def test_validation_preserves_rng_on_success_failure_and_callback_cancellation():
    """Fake/real construction and probes never consume caller RNG; cancellation is not a validation verdict."""
    before = torch.get_rng_state().clone()
    assert validate_snapshot(source_snapshot(), device="cpu").report.verdict == "valid"
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    assert validate_snapshot(source_snapshot(SOURCE.replace("hidden * self.gain", "hidden")), device="cpu").report.verdict == "invalid"
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)

    def cancelled(phase):
        """Simulate a worker's cancellation boundary before any further source execution."""
        raise InterruptedError("cancelled")

    with pytest.raises(InterruptedError):
        validate_snapshot(source_snapshot(), device="cpu", progress=cancelled)
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)


def test_hierarchy_stack_static_carry_edges_and_nested_repeat_feedback_are_faithful():
    """Independent layers have real inter-layer wiring; repeats expose carried feedback and captured ports."""
    expanded = build_template("expanded")
    scopes = architecture_scopes(expanded.plan, expanded)
    nodes, edges = scopes["root/1/body"]
    assert any(edge["source"] == "stack:0" and edge["target"] == "stack:1" and edge["target_port"] == "hidden" for edge in edges)
    assert any(edge["source"] == "stack:1" and edge["target"] == "output:hidden" for edge in edges)
    for index in (0, 1):
        assert {edge["target_port"] for edge in edges if edge["target"] == f"stack:{index}"} == {"hidden", "positions", "key_mask"}
    nested = build_template("nested")
    scopes = architecture_scopes(nested.plan, nested)
    for scope in ("root/1/body", "root/1/body/0/body"):
        nodes, edges = scopes[scope]
        assert any(edge["source"].startswith("output:") and edge["target"].startswith("carry:") for edge in edges)
    root_nodes, root_edges = scopes["root"]
    assert {edge["target_port"] for edge in root_edges if edge["target"] == "flow:1"} == {"outer_hidden", "positions", "key_mask"}


def test_stateless_integer_tensor_block_and_capture_budgets(monkeypatch):
    """Integer-only tensor routing needs no artificial loss, while oversized captures fail explicitly."""
    source = '''from torch import nn
from llm.architecture.sdk import block, BlockDeclaration, TensorPort
@block(BlockDeclaration(sdk_version=1,name="integer_identity",title="Integer identity",parameters={},
 inputs={"tokens":TensorPort(dtype="int64",shape=("batch","sequence"))},
 outputs={"tokens":TensorPort(dtype="int64",shape=("batch","sequence"))}))
class IntegerIdentity(nn.Module):
    """Route explicit integer tokens with no floating/trainable path."""
    def __init__(self):
        """Own no persistent tensor state."""
        super().__init__()
    def forward(self,tokens):
        """Return caller tokens without modifying their contents."""
        return {"tokens":tokens}
'''
    output = validate_snapshot(source_snapshot(source, tensor_inputs={"tokens": TensorInputSpec(dtype="int64", shape=(2, 3))}), device="cpu")
    assert output.report.verdict == "valid", output.report.diagnostics
    assert output.report.unique_parameter_count == 0
    monkeypatch.setattr("llm.architecture.capture.MAX_GRAPH_NODES", 1)
    output = validate_snapshot(source_snapshot(), device="cpu")
    assert output.report.verdict == "invalid" and output.report.diagnostics[0].code == "graph_node_limit"


def test_stale_receipt_profile_and_corrupt_graph_bytes_are_rejected():
    """Profile/domain changes and graph-byte tampering cannot reuse earlier validation evidence."""
    output = validate_snapshot(source_snapshot(), device="cpu")
    receipt = output.report.receipt
    with pytest.raises(ValueError):
        receipt.require_current(receipt.identity.model_copy(update={"parameter_digest": "b" * 64}))
    blobs = {**output.graph_blobs, receipt.graph_manifest_hash: "{}"}
    query = GraphQuery(identity_hash=receipt.identity_hash, view="operations", mode="eval", signature="full_prefix")
    with pytest.raises(ValueError):
        read_fragment(receipt.graph_manifest_hash, blobs, query)
    with pytest.raises(ValueError):
        verify_receipt_graphs(receipt, blobs)


def test_capture_receipt_records_actual_guards_and_refuses_supplied_mismatch():
    """Discovered strict export identities can be repeated exactly but stale claims cannot authorize a receipt."""
    snapshot = source_snapshot()
    output = validate_snapshot(snapshot, device="cpu")
    actual = output.report.receipt.identity.profile
    assert len(actual.export_guards) == 2
    repeated = snapshot.model_copy(update={"request": snapshot.request.model_copy(update={"profile": actual})})
    verified = validate_snapshot(repeated, device="cpu")
    assert verified.report.verdict == "valid", verified.report.diagnostics
    assert verified.report.receipt.identity_hash == output.report.receipt.identity_hash
    stale = actual.model_copy(update={"export_guards": ("stale guards",)})
    rejected = validate_snapshot(snapshot.model_copy(update={"request": snapshot.request.model_copy(update={"profile": stale})}), device="cpu")
    assert rejected.report.verdict == "invalid"
    assert rejected.report.diagnostics[0].code == "capture_guard_mismatch"
    assert rejected.report.receipt is None and not rejected.graph_blobs


def worker_policy() -> RuntimePolicy:
    """Admit CPU capture in a real spawned worker under finite address/RSS/deadline bounds."""
    return RuntimePolicy(deadlines=RuntimeDeadlines(import_seconds=30.0, construction_seconds=30.0, validation_seconds=90.0),
        resources=RuntimeResources(process_address_bytes=16 << 30, tree_resident_bytes=3 << 30,
            tree_memory_enforcement="preflight_and_monitor", device="cpu", device_allocator_bytes=0,
            owned_processes=2, artifact_bytes=8 << 20, total_artifact_bytes=16 << 20))


def test_supervised_inspection_full_validation_and_invalid_cleanup(tmp_path):
    """Only child execution imports/constructs source, and admission returns after verified process cleanup."""
    gate = DeviceGate()
    with WorkerManager(tmp_path / "runtime", gate) as manager:
        with manager.session(None, worker_policy(), "llm.architecture.worker:ValidationBackend", metadata_only=True) as session:
            artifact = session.artifacts.write_bytes(source_snapshot(inspect=True).model_dump_json().encode(), "source_snapshot_v1")
            completed = session.run(Inspect(type="inspect", snapshot=artifact))
            assert ValidationOutput.model_validate_json(session.artifacts.read_bytes(completed.result)).report.verdict == "inspection_only"
        with gate.admit() as lease:
            with manager.session(lease, worker_policy(), "llm.architecture.worker:ValidationBackend") as session:
                for snapshot in (recipe_snapshot(), source_snapshot(SOURCE.replace("hidden * self.gain", "hidden * self.gain / 0"))):
                    artifact = session.artifacts.write_bytes(snapshot.model_dump_json().encode(), "validation_snapshot_v1")
                    completed = session.run(Validate(type="validate", snapshot=artifact))
                    output = ValidationOutput.model_validate_json(session.artifacts.read_bytes(completed.result))
                    assert output.report.verdict == ("valid" if snapshot.recipe else "invalid"), output.report.diagnostics
        with gate.admit():
            pass
    assert not [path for path in (tmp_path / "runtime").iterdir() if path.is_dir()]


def test_supervised_custom_forward_hang_is_terminated_and_releases_admission(tmp_path):
    """The real validation backend executes hung source only inside a deadline-owned child process."""
    source = SOURCE.replace('return {"hidden": hidden * self.gain}', 'while True:\n            pass')
    gate = DeviceGate()
    policy = worker_policy()
    policy = policy.model_copy(update={"deadlines": policy.deadlines.model_copy(update={"validation_seconds": 0.3})})
    with WorkerManager(tmp_path / "runtime", gate) as manager:
        with gate.admit() as lease:
            with pytest.raises(WorkerStopped):
                with manager.session(lease, policy, "llm.architecture.worker:ValidationBackend") as session:
                    artifact = session.artifacts.write_bytes(source_snapshot(source).model_dump_json().encode(), "validation_snapshot_v1")
                    session.run(Validate(type="validate", snapshot=artifact))
        with gate.admit():
            pass


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU hardware validation deferred by user")
@pytest.mark.parametrize("precision", ["fp32", "fp16", "bf16"])
def test_actual_cuda_validation_profile(precision):
    """Keep complete real-device capture/preflight checks runnable on supported NVIDIA hardware."""
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        pytest.skip("device does not support bfloat16")
    output = validate_snapshot(recipe_snapshot(profile=profile(device="cuda", precision=precision,
        allow_tf32=torch.backends.cuda.matmul.allow_tf32)), device="cuda")
    assert output.report.verdict == "valid", output.report.diagnostics
