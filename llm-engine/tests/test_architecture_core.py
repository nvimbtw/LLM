"""Production SDK, resolver and tensor execution contracts before builtin integration."""

import copy
import hashlib
import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
import torch
from torch import nn

from llm.architecture.build import build_model, parameter_groups
from llm.architecture.errors import ArchitectureError
from llm.architecture.parse import parse_recipe
from llm.architecture.plan import ArchitectureDefinition, BlockDefinition
from llm.architecture.resolve import resolve_recipe
from llm.architecture.source import SourceLoader, SourceRevision
from llm.architecture.spec import BlockDeclaration, FinalTokenizer, Recipe, RevisionRef

ROOT = Path(__file__).resolve().parents[2] / "contracts/modular-architecture/v1"
SOURCE = (ROOT / "sdk/custom_scale.py").read_text()


def revision(source: str) -> SourceRevision:
    """Create an exact immutable source reference for an isolated test revision."""
    reference = RevisionRef(kind="block", resource_id=uuid4(), revision_id=uuid4(),
                            content_hash=hashlib.sha256(source.encode()).hexdigest())
    return SourceRevision(reference, source)


def tensor_recipe(reference: RevisionRef, operation: dict | None = None) -> Recipe:
    """Define one token-local tensor recipe with explicit loop carry and constructor bindings."""
    port = {"dtype": "floating", "shape": ["batch", "sequence", "width"]}
    return Recipe.model_validate({
        "schema_version": 1, "interface": "tensor", "parameters": {},
        "inputs": {"hidden": port}, "outputs": {"hidden": port},
        "result": {"hidden": "result"},
        "modules": {"scale": {"reference": reference, "parameters": {"gain": 0.5}}},
        "flow": [operation or {"repeat": 4,
            "carry": {"x": {"initial": "hidden", "input": "current", "output": "next_hidden", "result": "result"}},
            "body": [{"call": "scale", "inputs": {"hidden": "current"}, "outputs": {"hidden": "next_hidden"}}]}],
    })


def tokenizer() -> FinalTokenizer:
    """Supply finalized immutable vocabulary facts for shared causal recipe fixtures."""
    return FinalTokenizer(kind="final", tokenizer_id=uuid4(), fingerprint="a" * 64, vocabulary_size=32)


def test_reserved_token_binding_uses_the_authoritative_full_id_name():
    """A real tokenizer's pad_id reaches recipe construction unchanged; absent marker bindings fail resolution."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from llm.architecture.tokenizer_binding import binding_from_bytes

    raw = Tokenizer(WordLevel(vocab={"one": 0, "two": 1, "<|endoftext|>": 2},
                              unk_token="<|endoftext|>")).to_str().encode()
    binding = binding_from_bytes(uuid4(), raw, ("pad_id",))
    assert binding.reserved_token_ids == {"pad_id": 2}
    with SourceLoader() as loader:
        definition = loader.load(revision(SOURCE))
        value = tensor_recipe(definition.reference).model_dump(mode="python")
        value["modules"]["scale"]["parameters"]["gain"] = "${tokenizer.pad_id}"
        recipe = Recipe.model_validate(value)
        dependencies = {definition.reference.revision_id: definition}
        plan = resolve_recipe(recipe, {}, dependencies, binding)
        model = build_model(plan)
        hidden = torch.ones(1, 2, 8)
        torch.testing.assert_close(model.forward_named({"hidden": hidden})["hidden"], hidden * 16)
        assert model.instances["scale"].gain.item() == 2
        with pytest.raises(ArchitectureError) as raised:
            resolve_recipe(recipe, {}, dependencies, binding.model_copy(update={"reserved_token_ids": {}}))
        assert raised.value.code == "tokenizer_binding"


def fixture_dependencies() -> dict:
    """Read synthetic declaration snapshots, not production builtin implementations."""
    catalog = json.loads((ROOT / "sdk/catalog.json").read_text())
    dependencies = {}
    for entry in catalog["blocks"].values():
        reference = RevisionRef.model_validate(entry["reference"])
        dependencies[reference.revision_id] = BlockDefinition(reference, BlockDeclaration.model_validate(entry["declaration"]), nn.Module)
    for entry in catalog["architectures"].values():
        reference = RevisionRef.model_validate(entry["reference"])
        recipe = parse_recipe((ROOT / "sdk" / entry["recipe_path"]).read_text())
        dependencies[reference.revision_id] = ArchitectureDefinition(reference, recipe)
    return dependencies


@pytest.mark.parametrize("name", ["duplicate-key", "unsafe-tag", "yaml-alias", "zero-repeat", "boolean-repeat", "expression-binding"])
def test_production_parser_rejects_shared_bad_yaml(name):
    """The actual parser rejects duplicate/unsafe syntax and invalid typed declarations."""
    with pytest.raises(ArchitectureError):
        parse_recipe((ROOT / "recipes" / f"negative-{name}.yaml").read_text())


def test_source_inspection_never_constructs_and_same_named_revisions_coexist():
    """Same Python names retain different revision code; inspection never runs constructors."""
    failing = SOURCE.replace("super().__init__()", "raise RuntimeError('constructor invoked')")
    with SourceLoader() as loader:
        first = loader.load(revision(SOURCE))
        second = loader.load(revision(SOURCE.replace("hidden * self.gain", "hidden * self.gain + 1")))
        inspected = loader.load(revision(failing))
        assert inspected.declaration.name == "custom_scale"
        a, b = first.module_type(gain=0.5), second.module_type(gain=0.5)
        assert type(a).__name__ == type(b).__name__ and type(a).__module__ != type(b).__module__
        torch.testing.assert_close(b(torch.ones(1, 1, 8))["hidden"], a(torch.ones(1, 1, 8))["hidden"] + 1)
    torch.testing.assert_close(a(torch.ones(1, 1, 8))["hidden"], torch.full((1, 1, 8), 0.5))


@pytest.mark.parametrize("source,code", [
    ("import os\n" + SOURCE, "unsupported_import"),
    ("SCALE = []\n" + SOURCE, "unsupported_source"),
    (SOURCE.replace("return {\"hidden\": hidden * self.gain}", "self.counter = 1\n        return {\"hidden\": hidden}"), "persistent_mutation"),
    (SOURCE.replace("hidden * self.gain", "hidden * self.gain.item()"), "unsupported_source"),
])
def test_ineligible_source_is_rejected_before_import(source, code):
    """Actual source loading rejects ambient state, forbidden imports and scalar specialization."""
    with SourceLoader() as loader, pytest.raises(ArchitectureError) as raised:
        loader.load(revision(source))
    assert raised.value.code == code


def test_source_hash_is_verified_before_import():
    """Tampering cannot replace pinned revision code even if the Python names are unchanged."""
    pinned = revision(SOURCE)
    with SourceLoader() as loader, pytest.raises(ArchitectureError, match="source bytes"):
        loader.load(SourceRevision(pinned.reference, SOURCE + "\n"))


def test_repeat_keeps_one_owner_and_all_unrolled_gradients():
    """Fixed repeat routes four shared calls with full autograd and immutable caller inputs."""
    with SourceLoader() as loader:
        definition = loader.load(revision(SOURCE))
        plan = resolve_recipe(tensor_recipe(definition.reference), {}, {definition.reference.revision_id: definition})
        model = build_model(plan)
    hidden = torch.randn(2, 3, 8)
    before = hidden.clone()
    actual = model.forward_named({"hidden": hidden})["hidden"]
    actual.sum().backward()
    parameter = model.instances["scale"].gain
    torch.testing.assert_close(actual, hidden * parameter ** 4)
    torch.testing.assert_close(parameter.grad, (hidden * 4 * parameter.detach() ** 3).sum())
    torch.testing.assert_close(hidden, before)
    assert len(list(model.parameters())) == 1 and plan.expanded_calls == 4


def test_independent_stack_does_not_share_parameters():
    """Stack explicitly constructs independent child identities with stable public paths."""
    with SourceLoader() as loader:
        definition = loader.load(revision(SOURCE))
        operation = {"stack": 3, "name": "layers", "template": {"reference": definition.reference,
                     "parameters": {"gain": 0.5}}, "carry": {"x": {"initial": "hidden", "input": "hidden",
                     "output": "hidden", "result": "result"}}}
        recipe = tensor_recipe(definition.reference, operation)
        recipe = recipe.model_copy(update={"modules": {}})
        model = build_model(resolve_recipe(recipe, {}, {definition.reference.revision_id: definition}))
    assert tuple(model.instances) == ("layers/0", "layers/1", "layers/2")
    assert len({id(parameter) for parameter in model.parameters()}) == 3
    hidden = torch.randn(1, 2, 8)
    torch.testing.assert_close(model.forward_named({"hidden": hidden})["hidden"], hidden * 0.5 ** 3)


def alias_recipe(reference: RevisionRef) -> Recipe:
    """Define two instances sharing their exact scalar Parameter after construction."""
    recipe = tensor_recipe(reference).model_dump(mode="python")
    recipe["modules"] = {name: {"reference": reference, "parameters": {"gain": 0.5}} for name in ("left", "right")}
    recipe["aliases"] = [{"owner": "left.gain", "target": "right.gain"}]
    recipe["flow"] = [{"call": "left", "inputs": {"hidden": "hidden"}, "outputs": {"hidden": "middle"}},
                      {"call": "right", "inputs": {"hidden": "middle"}, "outputs": {"hidden": "result"}}]
    return Recipe.model_validate(recipe)


def test_alias_identity_optimizer_ownership_and_conflicting_policy():
    """Exact aliases survive dtype placement, unique ownership and state round trip."""
    with SourceLoader() as loader:
        definition = loader.load(revision(SOURCE))
        plan = resolve_recipe(alias_recipe(definition.reference), {}, {definition.reference.revision_id: definition})
        model = build_model(plan, dtype=torch.float64)
        restored = build_model(plan, dtype=torch.float64)
        restored.load_state_dict(copy.deepcopy(model.state_dict()))
    assert model.instances["left"].gain is model.instances["right"].gain
    assert restored.instances["left"].gain is restored.instances["right"].gain
    assert len(parameter_groups(model)) == 1
    with pytest.raises(ArchitectureError, match="conflicting"):
        parameter_groups(model, {"left.gain": {"lr": 0.1}, "right.gain": {"lr": 0.2}})


def test_separate_parameters_with_overlapping_storage_are_rejected():
    """A second trainable view cannot silently enter the optimizer as an independent owner."""
    source = SOURCE.replace("self.gain = nn.Parameter(torch.tensor(gain, dtype=torch.float32))",
                            "self.gain = nn.Parameter(torch.tensor(gain, dtype=torch.float32))\n        self.other = nn.Parameter(self.gain.view_as(self.gain))")
    with SourceLoader() as loader:
        definition = loader.load(revision(source))
        plan = resolve_recipe(tensor_recipe(definition.reference), {}, {definition.reference.revision_id: definition})
        with pytest.raises(ArchitectureError) as raised:
            build_model(plan)
    assert raised.value.code == "overlapping_parameters"


@pytest.mark.parametrize("name,code", [
    ("unresolved-value", "unresolved_value"), ("duplicate-value", "duplicate_value"),
    ("uninitialized-carry", "uninitialized_carry"), ("nested-work-limit", "expanded_work_limit"),
    ("stale-revision", "revision_snapshot_mismatch"), ("tokenizer-disagreement", "tokenizer_binding_mismatch"),
    ("floating-token-input", "causal_input_dtype"),
])
def test_resolver_rejects_shared_semantic_negatives(name, code):
    """Genuinely reject previously declaration-only negative scopes, work and immutable pins."""
    recipe = parse_recipe((ROOT / "recipes" / f"negative-{name}.yaml").read_text())
    with pytest.raises(ArchitectureError) as raised:
        resolve_recipe(recipe, {}, fixture_dependencies(), tokenizer())
    assert raised.value.code == code


@pytest.mark.parametrize("name", ["standard", "expanded-standard", "looped", "custom-lm", "custom-branch-nested", "reused-subarchitecture"])
def test_complete_shared_recipes_resolve_without_allocating_builtins(name):
    """Resolve complete production contracts; numerical builtin equivalence remains ISS-53."""
    recipe = parse_recipe((ROOT / "recipes" / f"positive-{name}.yaml").read_text())
    plan = resolve_recipe(recipe, {}, fixture_dependencies(), tokenizer())
    assert 0 < plan.expanded_calls <= 128


LM_SOURCE = '''
import torch
from torch import nn
from llm.architecture.sdk import block, BlockDeclaration, ParameterDeclaration, TensorPort

@block(BlockDeclaration(sdk_version=1,name="tiny_lm",title="Tiny LM",
 parameters={"vocab":ParameterDeclaration(type="integer",required=True,minimum=1)},
 inputs={"tokens":TensorPort(dtype="int64",shape=("batch","sequence")),
         "positions":TensorPort(dtype="int64",shape=("batch","sequence")),
         "key_mask":TensorPort(dtype="bool",shape=("batch","sequence"))},
 outputs={"logits":TensorPort(dtype="floating",shape=("batch","sequence","vocabulary"))}))
class TinyLM(nn.Module):
    """A token-local causal vocabulary projection."""
    def __init__(self,vocab:int):
        """Create the final tokenizer-bound projection and reconstructible buffer."""
        super().__init__()
        self.vocab=vocab
        self.weight=nn.Parameter(torch.randn(vocab,vocab))
        self.register_buffer("scale",torch.ones(1),persistent=False)
    def forward(self,tokens,positions,key_mask):
        """Return named token-local logits without input or persistent mutation."""
        hidden=torch.nn.functional.one_hot(tokens,self.vocab).float()
        return {"logits":hidden @ self.weight * self.scale}
'''


def causal_recipe(reference: RevisionRef) -> Recipe:
    """Bind one complete causal module to tokenizer-owned vocabulary and trained context."""
    recipe = parse_recipe((ROOT / "recipes/positive-standard.yaml").read_text()).model_dump(mode="python")
    recipe["modules"] = {"model": {"reference": reference, "parameters": {"vocab": "${tokenizer.vocabulary_size}"}}}
    return Recipe.model_validate(recipe)


def test_causal_adapter_strict_capture_gradients_and_boundary_inputs():
    """The real adapter is fully exportable and validates canonical token/position/mask bounds."""
    with SourceLoader() as loader:
        definition = loader.load(revision(LM_SOURCE))
        plan = resolve_recipe(causal_recipe(definition.reference), {}, {definition.reference.revision_id: definition}, tokenizer())
        model = build_model(plan).eval()
        tokens = torch.randint(0, 32, (2, 4))
        exported = torch.export.export(model, (tokens,), strict=True)
        torch.testing.assert_close(exported.module()(tokens), model(tokens))
        model(tokens).sum().backward()
        assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in model.parameters())
        for invalid in (torch.full_like(tokens, -1), torch.full_like(tokens, 32)):
            with pytest.raises(ArchitectureError, match="vocabulary"):
                model(invalid)
            with pytest.raises(RuntimeError, match="vocabulary"):
                exported.module()(invalid)
        with pytest.raises(ArchitectureError, match="positions"):
            model(tokens, positions=torch.zeros(1, 4, dtype=torch.long))
        with pytest.raises(ArchitectureError, match="mask"):
            model(tokens, key_mask=torch.ones_like(tokens))

def test_meta_device_and_reconstructed_nonpersistent_buffers():
    """Construction supports allocation-free meta placement and reconstructs unsaved derived buffers."""
    with SourceLoader() as loader:
        definition = loader.load(revision(LM_SOURCE))
        plan = resolve_recipe(causal_recipe(definition.reference), {}, {definition.reference.revision_id: definition}, tokenizer())
        meta = build_model(plan, device="meta")
        first = build_model(plan, dtype=torch.float64)
        second = build_model(plan, dtype=torch.float64)
        second.load_state_dict(first.state_dict())
    assert all(parameter.device.type == "meta" for parameter in meta.parameters())
    assert not any("scale" in key for key in first.state_dict())
    torch.testing.assert_close(first.instances["model"].scale, second.instances["model"].scale)
    assert first.instances["model"].scale.dtype == torch.float64


def test_real_execution_rejects_input_and_persistent_tensor_mutation():
    """Eager tensor version guards catch writes invisible to Python attribute-assignment checks."""
    with SourceLoader() as loader:
        source = SOURCE.replace('return {"hidden": hidden * self.gain}',
                                'hidden.add_(1)\n        return {"hidden": hidden * self.gain}')
        definition = loader.load(revision(source))
        model = build_model(resolve_recipe(tensor_recipe(definition.reference), {}, {definition.reference.revision_id: definition}))
        with pytest.raises(ArchitectureError) as raised:
            model.forward_named({"hidden": torch.randn(1, 2, 8)})
        assert raised.value.code == "input_mutation"


CACHE_SOURCE = '''
import torch
from torch import nn
from llm.architecture.sdk import block, BlockDeclaration, TensorPort
from llm.architecture.interfaces import BlockCachedResult

@block(BlockDeclaration(sdk_version=1,name="prefix_sum",title="Prefix sum",parameters={},
 inputs={"hidden":TensorPort(dtype="floating",shape=("batch","sequence","width"))},
 outputs={"hidden":TensorPort(dtype="floating",shape=("batch","sequence","width"))},capabilities=("cache",)))
class PrefixSum(nn.Module):
    """A causal hidden-value operation with explicit named cache state."""
    def __init__(self):
        """Construct no hidden request state."""
        super().__init__()
    def forward(self,hidden):
        """Return complete-prefix cumulative values."""
        return {"hidden":hidden.cumsum(dim=1)}
    def cached_forward(self,inputs,state):
        """Continue from caller-owned sums and return fresh state tensors."""
        hidden=inputs["hidden"]
        previous=state.get("sum",torch.zeros_like(hidden[:,:1]))
        result=hidden.cumsum(dim=1)+previous
        return BlockCachedResult({"hidden":result},{"sum":result[:,-1:].clone()})
'''


def nested_recipe(reference: RevisionRef) -> Recipe:
    """Define two nested shared-repeat levels with explicit lexical carry."""
    recipe = tensor_recipe(reference).model_dump(mode="python")
    inner = recipe["flow"][0]
    inner["repeat"] = 2
    inner["carry"]["x"]["initial"] = "outer_input"
    inner["carry"]["x"]["result"] = "inner_result"
    recipe["flow"] = [{"repeat": 2, "carry": {"x": {"initial": "hidden", "input": "outer_input",
                       "output": "inner_result", "result": "result"}}, "body": [inner]}]
    return Recipe.model_validate(recipe)


def test_nested_shared_cached_occurrences_match_full_prefix_and_isolate_requests():
    """Actual nested cache execution keeps four same-owner occurrences distinct without state mutation."""
    from llm.architecture.interfaces import CacheIdentity, RequestCache

    with SourceLoader() as loader:
        definition = loader.load(revision(CACHE_SOURCE))
        recipe = nested_recipe(definition.reference).model_dump(mode="python")
        recipe["modules"]["scale"]["parameters"] = {}
        model = build_model(resolve_recipe(Recipe.model_validate(recipe), {}, {definition.reference.revision_id: definition}))
        identity = CacheIdentity(uuid4(), "a" * 64)
        cache = RequestCache(identity, {})
        prefix = torch.randn(2, 3, 8)
        _, cache = model.forward_cached_named({"hidden": prefix}, cache, identity=identity)
        assert len(cache.entries) == 4
        assert len({occurrence.call_path for occurrence in cache.entries}) == 4
        assert {occurrence.instance_path for occurrence in cache.entries} == {"scale"}
        for _ in range(2):
            previous = {key: value["sum"].clone() for key, value in cache.entries.items()}
            current = torch.randn(2, 1, 8)
            actual, updated = model.forward_cached_named({"hidden": current}, cache, identity=identity)
            prefix = torch.cat((prefix, current), 1)
            expected = model.forward_named({"hidden": prefix})["hidden"][:, -1:]
            torch.testing.assert_close(actual["hidden"], expected)
            for key, value in cache.entries.items():
                torch.testing.assert_close(value["sum"], previous[key])
            cache = updated
        with pytest.raises(ArchitectureError, match="another request"):
            model.forward_cached_named({"hidden": current}, cache, identity=CacheIdentity(uuid4(), "a" * 64))


def routing_source(name: str, inputs: dict, outputs: dict, expression: str) -> str:
    """Create one explicit named-port routing block with no constructor state."""
    return f'''from torch import nn
from llm.architecture.sdk import block, BlockDeclaration, TensorPort
@block(BlockDeclaration(sdk_version=1,name={name!r},title={name!r},parameters={{}},
 inputs={{{','.join(f'{key!r}:TensorPort(dtype="floating",shape=("batch","sequence","width"))' for key in inputs)}}},
 outputs={{{','.join(f'{key!r}:TensorPort(dtype="floating",shape=("batch","sequence","width"))' for key in outputs)}}}))
class Routing(nn.Module):
    """A named tensor routing fixture."""
    def __init__(self):
        """Construct no parameters or persistent state."""
        super().__init__()
    def forward(self,{','.join(inputs)}):
        """Return explicit named outputs from immutable inputs."""
        return {expression}
'''


def test_subarchitecture_branches_multiple_ports_and_nested_carry_execute_with_gradients():
    """Real execution shares a subarchitecture's children across two calls and retains every gradient."""
    with SourceLoader() as loader:
        scale = loader.load(revision(SOURCE))
        fork = loader.load(revision(routing_source("fork", {"hidden": None}, {"left": None, "right": None},
                                                   '{"left":hidden,"right":hidden*2}')))
        merge = loader.load(revision(routing_source("merge", {"left": None, "right": None}, {"hidden": None},
                                                    '{"hidden":left+right}')))
        child_ref = RevisionRef(kind="architecture", resource_id=uuid4(), revision_id=uuid4(), content_hash="a" * 64)
        child = ArchitectureDefinition(child_ref, nested_recipe(scale.reference))
        recipe = tensor_recipe(scale.reference).model_dump(mode="python")
        recipe["modules"].update({"fork": {"reference": fork.reference}, "merge": {"reference": merge.reference},
                                  "child": {"reference": child_ref}})
        recipe["flow"] = [
            {"call": "fork", "inputs": {"hidden": "hidden"}, "outputs": {"left": "left", "right": "right"}},
            {"call": "scale", "inputs": {"hidden": "left"}, "outputs": {"hidden": "a"}},
            {"call": "scale", "inputs": {"hidden": "right"}, "outputs": {"hidden": "b"}},
            {"call": "merge", "inputs": {"left": "a", "right": "b"}, "outputs": {"hidden": "merged"}},
            {"call": "child", "inputs": {"hidden": "merged"}, "outputs": {"hidden": "first"}},
            {"call": "child", "inputs": {"hidden": "first"}, "outputs": {"hidden": "result"}},
        ]
        dependencies = {item.reference.revision_id: item for item in (scale, fork, merge, child)}
        model = build_model(resolve_recipe(Recipe.model_validate(recipe), {}, dependencies))
        hidden = torch.randn(2, 3, 8)
        actual = model.forward_named({"hidden": hidden})["hidden"]
        torch.testing.assert_close(actual, hidden * 1.5 * 0.5 ** 8)
        actual.sum().backward()
        parent = model.instances["scale"].gain
        nested = model.instances["child"].instances["scale"].gain
        torch.testing.assert_close(parent.grad, (hidden * 3 * 0.5 ** 8).sum())
        torch.testing.assert_close(nested.grad, (hidden * 1.5 * 8 * 0.5 ** 7).sum())
        assert len(list(model.parameters())) == 2


def test_recursive_architecture_and_missing_constructor_inputs_are_rejected():
    """Dependency recursion and required configuration fail before any constructor runs."""
    dependencies = fixture_dependencies()
    recipe = parse_recipe((ROOT / "recipes/positive-reused-subarchitecture.yaml").read_text())
    reference = recipe.modules["branch"].reference
    recipe = recipe.model_copy(update={"modules": {"branch": recipe.modules["branch"].model_copy(update={"parameters": {}})}})
    dependencies[reference.revision_id] = ArchitectureDefinition(reference, recipe)
    with pytest.raises(ArchitectureError) as raised:
        resolve_recipe(recipe, {}, dependencies)
    assert raised.value.code == "recursive_architecture"
    with SourceLoader() as loader:
        definition = loader.load(revision(SOURCE))
        declaration = definition.declaration.model_copy(update={"parameters": {
            "gain": definition.declaration.parameters["gain"].model_copy(update={"required": True, "default": None})}})
        definition = BlockDefinition(definition.reference, declaration, definition.module_type)
        recipe = tensor_recipe(definition.reference).model_dump(mode="python")
        recipe["modules"]["scale"]["parameters"] = {}
        with pytest.raises(ArchitectureError) as raised:
            resolve_recipe(Recipe.model_validate(recipe), {}, {definition.reference.revision_id: definition})
        assert raised.value.code == "required_parameter"


def test_incompatible_alias_and_alias_cycle_are_rejected():
    """Alias establishment requires exact compatible registered Parameters and an acyclic owner graph."""
    with SourceLoader() as loader:
        scalar = loader.load(revision(SOURCE))
        vector = loader.load(revision(SOURCE.replace("torch.tensor(gain, dtype=torch.float32)",
                                                    "torch.full((2,),gain,dtype=torch.float32)")))
        recipe = alias_recipe(scalar.reference).model_dump(mode="python")
        recipe["modules"]["right"]["reference"] = vector.reference
        plan = resolve_recipe(Recipe.model_validate(recipe), {}, {item.reference.revision_id: item for item in (scalar, vector)})
        with pytest.raises(ArchitectureError) as raised:
            build_model(plan)
        assert raised.value.code == "incompatible_parameter_alias"
        recipe = alias_recipe(scalar.reference).model_dump(mode="python")
        recipe["aliases"] = (*recipe["aliases"], {"owner": "right.gain", "target": "left.gain"})
        plan = resolve_recipe(Recipe.model_validate(recipe), {}, {scalar.reference.revision_id: scalar})
        with pytest.raises(ArchitectureError, match="cyclic"):
            build_model(plan)


def test_shared_incompatible_alias_fixture_rejects_real_parameter_shapes():
    """Exercise the eighth shared negative with real parameter-only fixture modules.

    These fixtures establish alias shape contracts only; ISS-53 owns actual builtin
    arithmetic, constructor RNG order and numerical compatibility.
    """
    dependencies = fixture_dependencies()
    for key, definition in tuple(dependencies.items()):
        if not isinstance(definition, BlockDefinition):
            continue
        name = definition.declaration.name
        if name in {"token_embedding", "language_model_head"}:
            module_type = VocabularyParameters
        elif name == "transformer_layer":
            module_type = AttentionParameters
        else:
            module_type = EmptyParameters
        dependencies[key] = BlockDefinition(definition.reference, definition.declaration, module_type)
    recipe = parse_recipe((ROOT / "recipes/negative-incompatible-alias.yaml").read_text())
    plan = resolve_recipe(recipe, {}, dependencies, tokenizer())
    with pytest.raises(ArchitectureError) as raised:
        build_model(plan)
    assert raised.value.code == "incompatible_parameter_alias"


class VocabularyParameters(nn.Module):
    """Parameter-only fixture for embedding/head alias dimensions, with no numerical claim."""

    def __init__(self, vocab: int, width: int):
        """Allocate a real vocabulary-by-width registered Parameter for alias verification."""
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab, width))


class AttentionParameters(nn.Module):
    """Parameter-only fixture matching the declared fused QKV parameter shape."""

    def __init__(self, dimensions: int, **_parameters):
        """Allocate the QKV shape needed by the incompatible shared alias fixture."""
        super().__init__()
        self.attn = nn.Module()
        self.attn.wqkv = nn.Linear(dimensions, 3 * dimensions, bias=False)


class EmptyParameters(nn.Module):
    """Constructor-only fixture for unused builtin components in alias validation."""

    def __init__(self, **_parameters):
        """Construct no trainable values or forward implementation."""
        super().__init__()


def test_runtime_rejects_bad_intermediate_ports_and_persistent_python_state():
    """A later shape compensation cannot hide an invalid block or persistent container mutation."""
    with SourceLoader() as loader:
        definition = loader.load(revision(SOURCE.replace('hidden * self.gain', 'hidden[:, :, 0] * self.gain')))
        model = build_model(resolve_recipe(tensor_recipe(definition.reference), {}, {definition.reference.revision_id: definition}))
        with pytest.raises(ArchitectureError) as raised:
            model.forward_named({"hidden": torch.ones(1, 2, 8)})
        assert raised.value.code == "port_shape"
        source = SOURCE.replace("super().__init__()", "super().__init__()\n        self.history=[]")
        source = source.replace('return {"hidden": hidden * self.gain}',
                                'self.history.append(1)\n        return {"hidden": hidden * self.gain}')
        with pytest.raises(ArchitectureError) as raised:
            loader.load(revision(source))
        assert raised.value.code == "persistent_mutation"


def test_module_cache_cleanup_identity_conflicts_and_effective_source_limit():
    """Source owner cleanup is exact, revision bytes immutable and inspection imports bounded."""
    import sys
    from llm.architecture.spec import MAX_DEPENDENCIES

    pinned = revision(SOURCE)
    loader = SourceLoader()
    definition = loader.load(pinned)
    module_name = definition.module_type.__module__
    assert module_name in sys.modules
    changed = SOURCE.replace("hidden * self.gain", "hidden * self.gain + 1")
    changed_reference = pinned.reference.model_copy(update={"content_hash": hashlib.sha256(changed.encode()).hexdigest()})
    with pytest.raises(ArchitectureError) as raised:
        loader.load(SourceRevision(changed_reference, changed))
    assert raised.value.code == "revision_snapshot_mismatch"
    for _ in range(MAX_DEPENDENCIES - 1):
        loader.load(revision(SOURCE))
    with pytest.raises(ArchitectureError) as raised:
        loader.load(revision(SOURCE))
    assert raised.value.code == "dependency_limit"
    owned_names = set(loader.modules)
    loader.close()
    assert not owned_names & sys.modules.keys()


@pytest.mark.parametrize("expression", ["torch.get_default_dtype()", "torch.get_default_device()"])
def test_source_rejects_unrecorded_ambient_defaults(expression):
    """Reject ambient execution defaults even where export could silently specialize a branch."""
    source = SOURCE.replace('return {"hidden": hidden * self.gain}',
                            f'ambient = {expression}\n        return {{"hidden": hidden * self.gain}}')
    with SourceLoader() as loader, pytest.raises(ArchitectureError) as raised:
        loader.load(revision(source))
    assert raised.value.code == "unsupported_source"


def test_source_rejects_import_alias_of_ambient_default_reader():
    """The supported import policy rejects renamed default readers before executing source."""
    source = "from torch import get_default_dtype as current_dtype\n" + SOURCE
    with SourceLoader() as loader, pytest.raises(ArchitectureError) as raised:
        loader.load(revision(source))
    assert raised.value.code == "unsupported_import"


def test_recipe_snapshots_own_their_values_and_initializer_order():
    """Mutation after resolution cannot alter planned values, and module order is stable."""
    with SourceLoader() as loader:
        definition = loader.load(revision(SOURCE))
        recipe = alias_recipe(definition.reference)
        plan = resolve_recipe(recipe, {}, {definition.reference.revision_id: definition})
        recipe.modules["left"].parameters["gain"] = 0.9
        definition.declaration.parameters.clear()
        model = build_model(plan)
    assert model.instances["left"].gain.item() == 0.5
    assert tuple(model.instances) == ("left", "right")
