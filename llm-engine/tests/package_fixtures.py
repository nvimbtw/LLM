"""Synthetic retained evidence and complete source snapshots for package contract tests."""

import hashlib
from uuid import uuid4

import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel

from llm.architecture.builtins.catalog import builtin_definitions, compatibility_manifest, copyable_source
from llm.architecture.builtins.recipes import recipe_template
from llm.architecture.dependencies import dependency_digest, recipe_dependencies
from llm.architecture.graphs import GraphArtifacts, export_guards
from llm.architecture.package_integrity import encode_json
from llm.architecture.resolve import parameter_values
from llm.architecture.spec import (
    DefinitionPackage, FinalTokenizer, ReceiptIdentity, ResolvedDefinition, RevisionRef,
    RevisionSubject, RuntimeProfile, ShapeDomain, SourceManifestEntry, ValidationReceipt, canonical_digest,
)


def payload(name: str = "looped", *, custom: bool = False, scale: int = 1) -> dict:
    """Freeze real builtin pins, exact editable custom source and explicitly synthetic graph evidence."""
    recipe = recipe_template(name)
    installed = builtin_definitions()
    builtin_manifest = compatibility_manifest()
    compatibility = canonical_digest(builtin_manifest)
    custom_source = None
    if custom:
        custom_source = copyable_source("normalization")
        if scale != 1:
            custom_source = custom_source.replace("return self.implementation(hidden=hidden)",
                f'return {{"hidden": self.implementation(hidden=hidden)["hidden"] * {float(scale)}}}')
        reference = RevisionRef(kind="block",resource_id=uuid4(),revision_id=uuid4(),
                                content_hash=hashlib.sha256(custom_source.encode()).hexdigest())
        raw = recipe.model_dump(mode="json")
        raw["modules"]["final_norm"]["reference"] = reference.model_dump(mode="json")
        recipe = type(recipe).model_validate(raw)
    entries, files = [], {}
    for reference in {value.revision_id: value for value in recipe_dependencies(recipe).values()}.values():
        if custom_source and reference == recipe.modules["final_norm"].reference:
            data = custom_source.encode()
            path = f"sources/{reference.revision_id}.py"
        else:
            builtin = next(value for value in installed.values() if value.reference == reference)
            data = encode_json({"manifest": builtin_manifest, "declaration": builtin.declaration.model_dump(mode="json")})
            path = f"sources/{reference.revision_id}.builtin.json"
        entries.append(SourceManifestEntry(reference=reference,path=path,bytes=len(data),sdk_version=1,builtin_compatibility=compatibility))
        files[path] = data.decode()
    source = yaml.safe_dump(recipe.model_dump(mode="json"), sort_keys=False).encode("utf-8")
    root = RevisionRef(kind="architecture",resource_id=uuid4(),revision_id=uuid4(),content_hash=hashlib.sha256(source).hexdigest())
    tokenizer = Tokenizer(WordLevel(vocab={str(index): index for index in range(24)},unk_token="0")).to_str().encode()
    binding = FinalTokenizer(kind="final",tokenizer_id=uuid4(),fingerprint=hashlib.sha256(tokenizer).hexdigest(),vocabulary_size=24)
    parameters = parameter_values(recipe.parameters, {"dimensions":16,"heads":2,"ff_multiplier":2,
        "context_window":12,"rope_base":10000.0,"learned_temperature":False})
    definition = ResolvedDefinition(schema_version=1,architecture=root,recipe=recipe,dependencies=tuple(entries),
        parameters=parameters,tokenizer=binding,context_window=12,aliases=recipe.aliases,sdk_version=1)
    identity = ReceiptIdentity(subject=RevisionSubject(kind="revision",reference=root),
        dependency_digest=dependency_digest(recipe_dependencies(recipe)),parameter_digest=canonical_digest(parameters),
        input_profile_digest="c"*64,tokenizer=binding,context_window=12,
        profile=RuntimeProfile(torch_version=torch.__version__,sdk_version=1,device="cpu",precision="fp32",
            capture_policy="torch-export-strict-v1",domains=(ShapeDomain(batch_min=1,batch_max=2,sequence_min=1,sequence_max=12,signatures=("full_prefix",)),),modes=("train","eval")))
    graphs = GraphArtifacts("")
    for mode in identity.profile.modes:
        graphs.add(view="operations",mode=mode,signature="full_prefix",scope="root",
            nodes=({"id":"fixture","kind":"input","label":"Synthetic package fixture","occurrence_path":"fixture"},),
            edges=(),strict=True,guards=())
    identity = identity.model_copy(update={"profile":identity.profile.model_copy(update={"export_guards":export_guards(graphs.entries)})})
    graphs.identity_hash = canonical_digest(identity)
    graph_hash, blobs = graphs.finish()
    receipt = ValidationReceipt(schema_version=1,receipt_id=uuid4(),identity=identity,
        identity_hash=canonical_digest(identity),verdict="valid",graph_manifest_hash=graph_hash)
    manifest = DefinitionPackage(format="slate.modular-model",format_version=1,definition_id=canonical_digest(definition),
        definition=definition,receipt_ids=(receipt.receipt_id,),graph_hashes=tuple(blobs),training_state_path="train_state.pt")
    files.update({"architecture.yaml":source.decode(),"tokenizer.json":tokenizer.decode(),
                  "meta.json":'{"result":{"parameters":123}}',f"receipts/{receipt.receipt_id}.json":encode_json(receipt).decode()})
    files.update({f"graphs/{digest}.json":source for digest, source in blobs.items()})
    return {"manifest":manifest.model_dump(mode="json"),"files":files}
