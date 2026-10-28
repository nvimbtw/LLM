"""Data-only builtin catalog startup; real numerical acceptance belongs to ordinary supervised validation jobs."""

from uuid import NAMESPACE_URL, UUID, uuid5

import torch
import yaml
from sqlalchemy.orm import Session

import store
from api import builtin_tokenizer
from api.validation_jobs import enqueue_validation, read_tokenizer_binding
from llm.architecture.builtins.catalog import builtin_definitions, readable_source
from llm.architecture.builtins.recipes import RecipeName, recipe_template
from llm.architecture.parse import parse_recipe
from llm.architecture.resolve import bound_port, parameter_values
from llm.architecture.spec import BlockDeclaration, DraftSubject, RuntimeProfile, ShapeDomain, canonical_digest
from llm.architecture.wire import ArchitectureCreate, BlockCreate, TensorInputSpec, ValidationRequest
from store import architecture_queries as catalog
from store.schema import Architecture, BlockRevision

BUILTIN_ACTOR = uuid5(NAMESPACE_URL,"slate:builtin-catalog")
RECIPE_NAMES: tuple[RecipeName,...] = ("standard","expanded","looped","nested")


def recipe_resource_id(name: RecipeName) -> UUID:
    """Identify supplied recipe families independently from their changing pinned source bytes/revision numbers."""
    return uuid5(NAMESPACE_URL,f"slate:builtin-recipe-v1:{name}")


def example_profile() -> RuntimeProfile:
    """Bind one finite CPU train/eval concrete example domain to the exact installed torch runtime."""
    return RuntimeProfile(torch_version=str(torch.__version__),sdk_version=1,device="cpu",precision="fp32",
        capture_policy="torch-export-strict-v1",modes=("train","eval"),
        domains=(ShapeDomain(batch_min=2,batch_max=2,sequence_min=3,sequence_max=3,signatures=("full_prefix",)),))


def example_inputs(declaration: BlockDeclaration) -> dict[str,TensorInputSpec]:
    """Choose one small CPU block profile from installed declarations; these synthetic tensors claim no tokenizer identity."""
    parameters = parameter_values(declaration.parameters,{})
    symbols = {"batch":2,"sequence":3}
    result = {}
    for name,port in declaration.inputs.items():
        port = bound_port(port,parameters)
        shape = tuple(dimension if type(dimension) is int else symbols.get(dimension,8) for dimension in port.shape)
        result[name] = TensorInputSpec(dtype="float32" if port.dtype == "floating" else port.dtype,shape=shape)
    return result


def seed_blocks() -> None:
    """Seed readable builtin roots and idempotent normal-queue checks, never fabricated accepted receipts.

    Call after exclusive runtime reconciliation and before queue draining. Failed
    or pruned bootstrap checks require explicit administrator retry/new validation;
    startup does not silently replay an unsuccessful numerical operation.
    """
    profile = example_profile()
    with store.session() as db:
        for definition in builtin_definitions().values():
            reference = definition.reference
            row = catalog.seed_builtin_block(db,BUILTIN_ACTOR,reference,BlockCreate(
                name=definition.declaration.title,description="Supplied read-only builtin implementation.",
                source=readable_source(reference)))
            if db.get(BlockRevision,reference.revision_id) is not None:
                continue
            request = ValidationRequest(schema_version=1,request_id=uuid5(reference.revision_id,canonical_digest({
                "profile":profile.model_dump(mode="json"),"draft_version":row.draft_version})),
                mode="full",subject=DraftSubject(kind="draft",resource_id=row.id,draft_version=row.draft_version,
                content_hash=row.draft_hash),profile=profile,tensor_inputs=example_inputs(definition.declaration))
            try:
                enqueue_validation(db,"block",row.id,request,BUILTIN_ACTOR)
            except catalog.CatalogError as error:
                if error.code != "operation_pruned":
                    raise
                print(f"builtin catalog: validation history pruned for {reference.resource_id}; administrator validation required")


def seed_recipes() -> None:
    """Publish a genuine tokenizer and data-only recipe drafts; queue checks only when their immutable block pins exist."""
    builtin_tokenizer.ensure_baseline()
    with store.session() as db:
        for name in RECIPE_NAMES:
            source = yaml.safe_dump(recipe_template(name).model_dump(mode="json",exclude_none=True),sort_keys=True)
            catalog.seed_builtin_architecture(db,BUILTIN_ACTOR,recipe_resource_id(name),ArchitectureCreate(
                name=f"Builtin {name} Transformer",description="Supplied read-only causal architecture template.",yaml=source))
        queue_ready_recipes(db)


def queue_ready_recipes(db: Session) -> None:
    """Enqueue concrete checks after all required accepted block pins exist, in the caller's terminal/bootstrap transaction."""
    rows = [db.get(Architecture,recipe_resource_id(name)) for name in RECIPE_NAMES]
    if not any(rows):
        return
    tokenizer_id,_ = builtin_tokenizer.baseline_bytes()
    try:
        binding = read_tokenizer_binding(tokenizer_id)
    except catalog.CatalogError as error:
        if error.code not in {"not_found","tokenizer_binding"}:
            raise
        print("builtin catalog: example tokenizer is unavailable; recipe checks await startup reconciliation")
        return
    for row in rows:
        if row is None:
            continue
        if row.origin != "builtin":
            raise catalog.CatalogError("identity_conflict","supplied recipe UUID belongs to a custom resource")
        recipe = parse_recipe(row.draft_yaml)
        pins = catalog.recipe_dependencies(recipe)
        if any(db.get(BlockRevision,reference.revision_id) is None for reference in pins.values()):
            continue
        if catalog.architecture_revision_for_source(db,row.id,row.draft_hash,catalog.dependency_digest(pins)) is not None:
            continue
        request = ValidationRequest(schema_version=1,request_id=uuid5(row.id,canonical_digest({
            "source":row.draft_hash,"draft_version":row.draft_version,"tokenizer":binding.model_dump(mode="json"),
            "profile":example_profile().model_dump(mode="json")})),mode="full",
            subject=DraftSubject(kind="draft",resource_id=row.id,draft_version=row.draft_version,content_hash=row.draft_hash),
            tokenizer=binding,profile=example_profile())
        try:
            enqueue_validation(db,"architecture",row.id,request,BUILTIN_ACTOR)
        except catalog.CatalogError as error:
            if error.code != "operation_pruned":
                raise
            print(f"builtin catalog: validation history pruned for {row.id}; administrator validation required")
