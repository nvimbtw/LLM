"""Immutable content-pinned builtin snapshots for catalog and package consumers.

No database registration or custom-source import occurs here. A changed pinned
implementation creates new revision identities instead of replacing old bytes.
"""

import hashlib
import inspect
from pathlib import Path
from uuid import NAMESPACE_URL, UUID, uuid5

from llm.architecture.builtins import (
    Add, CausalAttention, Concat, FeedForwardBlock, Normalization, OutputProjection,
    StandardTransformer, TokenEmbedding, TransformerLayer,
)
from llm.architecture.plan import BlockDefinition
from llm.architecture.errors import ArchitectureError
from llm.architecture.sdk import declaration_of
from llm.architecture.spec import RevisionRef, canonical_digest
from llm.model import transformer

BUILTIN_ABI = "slate-transformer-v1"
BLOCK_TYPES = (StandardTransformer, TokenEmbedding, TransformerLayer, Normalization,
               OutputProjection, CausalAttention, FeedForwardBlock, Add, Concat)


def compatibility_manifest() -> dict:
    """Return owned exact source hashes and ABI/SDK versions for this installed builtin set."""
    paths = {Path(inspect.getfile(module_type)) for module_type in BLOCK_TYPES}
    paths.add(Path(inspect.getfile(transformer)))
    sources = {str(path.relative_to(Path(__file__).resolve().parents[3])):
               hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(paths)}
    return {"abi": BUILTIN_ABI, "sdk_version": 1, "sources": sources}


def builtin_definitions() -> dict[str, BlockDefinition]:
    """Return fresh immutable-reference/declaration snapshots for read-only builtin resources."""
    manifest = compatibility_manifest()
    definitions = {}
    for module_type in BLOCK_TYPES:
        declaration = declaration_of(module_type)
        digest = canonical_digest({"manifest": manifest, "declaration": declaration.model_dump(mode="json")})
        resource_id = uuid5(NAMESPACE_URL, f"slate:builtin:{BUILTIN_ABI}:{declaration.name}")
        reference = RevisionRef(kind="block", resource_id=resource_id,
                                revision_id=uuid5(resource_id, digest), content_hash=digest)
        definitions[declaration.name] = BlockDefinition(reference, declaration, module_type)
    return definitions


def pinned_builtins(references: tuple[RevisionRef,...]) -> dict[UUID,BlockDefinition]:
    """Require exact installed implementation pins before any source that imports public builtins executes."""
    installed = {value.reference.revision_id:value for value in builtin_definitions().values()}
    result = {}
    for reference in references:
        definition = installed.get(reference.revision_id)
        if definition is None or definition.reference != reference:
            raise ArchitectureError("builtin_compatibility","installed builtin implementation differs from the frozen reference")
        result[reference.revision_id] = definition
    return result


def readable_source(reference: RevisionRef) -> str:
    """Read the exact installed Python implementation for a verified builtin pin, without constructing a model."""
    definition = pinned_builtins((reference,))[reference.revision_id]
    return Path(inspect.getfile(definition.module_type)).read_text(encoding="utf-8")


def verified_builtin(reference: RevisionRef, source: str) -> BlockDefinition:
    """Resolve a builtin root only when both its implementation manifest and frozen readable source match."""
    definition = pinned_builtins((reference,))[reference.revision_id]
    if readable_source(reference) != source:
        raise ArchitectureError("builtin_compatibility","frozen builtin inspection source differs from the installed implementation")
    return definition


def copyable_source(name: str) -> str:
    """Return eligible SDK/public-builtin wrapper source for a new custom block revision.

    This exact UTF-8 source receives its own source hash and resource/revision
    identities when saved. Its public builtin dependency must retain this
    installation's compatibility manifest. Copies expose the full-prefix baseline;
    they do not advertise undelegated optional child capabilities.
    """
    definitions = builtin_definitions()
    if name not in definitions:
        raise ValueError("unknown builtin block")
    definition = definitions[name]
    declaration = definition.declaration.model_dump(mode="python")
    declaration["capabilities"] = ()
    parameters = tuple(declaration["parameters"])
    inputs = tuple(declaration["inputs"])
    constructor_arguments = ", ".join(("self", *parameters))
    forward_arguments = ", ".join(("self", *inputs))
    construction = ", ".join(f"{key}={key}" for key in parameters)
    forwarding = ", ".join(f"{key}={key}" for key in inputs)
    return (
        '"""Editable custom wrapper around a version-pinned public builtin dependency."""\n\n'
        "from torch import nn\n"
        "from llm.architecture.sdk import block, BlockDeclaration\n"
        f"from llm.architecture.builtins import {definition.module_type.__name__}\n\n\n"
        f"@block(BlockDeclaration.model_validate({declaration!r}))\n"
        "class CustomBlock(nn.Module):\n"
        '    """Copyable full-prefix computation with its own module/parameter ownership."""\n\n'
        f"    def __init__({constructor_arguments}):\n"
        '        """Construct the configured public builtin once for this custom instance."""\n'
        "        super().__init__()\n"
        f"        self.implementation = {definition.module_type.__name__}({construction})\n\n"
        f"    def forward({forward_arguments}):\n"
        '        """Return declared named tensors without retaining caller/request state."""\n'
        f"        return self.implementation({forwarding})\n"
    )
