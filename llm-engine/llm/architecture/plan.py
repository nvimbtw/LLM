"""Resolved immutable execution records shared by the pure resolver and model builder."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from torch import nn

from llm.architecture.spec import (
    BlockDeclaration, CarryBinding, ParameterAlias, Recipe, RevisionRef, Scalar, TensorPort,
)


@dataclass(frozen=True)
class BlockDefinition:
    """Pinned module declaration/class supplied by a verified source or builtin loader."""

    reference: RevisionRef
    declaration: BlockDeclaration
    module_type: type[nn.Module]


@dataclass(frozen=True)
class ArchitectureDefinition:
    """Pinned reusable composition; resolution never looks up a mutable latest revision."""

    reference: RevisionRef
    recipe: Recipe


@dataclass(frozen=True)
class PlanCall:
    """Route immutable port bindings into one already instantiated local module path."""

    instance: str
    inputs: Mapping[str, str]
    outputs: Mapping[str, str]


@dataclass(frozen=True)
class PlanRepeat:
    """Execute fixed shared occurrences with explicit lexical carry."""

    count: int
    carry: tuple[CarryBinding, ...]
    body: tuple[PlanOperation, ...]


@dataclass(frozen=True)
class PlanStack:
    """Call distinct indexed instances while forwarding explicit named template carry."""

    instances: tuple[str, ...]
    inputs: Mapping[str, str]
    carry: tuple[CarryBinding, ...]


PlanOperation: TypeAlias = PlanCall | PlanRepeat | PlanStack


@dataclass(frozen=True)
class ResolvedInstance:
    """One owned module configuration or independently resolved subarchitecture."""

    path: str
    definition: BlockDefinition | ResolvedPlan
    parameters: Mapping[str, Scalar]


@dataclass(frozen=True)
class ResolvedPlan:
    """A bounded profile-independent model computation with all constructor bindings fixed."""

    parameters: Mapping[str, Scalar]
    inputs: Mapping[str, TensorPort]
    outputs: Mapping[str, TensorPort]
    result: Mapping[str, str]
    instances: tuple[ResolvedInstance, ...]
    operations: tuple[PlanOperation, ...]
    aliases: tuple[ParameterAlias, ...]
    expanded_calls: int
    instance_count: int
    vocabulary_size: int | None
    context_window: int | None
    interface: str
