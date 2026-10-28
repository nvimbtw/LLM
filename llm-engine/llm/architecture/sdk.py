"""Small revision-local block declaration API; importing it never registers global blocks."""

import inspect
from collections.abc import Callable
from typing import TypeVar

from torch import nn

from llm.architecture.errors import ArchitectureError
from llm.architecture.spec import BlockDeclaration, ParameterDeclaration, TensorPort

ModuleType = TypeVar("ModuleType", bound=type[nn.Module])
DECLARATION_ATTRIBUTE = "__slate_block_declaration__"


def checked_arguments(function: Callable, declared: set[str], purpose: str) -> None:
    """Require explicit named constructor/forward parameters matching their declaration."""
    arguments = list(inspect.signature(function).parameters.values())
    parameters = arguments[1:]
    if {value.name for value in parameters} != declared or any(
        value.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD,
                       inspect.Parameter.POSITIONAL_ONLY) for value in parameters
    ):
        raise ArchitectureError("interface_signature", f"{purpose} arguments must match declared names")


def block(declaration: BlockDeclaration) -> Callable[[ModuleType], ModuleType]:
    """Attach owned metadata to a module class without constructing it or global registration."""
    snapshot = BlockDeclaration.model_validate(declaration.model_dump(mode="python"))

    def decorate(module_type: ModuleType) -> ModuleType:
        """Validate one module class and preserve it with revision-local declaration metadata."""
        if not issubclass(module_type, nn.Module):
            raise ArchitectureError("invalid_block", "a block must subclass torch.nn.Module")
        if DECLARATION_ATTRIBUTE in module_type.__dict__:
            raise ArchitectureError("duplicate_declaration", "block class was declared twice")
        checked_arguments(module_type.__init__, set(snapshot.parameters), "constructor")
        checked_arguments(module_type.forward, set(snapshot.inputs), "forward")
        setattr(module_type, DECLARATION_ATTRIBUTE, snapshot)
        return module_type

    return decorate


def declaration_of(module_type: type[nn.Module]) -> BlockDeclaration:
    """Return an owned declaration snapshot for an explicitly decorated class."""
    declaration = module_type.__dict__.get(DECLARATION_ATTRIBUTE)
    if not isinstance(declaration, BlockDeclaration):
        raise ArchitectureError("missing_declaration", "block class has no SDK declaration")
    return declaration.model_copy(deep=True)
