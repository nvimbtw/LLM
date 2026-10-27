"""Pure immutable recipe pin identities shared by catalog and self-contained packages."""

from llm.architecture.errors import ArchitectureError
from llm.architecture.spec import MAX_DEPENDENCIES, MAX_NESTING, Recipe, Repeat, RevisionRef, Stack, canonical_digest


def recipe_dependencies(recipe: Recipe) -> dict[str, RevisionRef]:
    """Collect bounded direct module/stack pins without source imports or loop expansion."""
    pins = {f"modules/{name}": declaration.reference for name, declaration in recipe.modules.items()}
    pending = [(recipe.flow, "flow", 0)]
    while pending:
        operations, prefix, depth = pending.pop()
        if depth > MAX_NESTING:
            raise ArchitectureError("nesting_limit", "recipe nesting exceeds dependency budget")
        for index, operation in enumerate(operations):
            path = f"{prefix}/{index}"
            if isinstance(operation, Stack):
                pins[f"{path}/template"] = operation.template.reference
            elif isinstance(operation, Repeat):
                pending.append((operation.body, f"{path}/body", depth + 1))
        if len(pins) > MAX_DEPENDENCIES:
            raise ArchitectureError("dependency_limit", "too many direct recipe dependencies")
    return pins


def dependency_digest(pins: dict[str, RevisionRef]) -> str:
    """Hash exact path-to-revision pins independently of catalog names and archive state."""
    return canonical_digest({name: reference.model_dump(mode="json") for name, reference in sorted(pins.items())})
