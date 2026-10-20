"""Verified revision-local block imports, callable only inside an admitted worker.

Eligibility is a conservative supported-code contract, not a Python sandbox.
The process owner still bounds import/construction time and host/device memory.
"""

import ast
import hashlib
import sys
import types
from dataclasses import dataclass
from uuid import uuid4

from torch import nn

from llm.architecture.errors import ArchitectureError
from llm.architecture.plan import BlockDefinition
from llm.architecture.sdk import DECLARATION_ATTRIBUTE, declaration_of
from llm.architecture.spec import MAX_DEPENDENCIES, MAX_SOURCE_BYTES, RevisionRef

ALLOWED_IMPORTS = frozenset({
    "torch", "torch.nn", "torch.nn.functional", "torch.nn.init", "math", "typing",
    "torch.utils.checkpoint", "llm.architecture.sdk", "llm.architecture.interfaces",
    "llm.architecture.builtins",
})
FORBIDDEN_CALLS = frozenset({
    "eval", "exec", "open", "compile", "__import__", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "input", "print", "breakpoint",
})
FORBIDDEN_MEMBERS = frozenset({
    "save", "load", "load_state_dict", "state_dict", "from_file", "numpy", "item", "tolist",
    "manual_seed", "seed", "set_rng_state", "set_default_device", "set_default_dtype",
    "register_forward_hook", "register_forward_pre_hook", "register_backward_hook",
    "register_full_backward_hook", "compile", "synchronize", "set_device", "data",
    "hub", "distributed", "multiprocessing", "cpp_extension", "__dict__", "__class__",
    "os", "sys", "subprocess", "importlib", "builtins", "inspect",
})


@dataclass(frozen=True)
class SourceRevision:
    """Exact source bytes and their immutable catalog/package reference."""

    reference: RevisionRef
    source: str


def immutable_literal(node: ast.AST) -> bool:
    """Permit scalar/tuple configuration constants, refusing ambient mutable containers."""
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError):
        return False
    return value is None or type(value) in (str, int, float, bool) or (
        isinstance(value, tuple) and all(immutable_literal(ast.Constant(item)) for item in value)
    )


def check_source_eligibility(source: str) -> ast.Module:
    """Reject unsupported imports, ambient effects/state and persistent forward mutation."""
    if len(source.encode("utf-8")) > MAX_SOURCE_BYTES:
        raise ArchitectureError("source_byte_limit", "block source exceeds byte budget")
    try:
        tree = ast.parse(source)
    except SyntaxError as error:
        raise ArchitectureError("invalid_source", "block source has invalid Python syntax") from error
    for statement in tree.body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if isinstance(statement, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
            continue
        value = statement.value if isinstance(statement, (ast.Assign, ast.AnnAssign)) else None
        if value is None or not immutable_literal(value):
            raise ArchitectureError("unsupported_source", "module scope permits imports, definitions and immutable literals")
    checker = SourceEligibility()
    checker.visit(tree)
    return tree


class SourceEligibility(ast.NodeVisitor):
    """Check explicit source policy before importing any custom revision bytes."""

    def __init__(self):
        """Start outside constructors so mutable state is permitted only during initialization."""
        self.constructing = False
        self.function_depth = 0

    def visit_Import(self, node: ast.Import) -> None:
        """Require exact supported environment/SDK import modules."""
        if self.function_depth or any(alias.name not in ALLOWED_IMPORTS for alias in node.names):
            raise ArchitectureError("unsupported_import", "import is outside the supported SDK environment")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Reject relative, wildcard and service/storage imports before execution."""
        if self.function_depth or node.level or node.module not in ALLOWED_IMPORTS or any(
            alias.name == "*" or alias.name in FORBIDDEN_MEMBERS for alias in node.names
        ):
            raise ArchitectureError("unsupported_import", "import is outside the supported SDK environment")

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Refuse metaclasses/dunder hooks and executable class-scope state."""
        if node.keywords:
            raise ArchitectureError("unsupported_source", "custom metaclasses are unsupported")
        if any(not isinstance(base, (ast.Name, ast.Attribute)) for base in node.bases):
            raise ArchitectureError("unsupported_source", "dynamic class bases are unsupported")
        if any(not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Name)
               or decorator.func.id != "block" for decorator in node.decorator_list):
            raise ArchitectureError("unsupported_source", "only the SDK block decorator is supported")
        for item in node.body:
            if isinstance(item, (ast.Assign, ast.AnnAssign)) and not immutable_literal(item.value):
                raise ArchitectureError("ambient_state", "class constants must be immutable literals")
            if not isinstance(item, (ast.FunctionDef, ast.Assign, ast.AnnAssign, ast.Pass)) and not (
                isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant)
            ):
                raise ArchitectureError("unsupported_source", "class scope permits methods and immutable constants")
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Limit hooks/decorators and track constructor-only persistent assignments."""
        if node.name.startswith("__") and node.name != "__init__":
            raise ArchitectureError("unsupported_source", "custom Python lifecycle hooks are unsupported")
        if node.name in {"train", "to", "parameters", "named_parameters", "buffers", "named_buffers",
                         "state_dict", "load_state_dict", "apply", "_apply", "get_submodule", "get_parameter"}:
            raise ArchitectureError("unsupported_source", "overriding PyTorch lifecycle methods is unsupported")
        if node.decorator_list:
            raise ArchitectureError("unsupported_source", "custom method/function decorators are unsupported")
        if any(not immutable_literal(default) for default in (*node.args.defaults, *node.args.kw_defaults)
               if default is not None):
            raise ArchitectureError("ambient_state", "function defaults must be immutable literals")
        constructing = self.constructing
        self.constructing = node.name == "__init__"
        self.function_depth += 1
        self.generic_visit(node)
        self.function_depth -= 1
        self.constructing = constructing

    def visit_Call(self, node: ast.Call) -> None:
        """Reject reflection/I/O/untracked scalar reads and forward registration changes."""
        name = node.func.id if isinstance(node.func, ast.Name) else (
            node.func.attr if isinstance(node.func, ast.Attribute) else None
        )
        if name in FORBIDDEN_CALLS or name in FORBIDDEN_MEMBERS:
            raise ArchitectureError("unsupported_source", f"unsupported source operation: {name}")
        scalar_read = isinstance(node.func, ast.Name) and name in {"int", "float", "bool", "str", "repr", "complex"}
        if not self.constructing and (scalar_read or name in {"register_buffer", "register_parameter", "train", "eval", "requires_grad_"}):
            raise ArchitectureError("python_specialization", "forward cannot extract scalars or register state")
        if not self.constructing and isinstance(node.func, ast.Attribute):
            root = node.func.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "self" and name in {
                "append", "extend", "insert", "pop", "clear", "update", "setdefault", "remove", "sort", "reverse",
            }:
                raise ArchitectureError("persistent_mutation", "forward cannot mutate persistent Python containers")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Reject forbidden ambient APIs and persistent attribute writes outside construction."""
        if node.attr in FORBIDDEN_MEMBERS or node.attr.startswith("__") and node.attr != "__init__":
            raise ArchitectureError("unsupported_source", f"unsupported attribute: {node.attr}")
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, ast.Attribute):
                root = root.value
            if not self.constructing or not isinstance(root, ast.Name) or root.id != "self":
                raise ArchitectureError("persistent_mutation", "only constructor-owned attributes may be assigned")
        self.generic_visit(node)

    def visit_Global(self, node: ast.Global) -> None:
        """Reject writes to unrecorded ambient Python globals."""
        raise ArchitectureError("ambient_state", "global state is unsupported")

    def visit_Subscript(self, node: ast.Subscript) -> None:
        """Reject assignments through persistent Python containers and buffer indexing."""
        if not self.constructing and isinstance(node.ctx, (ast.Store, ast.Del)):
            root = node.value
            while isinstance(root, (ast.Attribute, ast.Subscript)):
                root = root.value
            if isinstance(root, ast.Name) and root.id == "self":
                raise ArchitectureError("persistent_mutation", "forward cannot assign persistent indexed values")
        self.generic_visit(node)

    def visit_Nonlocal(self, node: ast.Nonlocal) -> None:
        """Reject persistent closure state outside explicit tensor carry."""
        raise ArchitectureError("ambient_state", "nonlocal state is unsupported")


class SourceLoader:
    """Own revision imports for the model lifetime, including strict capture/compilation."""

    def __init__(self):
        """Create a private import namespace and bounded revision cache."""
        self.namespace = uuid4().hex
        self.loaded: dict[tuple, BlockDefinition] = {}
        self.modules: dict[str, types.ModuleType] = {}
        self.revisions: dict = {}

    def load(self, revision: SourceRevision) -> BlockDefinition:
        """Verify and inspect source without construction; retain revision-local class identity."""
        reference = revision.reference
        digest = hashlib.sha256(revision.source.encode("utf-8")).hexdigest()
        if reference.kind != "block" or reference.content_hash != digest:
            raise ArchitectureError("source_hash_mismatch", "source bytes do not match block revision")
        if reference.revision_id in self.revisions and self.revisions[reference.revision_id] != reference:
            raise ArchitectureError("revision_snapshot_mismatch", "one revision identity cannot name conflicting bytes/resources")
        key = (reference.resource_id, reference.revision_id, digest)
        if key in self.loaded:
            return self.loaded[key]
        if len(self.loaded) >= MAX_DEPENDENCIES:
            raise ArchitectureError("dependency_limit", "source dependency budget exceeded")
        tree = check_source_eligibility(revision.source)
        name = f"_slate_block_{reference.revision_id.hex}_{digest}_{self.namespace}"
        module = types.ModuleType(name)
        sys.modules[name] = module
        try:
            exec(compile(tree, f"blocks/{reference.revision_id}.py", "exec"), module.__dict__)
            classes = [value for value in module.__dict__.values() if isinstance(value, type)
                       and issubclass(value, nn.Module) and value.__module__ == name
                       and DECLARATION_ATTRIBUTE in value.__dict__]
            if len(classes) != 1:
                raise ArchitectureError("invalid_block", "source must declare exactly one SDK block")
            definition = BlockDefinition(reference, declaration_of(classes[0]), classes[0])
        except BaseException:
            sys.modules.pop(name, None)
            raise
        self.modules[name] = module
        self.loaded[key] = definition
        self.revisions[reference.revision_id] = reference
        return definition

    def close(self) -> None:
        """Release this owner after model unload; eager classes survive but capture needs live imports."""
        for name, module in self.modules.items():
            if sys.modules.get(name) is module:
                del sys.modules[name]
        self.modules.clear()
        self.loaded.clear()
        self.revisions.clear()

    def __enter__(self) -> "SourceLoader":
        """Return the owner for bounded inspection/build scopes."""
        return self

    def __exit__(self, *_exc: object) -> None:
        """Release import registrations after success or failure."""
        self.close()
