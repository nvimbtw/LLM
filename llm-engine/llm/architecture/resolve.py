"""Pure bounded recipe resolution; dependencies are immutable injected snapshots."""

from collections.abc import Mapping
from uuid import UUID

from llm.architecture.errors import ArchitectureError
from llm.architecture.plan import (
    ArchitectureDefinition, BlockDefinition, PlanCall, PlanRepeat, PlanStack,
    ResolvedInstance, ResolvedPlan,
)
from llm.architecture.spec import (
    MAX_DEPENDENCIES, MAX_EXPANDED_CALLS, MAX_MODULES, MAX_NESTING,
    Call, CarryBinding, FinalTokenizer, ModuleDeclaration, ParameterDeclaration,
    Recipe, Repeat, Scalar, Stack, TensorPort,
)

Dependency = BlockDefinition | ArchitectureDefinition


def parameter_values(
    declarations: Mapping[str, ParameterDeclaration], supplied: Mapping[str, Scalar],
) -> dict[str, Scalar]:
    """Resolve explicit/default constructor values without coercion or implicit extra fields."""
    if supplied.keys() - declarations.keys():
        raise ArchitectureError("unknown_parameter", "unknown constructor/public parameter")
    values = {}
    for name, declaration in declarations.items():
        value = supplied.get(name, declaration.default)
        if value is None:
            raise ArchitectureError("required_parameter", f"parameter {name} requires an explicit value")
        try:
            values[name] = declaration.validate_value(value)
        except ValueError as error:
            raise ArchitectureError("parameter_value", f"invalid value for {name}") from error
    return values


def resolve_value(value: Scalar, parameters: Mapping[str, Scalar], tokenizer: FinalTokenizer | None) -> Scalar:
    """Resolve one whole declared parameter or authoritative tokenizer binding without eval."""
    if not isinstance(value, str) or not value.startswith("${"):
        return value
    name = value[2:-1]
    if name.startswith("tokenizer."):
        if tokenizer is None:
            raise ArchitectureError("pending_tokenizer", "final tokenizer bindings are required for construction")
        key = name[len("tokenizer."):]
        if key == "vocabulary_size":
            return tokenizer.vocabulary_size
        token_name = key[:-3] if key.endswith("_id") else key
        if token_name not in tokenizer.reserved_token_ids:
            raise ArchitectureError("tokenizer_binding", "unknown tokenizer token ID binding")
        return tokenizer.reserved_token_ids[token_name]
    if name not in parameters:
        raise ArchitectureError("unresolved_parameter", f"unknown parameter reference {name}")
    return parameters[name]


def positive_count(value: Scalar) -> int:
    """Bound actual fixed loop work after resolving a public parameter value."""
    if type(value) is not int or not 1 <= value <= MAX_EXPANDED_CALLS:
        raise ArchitectureError("expanded_work_limit", "fixed count exceeds expanded work budget")
    return value


def bound_port(port: TensorPort, parameters: Mapping[str, Scalar]) -> TensorPort:
    """Bind known positive dimension symbols while retaining profile-specific unknown axes."""
    shape = tuple(parameters.get(axis, axis) if isinstance(axis, str) else axis for axis in port.shape)
    if any(type(axis) is not int and not isinstance(axis, str) or type(axis) is int and axis < 1 for axis in shape):
        raise ArchitectureError("port_shape", "port dimensions must resolve to positive integers or symbols")
    return TensorPort(dtype=port.dtype, shape=shape)


def compatible_port(actual: TensorPort, expected: TensorPort, symbols: dict[str, int | str]) -> None:
    """Unify named call dimensions and refuse incompatible ranks, literal sizes and dtypes."""
    floating = {"floating", "float32", "float16", "bfloat16"}
    if actual.dtype != expected.dtype and not (actual.dtype in floating and expected.dtype in floating
                                               and "floating" in {actual.dtype, expected.dtype}):
        raise ArchitectureError("port_dtype", "connected tensor dtypes are incompatible")
    if len(actual.shape) != len(expected.shape):
        raise ArchitectureError("port_shape", "connected tensor ranks are incompatible")
    for got, wanted in zip(actual.shape, expected.shape):
        if isinstance(wanted, str):
            wanted = symbols.setdefault(wanted, got)
        if type(got) is int and type(wanted) is int and got != wanted:
            raise ArchitectureError("port_shape", "connected tensor dimensions are incompatible")


def introduce(environment: dict[str, TensorPort], name: str, port: TensorPort) -> None:
    """Introduce one lexical SSA value, rejecting writes to a visible value."""
    if name in environment:
        raise ArchitectureError("duplicate_value", f"lexical value {name} already exists")
    environment[name] = port


def read_value(environment: Mapping[str, TensorPort], name: str, code: str = "unresolved_value") -> TensorPort:
    """Read a visible lexical tensor contract or report the exact unresolved name."""
    if name not in environment:
        raise ArchitectureError(code, f"lexical value {name} is not initialized")
    return environment[name]


class Resolver:
    """Own one bounded dependency/instance resolution lifetime with no allocation or imports."""

    def __init__(self, dependencies: Mapping[UUID, Dependency], tokenizer: FinalTokenizer | None):
        """Snapshot injected dependencies and bind final tokenizer facts for this resolution."""
        if len(dependencies) > MAX_DEPENDENCIES:
            raise ArchitectureError("dependency_limit", "dependency closure exceeds budget")
        self.dependencies = dict(dependencies)
        self.tokenizer = tokenizer
        self.total_instances = 0

    def dependency(self, declaration: ModuleDeclaration) -> Dependency:
        """Require the full revision/resource/hash pin, never a mutable latest lookup."""
        dependency = self.dependencies.get(declaration.reference.revision_id)
        if dependency is None or dependency.reference != declaration.reference:
            raise ArchitectureError("revision_snapshot_mismatch", "dependency revision snapshot does not match recipe")
        if isinstance(dependency, BlockDefinition) != (declaration.reference.kind == "block"):
            raise ArchitectureError("revision_snapshot_mismatch", "dependency kind does not match recipe")
        return dependency

    def instance(
        self, path: str, declaration: ModuleDeclaration, parameters: Mapping[str, Scalar],
        ancestry: tuple[UUID, ...], depth: int,
    ) -> ResolvedInstance:
        """Resolve one module/subarchitecture instance after validating its exact constructor values."""
        self.total_instances += 1
        if self.total_instances > MAX_MODULES:
            raise ArchitectureError("module_limit", "effective module instances exceed budget")
        dependency = self.dependency(declaration)
        supplied = {name: resolve_value(value, parameters, self.tokenizer)
                    for name, value in declaration.parameters.items()}
        if isinstance(dependency, ArchitectureDefinition):
            revision = dependency.reference.revision_id
            if revision in ancestry:
                raise ArchitectureError("recursive_architecture", "recursive subarchitecture dependency")
            plan = self.recipe(dependency.recipe, supplied, ancestry + (revision,), depth + 1)
            return ResolvedInstance(path, plan, dict(plan.parameters))
        values = parameter_values(dependency.declaration.parameters, supplied)
        if self.tokenizer is not None and any(values.get(name, self.tokenizer.vocabulary_size)
                                               != self.tokenizer.vocabulary_size
                                               for name in ("vocab", "vocab_size", "vocabulary_size")):
            raise ArchitectureError("tokenizer_binding_mismatch", "block vocabulary disagrees with tokenizer")
        return ResolvedInstance(path, dependency, values)

    def recipe(
        self, recipe: Recipe, supplied: Mapping[str, Scalar], ancestry: tuple[UUID, ...], depth: int,
    ) -> ResolvedPlan:
        """Resolve a complete local composition and recursively bounded dependency closure."""
        if depth > MAX_NESTING:
            raise ArchitectureError("nesting_limit", "composition nesting exceeds budget")
        recipe = recipe.model_copy(deep=True)
        parameters = parameter_values(recipe.parameters, supplied)
        context = resolve_value(recipe.context_window, parameters, self.tokenizer) if recipe.context_window is not None else None
        if context is not None and (type(context) is not int or context < 1):
            raise ArchitectureError("context_binding", "context must resolve to one positive integer")
        if recipe.interface == "causal_lm":
            if self.tokenizer is None:
                raise ArchitectureError("pending_tokenizer", "causal models require a finalized tokenizer")
            if recipe.inputs["tokens"].dtype != "int64" or len(recipe.inputs["tokens"].shape) != 2:
                raise ArchitectureError("causal_input_dtype", "causal input requires rank-two int64 tokens")
            if recipe.outputs["logits"].dtype not in {"floating", "float32", "float16", "bfloat16"} or len(recipe.outputs["logits"].shape) != 3:
                raise ArchitectureError("causal_output", "causal output requires rank-three floating logits")
        dimensions = {**parameters, **({"vocabulary": self.tokenizer.vocabulary_size} if self.tokenizer else {})}
        inputs = {name: bound_port(port, dimensions) for name, port in recipe.inputs.items()}
        outputs = {name: bound_port(port, dimensions) for name, port in recipe.outputs.items()}
        instances = {name: self.instance(name, declaration, parameters, ancestry, depth)
                     for name, declaration in sorted(recipe.modules.items())}
        environment = dict(inputs)
        operations, calls = self.flow(recipe.flow, environment, instances, parameters, ancestry, depth)
        for name, value in recipe.result.items():
            compatible_port(read_value(environment, value), outputs[name], {})
        if calls > MAX_EXPANDED_CALLS:
            raise ArchitectureError("expanded_work_limit", "expanded model calls exceed budget")
        count = sum(1 + (instance.definition.instance_count if isinstance(instance.definition, ResolvedPlan) else 0)
                    for instance in instances.values())
        return ResolvedPlan(parameters, inputs, outputs, dict(recipe.result), tuple(instances.values()),
                            operations, recipe.aliases, calls, count,
                            self.tokenizer.vocabulary_size if self.tokenizer else None, context, recipe.interface)

    def call(
        self, operation: Call, environment: dict[str, TensorPort], instance: ResolvedInstance,
    ) -> tuple[PlanCall, int]:
        """Check complete named ports and introduce outputs from one resolved module occurrence."""
        definition = instance.definition
        inputs = definition.inputs if isinstance(definition, ResolvedPlan) else definition.declaration.inputs
        outputs = definition.outputs if isinstance(definition, ResolvedPlan) else definition.declaration.outputs
        if set(operation.inputs) != set(inputs) or set(operation.outputs) != set(outputs):
            raise ArchitectureError("port_names", "call port names do not match pinned interface")
        symbols = dict(instance.parameters)
        if self.tokenizer:
            symbols["vocabulary"] = self.tokenizer.vocabulary_size
        for port, value in operation.inputs.items():
            compatible_port(read_value(environment, value), bound_port(inputs[port], symbols), symbols)
        for port, value in operation.outputs.items():
            introduce(environment, value, bound_port(outputs[port], symbols))
        calls = definition.expanded_calls if isinstance(definition, ResolvedPlan) else 1
        return PlanCall(instance.path, dict(operation.inputs), dict(operation.outputs)), calls

    def flow(
        self, flow: tuple, environment: dict[str, TensorPort], instances: dict[str, ResolvedInstance],
        parameters: Mapping[str, Scalar], ancestry: tuple[UUID, ...], depth: int,
    ) -> tuple[tuple, int]:
        """Resolve lexical operations with fixed loop work counted before model construction."""
        if depth > MAX_NESTING:
            raise ArchitectureError("nesting_limit", "flow nesting exceeds budget")
        operations, total = [], 0
        for operation in flow:
            if isinstance(operation, Call):
                if operation.call not in instances:
                    raise ArchitectureError("unknown_module", "call refers to an unknown module instance")
                resolved, work = self.call(operation, environment, instances[operation.call])
            elif isinstance(operation, Repeat):
                count = positive_count(resolve_value(operation.repeat, parameters, self.tokenizer))
                inner = dict(environment)
                carries = tuple(operation.carry.values())
                for carry in carries:
                    introduce(inner, carry.input, read_value(environment, carry.initial))
                body, work = self.flow(operation.body, inner, instances, parameters, ancestry, depth + 1)
                for carry in carries:
                    result = read_value(inner, carry.output, "uninitialized_carry")
                    compatible_port(result, read_value(inner, carry.input), {})
                    introduce(environment, carry.result, result)
                resolved, work = PlanRepeat(count, carries, body), work * count
            else:
                resolved, work = self.stack(operation, environment, instances, parameters, ancestry, depth)
            total += work
            if total > MAX_EXPANDED_CALLS:
                raise ArchitectureError("expanded_work_limit", "expanded model calls exceed budget")
            operations.append(resolved)
        return tuple(operations), total

    def stack(
        self, operation: Stack, environment: dict[str, TensorPort], instances: dict[str, ResolvedInstance],
        parameters: Mapping[str, Scalar], ancestry: tuple[UUID, ...], depth: int,
    ) -> tuple[PlanStack, int]:
        """Create independent stable child paths and validate their port-carried sequential flow."""
        count = positive_count(resolve_value(operation.stack, parameters, self.tokenizer))
        carries = tuple(operation.carry.values())
        if len({carry.input for carry in carries}) != len(carries) or set(operation.inputs) & {carry.input for carry in carries}:
            raise ArchitectureError("port_names", "stack input ports are supplied more than once")
        inner = dict(environment)
        values = {carry.input: carry.initial for carry in carries}
        paths, total = [], 0
        for index in range(count):
            path = f"{operation.name}/{index}"
            if path in instances:
                raise ArchitectureError("duplicate_instance", "stack instance path is already declared")
            instance = self.instance(path, operation.template, parameters, ancestry, depth)
            instances[path] = instance
            outputs = {carry.output: f"stack-{operation.name}-{index}-{number}" for number, carry in enumerate(carries)}
            call = Call(call=operation.name, inputs={**operation.inputs, **values}, outputs=outputs)
            _, work = self.call(call, inner, instance)
            values = {carry.input: outputs[carry.output] for carry in carries}
            total += work
            paths.append(path)
        for carry in carries:
            port = read_value(inner, values[carry.input])
            compatible_port(port, read_value(environment, carry.initial), {})
            introduce(environment, carry.result, port)
        return PlanStack(tuple(paths), dict(operation.inputs), carries), total


def resolve_recipe(
    recipe: Recipe, parameters: Mapping[str, Scalar], dependencies: Mapping[UUID, Dependency],
    tokenizer: FinalTokenizer | None = None,
) -> ResolvedPlan:
    """Resolve exact owned recipe/dependency snapshots into a bounded construction-ready plan."""
    snapshots = {key: (BlockDefinition(value.reference.model_copy(deep=True),
                                      value.declaration.model_copy(deep=True), value.module_type)
                       if isinstance(value, BlockDefinition) else ArchitectureDefinition(
                           value.reference.model_copy(deep=True), value.recipe.model_copy(deep=True)))
                 for key, value in dependencies.items()}
    return Resolver(snapshots, tokenizer.model_copy(deep=True) if tokenizer else None).recipe(recipe, parameters, (), 1)
