"""Bounded hierarchical inspection artifacts and immutable scope/cursor reads."""

import base64
import hashlib
import json

from llm.architecture.errors import ArchitectureError
from llm.architecture.plan import BlockDefinition, PlanCall, PlanRepeat, PlanStack, ResolvedPlan
from llm.architecture.spec import MAX_GRAPH_BYTES, MAX_GRAPH_EDGES, MAX_GRAPH_NODES, ValidationReceipt
from llm.architecture.validation_records import canonical_bytes
from llm.architecture.wire import GraphCursor, GraphEdge, GraphFragment, GraphNode, GraphQuery


class GraphArtifacts:
    """Own one bounded set of canonical graph blobs with a manifest awaiting parent publication."""

    def __init__(self, identity_hash: str):
        """Start an empty profile-specific inventory, separate from model semantic identity."""
        self.identity_hash = identity_hash
        self.blobs: dict[str, str] = {}
        self.entries: list[dict] = []
        self.total_bytes = 0

    def add(self, *, view: str, mode: str, signature: str, scope: str,
            nodes: tuple | list, edges: tuple | list, **metadata) -> None:
        """Validate and retain one complete bounded scope, never opaque Python/FX objects."""
        if len(nodes) > MAX_GRAPH_NODES or len(edges) > MAX_GRAPH_EDGES:
            raise ArchitectureError("graph_size_limit", "inspection graph exceeds node/edge budget")
        payload = {"nodes": [GraphNode.model_validate(node).model_dump(mode="json", exclude_none=True) for node in nodes],
                   "edges": [GraphEdge.model_validate(edge).model_dump(mode="json") for edge in edges]}
        raw = canonical_bytes(payload)
        digest = hashlib.sha256(raw).hexdigest()
        if digest not in self.blobs:
            self.total_bytes += len(raw)
        if self.total_bytes > MAX_GRAPH_BYTES:
            raise ArchitectureError("graph_byte_limit", "inspection graph artifact bytes exceed budget")
        self.blobs[digest] = raw.decode("utf-8")
        self.entries.append({"view": view, "mode": mode, "signature": signature, "scope": scope,
                             "sha256": digest, **metadata})

    def finish(self) -> tuple[str, dict[str, str]]:
        """Seal the exact-byte SHA manifest after every enabled signature/scope is complete."""
        raw = canonical_bytes({"schema_version": 1, "identity_hash": self.identity_hash,
                               "complete": True, "graphs": self.entries})
        if self.total_bytes + len(raw) > MAX_GRAPH_BYTES:
            raise ArchitectureError("graph_byte_limit", "graph manifest exceeds aggregate byte budget")
        digest = hashlib.sha256(raw).hexdigest()
        return digest, {**self.blobs, digest: raw.decode("utf-8")}


def export_guards(entries: list[dict]) -> tuple[str, ...]:
    """Identify actual strict capture partitions, including their static shapes, within receipt budgets."""
    guards = tuple(canonical_bytes({key: entry[key] for key in
                   ("mode", "signature", "scope", "strict", "guards", "domain") if key in entry}).decode("utf-8")
                   for entry in entries if entry.get("strict"))
    if len(guards) > 128 or any(len(value) > 1024 for value in guards):
        raise ArchitectureError("capture_guard_limit", "labeled export guards exceed receipt budget")
    return guards


def verify_receipt_graphs(receipt: ValidationReceipt, blobs: dict[str, str]) -> None:
    """Validate a bounded publication inventory and classify malformed graph containers as domain refusals."""
    try:
        verify_graph_inventory(receipt,blobs)
    except (KeyError,TypeError,AttributeError) as error:
        raise ArchitectureError("graph_identity","graph manifest or scope structure is invalid") from error


def verify_graph_inventory(receipt: ValidationReceipt, blobs: dict[str, str]) -> None:
    """Verify a complete immutable graph inventory against its receipt before package/publication adoption.

    This data-only check verifies profile linkage, referenced bytes, graph structure
    and strict mode coverage. It does not execute or recertify model arithmetic.
    """
    total = 0
    for digest, source in blobs.items():
        raw = source.encode("utf-8")
        total += len(raw)
        if hashlib.sha256(raw).hexdigest() != digest or total > MAX_GRAPH_BYTES:
            raise ArchitectureError("graph_identity", "graph inventory bytes are corrupt or exceed budget")
    if receipt.graph_manifest_hash not in blobs:
        raise ArchitectureError("graph_identity", "receipt graph manifest is missing")
    manifest = json.loads(blobs[receipt.graph_manifest_hash])
    if (manifest.get("schema_version") != 1 or manifest.get("complete") is not True
            or manifest.get("identity_hash") != receipt.identity_hash):
        raise ArchitectureError("graph_identity", "graph manifest differs from its complete receipt identity")
    entries = manifest.get("graphs")
    if not isinstance(entries, list) or not entries or len(entries) > MAX_GRAPH_NODES:
        raise ArchitectureError("graph_size_limit", "graph manifest scopes are missing or exceed budget")
    keys, children = set(), []
    for entry in entries:
        query = GraphQuery(identity_hash=receipt.identity_hash,
            **{key: entry[key] for key in ("view", "mode", "signature", "scope")})
        key = (query.view, query.mode, query.signature, query.scope)
        if key in keys or entry["sha256"] not in blobs:
            raise ArchitectureError("graph_identity", "duplicate graph scope or missing graph bytes")
        keys.add(key)
        graph = json.loads(blobs[entry["sha256"]])
        if len(graph["nodes"]) > MAX_GRAPH_NODES or len(graph["edges"]) > MAX_GRAPH_EDGES:
            raise ArchitectureError("graph_size_limit", "graph scope exceeds node/edge budget")
        nodes = [GraphNode.model_validate(value) for value in graph["nodes"]]
        ids = {node.id for node in nodes}
        if len(ids) != len(nodes):
            raise ArchitectureError("graph_identity", "graph scope contains duplicate node identities")
        for value in graph["edges"]:
            edge = GraphEdge.model_validate(value)
            if edge.source not in ids or edge.target not in ids:
                raise ArchitectureError("graph_identity", "graph edge refers to a missing node")
        children.extend((query.view, query.mode, query.signature, node.child_scope) for node in nodes if node.child_scope)
    if any(child not in keys for child in children):
        raise ArchitectureError("graph_scope", "graph child scope is missing")
    strict_modes = {entry["mode"] for entry in entries if entry.get("strict") is True and entry["view"] == "operations"}
    if strict_modes != set(receipt.identity.profile.modes) or export_guards(entries) != receipt.identity.profile.export_guards:
        raise ArchitectureError("graph_identity", "graph strict exports differ from receipt mode/guard evidence")


def architecture_scopes(plan: ResolvedPlan, model, prefix: str = "root") -> dict[str, tuple[list, list]]:
    """Retain instances, fixed repeats and explicit port routing without expanding shared loop calls."""
    scopes = {}
    nodes, edges = [], []
    owners = {}
    for path, parameter in model.parameter_paths().items():
        owners.setdefault(id(parameter), path)
    for instance in plan.instances:
        child = model.instances[instance.path]
        child_scope = f"{prefix}/{instance.path}" if isinstance(instance.definition, ResolvedPlan) else None
        names = tuple(dict.fromkeys(owners[id(parameter)] for parameter in child.parameters()))
        nodes.append({"id": f"module:{instance.path}", "kind": "module", "label": instance.path[:120],
                      "instance_path": instance.path, "occurrence_path": instance.path,
                      "parameter_owner_ids": names[:128], "parameters": dict(instance.parameters),
                      "unique_parameter_count": sum(parameter.numel() for parameter in child.parameters()),
                      **({"child_scope": child_scope} if child_scope else {})})
        if child_scope:
            scopes.update(architecture_scopes(instance.definition, child, child_scope))
    environment = {}
    for name in plan.inputs:
        node_id = f"input:{name}"
        nodes.append({"id": node_id, "kind": "input", "label": name, "occurrence_path": node_id})
        environment[name] = (node_id, name)
    flow_scopes(plan.operations, prefix, environment, nodes, edges, scopes)
    for port, value in plan.result.items():
        node_id = f"output:{port}"
        nodes.append({"id": node_id, "kind": "output", "label": port, "occurrence_path": node_id})
        source, source_port = environment[value]
        edges.append({"source": source, "source_port": source_port, "target": node_id, "target_port": port})
    scopes[prefix] = (nodes, edges)
    return scopes


def flow_scopes(operations: tuple, scope: str, environment: dict, nodes: list, edges: list, scopes: dict) -> None:
    """Build lexical call/loop wiring with separate body scopes and explicit carry edges."""
    for index, operation in enumerate(operations):
        node_id = f"flow:{index}"
        occurrence = f"{scope}/{index}"
        if isinstance(operation, PlanCall):
            nodes.append({"id": node_id, "kind": "call", "label": operation.instance[:120],
                          "instance_path": operation.instance, "occurrence_path": occurrence})
            for port, value in operation.inputs.items():
                source, source_port = environment[value]
                edges.append({"source": source, "source_port": source_port, "target": node_id, "target_port": port})
            environment.update({value: (node_id, port) for port, value in operation.outputs.items()})
        else:
            count = operation.count if isinstance(operation, PlanRepeat) else len(operation.instances)
            child_scope = f"{occurrence}/body"
            nodes.append({"id": node_id, "kind": "loop", "label": "Shared repeat" if isinstance(operation, PlanRepeat) else "Independent stack",
                          "occurrence_path": occurrence, "iterations": count, "child_scope": child_scope,
                          "parameters": {f"carry_{number}_{field}": getattr(carry, field)
                                         for number, carry in enumerate(operation.carry)
                                         for field in ("initial", "input", "output", "result")}})
            inner = {}
            children, child_edges = [], []
            captures = (required_values(operation.body, {carry.input for carry in operation.carry})
                        if isinstance(operation, PlanRepeat) else set(operation.inputs.values()))
            for name in sorted(captures):
                source, port = environment[name]
                input_id = f"input:{name}"
                children.append({"id": input_id, "kind": "input", "label": name, "occurrence_path": f"{child_scope}/{name}"})
                inner[name] = (input_id, name)
                if isinstance(operation, PlanRepeat):
                    edges.append({"source": source, "source_port": port, "target": node_id, "target_port": name})
            for name, carry in enumerate(operation.carry):
                source, port = environment[carry.initial]
                edges.append({"source": source, "source_port": port, "target": node_id, "target_port": carry.input})
                input_id = f"carry:{carry.input}"
                children.append({"id": input_id, "kind": "input", "label": carry.input,
                                 "occurrence_path": f"{child_scope}/{carry.input}"})
                inner[carry.input] = (input_id, carry.input)
                environment[carry.result] = (node_id, carry.output)
            if isinstance(operation, PlanRepeat):
                flow_scopes(operation.body, child_scope, inner, children, child_edges, scopes)
            else:
                static = {port: inner[value] for port, value in operation.inputs.items()}
                for port, value in operation.inputs.items():
                    source, source_port = environment[value]
                    edges.append({"source": source, "source_port": source_port, "target": node_id, "target_port": port})
                for number, path in enumerate(operation.instances):
                    call_id = f"stack:{number}"
                    children.append({"id": call_id, "kind": "call", "label": path[:120],
                                     "instance_path": path, "occurrence_path": f"{occurrence}/{number}"})
                    inputs = {**static, **{carry.input: inner[carry.input] for carry in operation.carry}}
                    for port, (source, source_port) in inputs.items():
                        child_edges.append({"source": source, "source_port": source_port, "target": call_id, "target_port": port})
                    for carry in operation.carry:
                        inner[carry.input] = (call_id, carry.output)
            for carry in operation.carry:
                source, source_port = inner[carry.output if isinstance(operation, PlanRepeat) else carry.input]
                output_id = f"output:{carry.output}"
                children.append({"id": output_id, "kind": "output", "label": carry.output,
                                 "occurrence_path": f"{child_scope}/{carry.output}"})
                child_edges.append({"source": source, "source_port": source_port, "target": output_id, "target_port": carry.output})
                if isinstance(operation, PlanRepeat):
                    child_edges.append({"source": output_id, "source_port": carry.output,
                                        "target": f"carry:{carry.input}", "target_port": carry.input})
            scopes[child_scope] = (children, child_edges)


def required_values(operations: tuple, initialized: set[str]) -> set[str]:
    """Find free lexical tensor reads so loop scopes expose actual captured input dependencies."""
    available, required = set(initialized), set()
    for operation in operations:
        if isinstance(operation, PlanCall):
            reads = set(operation.inputs.values())
            outputs = set(operation.outputs.values())
        else:
            reads = {carry.initial for carry in operation.carry}
            if isinstance(operation, PlanRepeat):
                reads |= required_values(operation.body, {carry.input for carry in operation.carry})
            else:
                reads |= set(operation.inputs.values())
            outputs = {carry.result for carry in operation.carry}
        required |= reads - available
        available |= outputs
    return required


def read_fragment(manifest_hash: str, blobs: dict[str, str], query: GraphQuery) -> GraphFragment:
    """Page published immutable graph scopes with exact manifest/profile-bound cursor identity."""
    if manifest_hash not in blobs or hashlib.sha256(blobs[manifest_hash].encode()).hexdigest() != manifest_hash:
        raise ArchitectureError("graph_identity", "graph manifest bytes are missing or corrupt")
    manifest = json.loads(blobs[manifest_hash])
    if manifest.get("identity_hash") != query.identity_hash or not manifest.get("complete"):
        raise ArchitectureError("graph_identity", "graph query belongs to another validation identity")
    selected = next((entry for entry in manifest["graphs"] if all(entry[key] == getattr(query, key)
                    for key in ("view", "mode", "signature", "scope"))), None)
    if selected is None:
        raise ArchitectureError("graph_scope", "graph scope/signature is unavailable")
    raw = blobs.get(selected["sha256"], "").encode()
    if hashlib.sha256(raw).hexdigest() != selected["sha256"]:
        raise ArchitectureError("graph_identity", "graph scope bytes are missing or corrupt")
    offset = 0
    if query.cursor:
        try:
            cursor = GraphCursor.model_validate_json(base64.urlsafe_b64decode(query.cursor))
            cursor.require_query(query)
        except (ValueError, TypeError) as error:
            raise ArchitectureError("graph_cursor", "invalid or replayed graph cursor") from error
        if cursor.graph_hash != manifest_hash:
            raise ArchitectureError("graph_cursor", "cursor belongs to another graph manifest")
        offset = cursor.offset
    graph = json.loads(raw)
    nodes = graph["nodes"][offset:offset + query.limit]
    ids = {node["id"] for node in nodes}
    edges = [edge for edge in graph["edges"] if edge["target"] in ids]
    if len(edges) > 1024:
        raise ArchitectureError("graph_page_edges", "choose a smaller graph node page for this dense scope")
    cursor = None
    if offset + len(nodes) < len(graph["nodes"]):
        value = GraphCursor(graph_hash=manifest_hash, identity_hash=query.identity_hash, view=query.view,
                            mode=query.mode, signature=query.signature, scope=query.scope, offset=offset + len(nodes))
        cursor = base64.urlsafe_b64encode(value.model_dump_json().encode()).decode()
    return GraphFragment(schema_version=1, graph_hash=manifest_hash, query=query,
                         nodes=tuple(nodes), edges=tuple(edges), next_cursor=cursor)
