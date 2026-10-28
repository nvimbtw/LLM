import type {
  GraphFragment,
  GraphQuery,
} from "@/lib/architecture/contracts.generated"
import { decodeArchitecture } from "@/lib/architecture/decoder"
export const graphFixtureQuery: GraphQuery = {
  identity_hash: "a".repeat(64),
  view: "architecture",
  mode: "eval",
  signature: "full_prefix",
  scope: "root",
}
/** Builds a contract-valid synthetic scope with shared occurrences and independent stack owners. */
export function graphFixture(patch: Partial<GraphQuery> = {}): GraphFragment {
  return decodeArchitecture("GraphFragment", {
    schema_version: 1,
    graph_hash: "b".repeat(64),
    query: { ...graphFixtureQuery, ...patch },
    nodes: [
      {
        id: "loop",
        kind: "loop",
        label: "Repeated core",
        occurrence_path: "root/repeat",
        iterations: 4,
        child_scope: "root/repeat",
        parameter_owner_ids: ["core.weight"],
        unique_parameter_count: 64,
      },
      {
        id: "shared-a",
        kind: "call",
        label: "Shared pass one",
        instance_path: "core",
        occurrence_path: "root/repeat/0",
        parameter_owner_ids: ["core.weight"],
        outputs: { hidden: { shape: [1, 8, 16], dtype: "float32" } },
      },
      {
        id: "shared-b",
        kind: "call",
        label: "Shared pass two",
        instance_path: "core",
        occurrence_path: "root/repeat/1",
        parameter_owner_ids: ["core.weight"],
      },
      {
        id: "stacked",
        kind: "module",
        label: "Independent layer",
        instance_path: "stack/1",
        occurrence_path: "root/stack/1",
        parameter_owner_ids: ["stack.1.weight"],
        source: {
          revision_id: "30000000-0000-4000-8000-000000000001",
          file: "blocks/core.py",
          line: 12,
        },
      },
    ],
    edges: [
      {
        source: "shared-a",
        source_port: "hidden",
        target: "shared-b",
        target_port: "hidden",
      },
    ],
    next_cursor: "page-2",
  })
}
