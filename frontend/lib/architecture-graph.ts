import type {
  GraphFragment,
  GraphNode,
  GraphQuery,
} from "./architecture/contracts.generated"
import { decodeArchitecture } from "./architecture/decoder"
import {
  ArchitectureReadOwner,
  type ResourceIdentity,
} from "./architecture/read-owner"
export interface PositionedGraphNode {
  node: GraphNode
  x: number
  y: number
}
export interface GraphLayout {
  nodes: PositionedGraphNode[]
  width: number
  height: number
  outsideEdges: number
}
export interface GraphScopeState {
  query: GraphQuery
  fragment: GraphFragment | null
  loading: boolean
  error: string | null
}
export type GraphScopeLoader = (
  query: GraphQuery,
  signal: AbortSignal
) => Promise<GraphFragment>

/** Positions one bounded engine graph page without expanding loops or inferring execution/parameter semantics. */
export function layoutGraph(fragment: GraphFragment): GraphLayout {
  if (fragment.nodes.length > 256 || fragment.edges.length > 1024)
    throw new Error("Graph layout page budget exceeded.")
  const nodeIds = new Set(fragment.nodes.map((node) => node.id))
  const inputs = new Map(fragment.nodes.map((node) => [node.id, 0]))
  const children = new Map<string, string[]>()
  let outsideEdges = 0
  for (const edge of fragment.edges) {
    if (!nodeIds.has(edge.source) || !nodeIds.has(edge.target)) {
      outsideEdges++
      continue
    }
    inputs.set(edge.target, inputs.get(edge.target)! + 1)
    children.set(edge.source, [
      ...(children.get(edge.source) ?? []),
      edge.target,
    ])
  }
  const ready = fragment.nodes
    .filter((node) => inputs.get(node.id) === 0)
    .map((node) => node.id)
  const levels = new Map<string, number>()
  for (let index = 0; index < ready.length; index++) {
    const id = ready[index]
    for (const child of children.get(id) ?? []) {
      levels.set(
        child,
        Math.max(levels.get(child) ?? 0, (levels.get(id) ?? 0) + 1)
      )
      inputs.set(child, inputs.get(child)! - 1)
      if (inputs.get(child) === 0) ready.push(child)
    }
  }
  const rows = new Map<number, number>()
  const nodes = fragment.nodes.map((node) => {
    const column = Math.min(levels.get(node.id) ?? 0, 15)
    const row = rows.get(column) ?? 0
    rows.set(column, row + 1)
    return { node, x: 24 + column * 264, y: 24 + row * 112 }
  })
  return {
    nodes,
    width: Math.max(280, ...nodes.map((node) => node.x + 248)),
    height: Math.max(160, ...nodes.map((node) => node.y + 104)),
    outsideEdges,
  }
}

/** Owns one graph session/resource's bounded lazy reads and suppresses stale scope/page completions. */
export class GraphScopeOwner {
  private readonly reads: ArchitectureReadOwner
  private readonly listeners = new Set<() => void>()
  private state: GraphScopeState
  private disposed = false
  private generation = 0
  /** Captures the actual session/resource and initial presentation query; performs no network effects. */
  constructor(
    identity: ResourceIdentity,
    query: GraphQuery,
    private readonly loader: GraphScopeLoader
  ) {
    this.reads = new ArchitectureReadOwner(identity)
    this.state = {
      query: decodeArchitecture("GraphQuery", query),
      fragment: null,
      loading: false,
      error: null,
    }
  }
  /** Returns the stable current snapshot for a React external-store subscription. */
  snapshot = (): GraphScopeState => this.state
  /** Subscribes one view and returns the cleanup for its listener. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }
  /** Loads a single scope/page and discards any completion after a newer read or disposal. */
  async load(query: GraphQuery): Promise<void> {
    if (this.disposed) return
    const previous = this.state.fragment
    const generation = ++this.generation
    const expected = decodeArchitecture("GraphQuery", query)
    this.update({ query: expected, fragment: null, loading: true, error: null })
    try {
      const fragment = await this.reads.read(async (signal) => {
        const value = decodeArchitecture(
          "GraphFragment",
          await this.loader(expected, signal)
        )
        if (
          Object.keys(expected).some(
            (key) =>
              expected[key as keyof GraphQuery] !==
              value.query[key as keyof GraphQuery]
          )
        )
          throw new Error("Graph response belongs to another scope or profile.")
        if (
          previous &&
          ["identity_hash", "view", "mode", "signature"].every(
            (key) =>
              previous.query[key as keyof GraphQuery] ===
              expected[key as keyof GraphQuery]
          ) &&
          previous.graph_hash !== value.graph_hash
        )
          throw new Error("Graph artifact changed across scopes or pages.")
        return value
      })
      if (fragment && generation === this.generation)
        this.update({ query: expected, fragment, loading: false, error: null })
    } catch (error) {
      if (generation === this.generation)
        this.update({
          query: expected,
          fragment: null,
          loading: false,
          error:
            error instanceof Error ? error.message : "Could not load graph.",
        })
    }
  }
  /** Publishes a new graph snapshot to this owner's current subscribers. */
  private update(state: GraphScopeState): void {
    if (!this.disposed) {
      this.state = state
      this.listeners.forEach((listener) => listener())
    }
  }
  /** Aborts active observation and removes subscriptions without cancelling durable validation work. */
  dispose(): void {
    this.disposed = true
    this.reads.dispose()
    this.listeners.clear()
  }
}
