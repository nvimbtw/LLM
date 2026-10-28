"use client"
import { useEffect, useState } from "react"
import { ArchitectureGraphView } from "./graph-view"
import { graphFixture, graphFixtureQuery } from "./fixtures"
import { GraphScopeOwner, type GraphScopeState } from "@/lib/architecture-graph"
import { decodeArchitecture } from "@/lib/architecture/decoder"
import type { GraphQuery } from "@/lib/architecture/contracts.generated"

/** Exercises bounded synthetic graph scopes and real presentation interactions without claiming capture evidence. */
export function ArchitectureGraphFixture() {
  const [query, setQuery] = useState<GraphQuery>(() =>
    decodeArchitecture("GraphQuery", graphFixtureQuery)
  )
  const [state, setState] = useState<GraphScopeState>({
    query,
    fragment: null,
    loading: true,
    error: null,
  })
  const [source, setSource] = useState("")
  useEffect(() => {
    const owner = new GraphScopeOwner(
      { token: "local-fixture", resource: "synthetic-graph" },
      query,
      async (requested) => {
        const fragment = graphFixture(requested)
        if (requested.scope === "root/repeat")
          fragment.nodes = fragment.nodes.map((node) =>
            node.id === "loop"
              ? {
                  ...node,
                  label: "Nested repeat",
                  child_scope: "root/repeat/nested",
                }
              : node
          )
        if (requested.cursor) {
          fragment.nodes = fragment.nodes.slice(3)
          fragment.next_cursor = null
        }
        return fragment
      }
    )
    const unsubscribe = owner.subscribe(() => setState(owner.snapshot()))
    void owner.load(query)
    return () => {
      unsubscribe()
      owner.dispose()
    }
  }, [query])
  const current =
    state.query === query ||
    JSON.stringify(state.query) === JSON.stringify(query)
  return (
    <div className="grid gap-6 p-6">
      <h1>Synthetic graph fixture</h1>
      <p className="text-sm text-muted-foreground">
        Handcrafted contract fixture; actual capture acceptance awaits ISS-54.
      </p>
      <ArchitectureGraphView
        fragment={current ? state.fragment : null}
        query={query}
        loading={!current || state.loading}
        error={current ? state.error : null}
        stale
        onQuery={setQuery}
        onSource={(revision, line) => setSource(`${revision}:${line}`)}
      />
      <output aria-label="Selected source">{source}</output>
    </div>
  )
}
