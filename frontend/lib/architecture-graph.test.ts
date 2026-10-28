import { test } from "node:test"
import assert from "node:assert/strict"
import { createElement } from "react"
import { renderToStaticMarkup } from "react-dom/server"
import {
  ArchitectureGraphView,
  GraphNodeDetails,
} from "../components/architecture-graph/graph-view"
import { GraphScopeOwner, layoutGraph } from "./architecture-graph"
import type { GraphFragment } from "./architecture/contracts.generated"
import {
  graphFixture,
  graphFixtureQuery as query,
} from "../components/architecture-graph/fixtures"
/** Provides a controllable graph response for scope-replacement races. */
function pending() {
  let resolve!: (value: GraphFragment) => void
  const promise = new Promise<GraphFragment>((complete) => {
    resolve = complete
  })
  return { promise, resolve }
}

test("bounded layout preserves shared occurrence identities and independent parameter ownership", () => {
  const fragment = graphFixture()
  const layout = layoutGraph(fragment)
  assert.equal(layout.nodes.length, 4)
  assert.notEqual(layout.nodes[1].x, layout.nodes[2].x)
  assert.equal(
    layout.nodes[1].node.parameter_owner_ids?.[0],
    layout.nodes[2].node.parameter_owner_ids?.[0]
  )
  assert.notEqual(
    layout.nodes[2].node.parameter_owner_ids?.[0],
    layout.nodes[3].node.parameter_owner_ids?.[0]
  )
  assert.equal(layout.nodes[0].node.iterations, 4)
  assert.equal(fragment.nodes.length, 4)
  assert.throws(
    () =>
      layoutGraph({
        ...fragment,
        nodes: Array.from({ length: 257 }, () => fragment.nodes[0]),
      }),
    /budget/
  )
})
test("presentation exposes text equivalent, exact source and absent metrics without fabricated zeros", () => {
  const fragment = graphFixture()
  const markup = renderToStaticMarkup(
    createElement(ArchitectureGraphView, {
      fragment,
      query,
      loading: false,
      error: null,
      stale: true,
      onQuery: () => {},
    })
  )
  assert.match(markup, /Graph nodes/)
  assert.match(markup, /Open Repeated core/)
  assert.match(markup, /Next graph page/)
  assert.match(markup, /Stale configuration/)
  const missing = renderToStaticMarkup(
    createElement(GraphNodeDetails, { node: fragment.nodes[1] })
  )
  assert.match(missing, /Unavailable/)
  assert.match(missing, /float32/)
  assert.match(missing, /core.weight/)
  const source = renderToStaticMarkup(
    createElement(GraphNodeDetails, {
      node: fragment.nodes[3],
      onSource: () => {},
    })
  )
  assert.match(source, /blocks\/core.py:12/)
})
test("new scopes abort old reads and cannot be replaced by late responses", async () => {
  const old = pending()
  let oldSignal: AbortSignal | undefined
  const owner = new GraphScopeOwner(
    { token: "session", resource: "run" },
    query,
    async (requested, signal) => {
      if (requested.scope === "old") {
        oldSignal = signal
        return old.promise
      }
      return graphFixture(requested)
    }
  )
  const first = owner.load({ ...query, scope: "old" })
  await owner.load({ ...query, scope: "root/repeat", cursor: "page-2" })
  assert.equal(oldSignal?.aborted, true)
  old.resolve(graphFixture({ scope: "old" }))
  await first
  assert.equal(owner.snapshot().query.scope, "root/repeat")
  assert.equal(owner.snapshot().query.cursor, "page-2")
  owner.dispose()
})
test("disposed graph owners suppress private late values; mismatched profiles fail visibly", async () => {
  const response = pending()
  const owner = new GraphScopeOwner(
    { token: "old", resource: "run" },
    query,
    () => response.promise
  )
  const loading = owner.load(query)
  owner.dispose()
  response.resolve(graphFixture())
  await loading
  assert.equal(owner.snapshot().fragment, null)
  const current = new GraphScopeOwner(
    { token: "new", resource: "run" },
    query,
    async () => graphFixture({ mode: "train" })
  )
  await current.load(query)
  assert.match(current.snapshot().error!, /another scope or profile/)
  current.dispose()
})

test("pagination cannot silently mix immutable graph artifacts", async () => {
  const owner = new GraphScopeOwner(
    { token: "session", resource: "run" },
    query,
    async (requested) => ({
      ...graphFixture(requested),
      graph_hash: requested.cursor ? "c".repeat(64) : "b".repeat(64),
    })
  )
  await owner.load(query)
  await owner.load({ ...query, cursor: "page-2" })
  assert.match(owner.snapshot().error!, /artifact changed/)
  assert.equal(owner.snapshot().fragment, null)
  owner.dispose()
})
test("external-page edges stay explicit and layout edge work remains bounded", () => {
  const fragment = graphFixture()
  const external = {
    source: "outside",
    source_port: "hidden",
    target: "shared-a",
    target_port: "hidden",
  }
  assert.equal(
    layoutGraph({ ...fragment, edges: [...fragment.edges, external] })
      .outsideEdges,
    1
  )
  assert.throws(
    () =>
      layoutGraph({
        ...fragment,
        edges: Array.from({ length: 1025 }, () => external),
      }),
    /budget/
  )
})
