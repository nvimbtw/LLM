"use client"
import { useMemo, useState } from "react"
import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"
import {
  Select,
  SelectTrigger,
  SelectValue,
  SelectContent,
  SelectItem,
} from "@/components/ui/select"
import type {
  GraphFragment,
  GraphNode,
  GraphQuery,
} from "@/lib/architecture/contracts.generated"
import { GraphCanvas } from "./graph-canvas"
import { layoutGraph } from "@/lib/architecture-graph"
export interface ArchitectureGraphViewProps {
  fragment: GraphFragment | null
  query: GraphQuery
  loading: boolean
  error: string | null
  architectureView?: boolean
  stale?: boolean
  onQuery: (query: GraphQuery) => void
  onSource?: (revisionId: string, line: number | null) => void
}
export interface GraphNodeDetailsProps {
  node: GraphNode
  onSource?: ArchitectureGraphViewProps["onSource"]
}

/** Presents one bounded engine-produced graph scope with keyboard controls and an equivalent inspection list. */
export function ArchitectureGraphView(props: ArchitectureGraphViewProps) {
  const {
    fragment,
    query,
    loading,
    error,
    architectureView = true,
    stale,
    onQuery,
    onSource,
  } = props
  const layout = useMemo(
    () => (fragment ? layoutGraph(fragment) : null),
    [fragment]
  )
  const [selected, setSelected] = useState<string | null>(null)
  const current = fragment?.nodes.find((node) => node.id === selected)
  const nextQuery = (patch: Partial<GraphQuery>) =>
    onQuery({ ...query, ...patch, cursor: null })
  return (
    <section
      className="grid gap-4"
      aria-label="Model graph"
      aria-busy={loading}
    >
      <GraphControls
        query={query}
        onQuery={nextQuery}
        architectureView={architectureView}
      />
      <div className="flex flex-wrap items-center gap-2 text-sm text-muted-foreground">
        <span className="min-w-0 break-all">{query.scope ?? "root"}</span>
        <span>{query.signature}</span>
        {stale && (
          <strong className="text-foreground" role="status">
            Stale configuration — revalidate before use
          </strong>
        )}
        {fragment && (
          <span title={fragment.graph_hash}>
            Graph {fragment.graph_hash.slice(0, 12)} · Profile{" "}
            {query.identity_hash.slice(0, 12)}
          </span>
        )}
      </div>
      {error && <LoadError title={error} onRetry={() => onQuery(query)} />}
      {loading && <p role="status">Loading graph scope…</p>}
      {!loading && !error && fragment && layout && (
        <>
          <GraphCanvas
            key={`${fragment.graph_hash}:${query.scope}:${query.cursor ?? ""}`}
            fragment={fragment}
            layout={layout}
            selected={selected}
            onSelect={setSelected}
          />
          {layout.outsideEdges > 0 && (
            <p className="text-sm text-muted-foreground">
              {layout.outsideEdges} connections continue outside this page.
            </p>
          )}
          <ul aria-label="Graph nodes" className="divide-y divide-border">
            {fragment.nodes.map((node) => (
              <li
                key={node.id}
                className="flex flex-wrap items-center justify-between gap-2 py-2"
              >
                <Button
                  variant="ghost"
                  className="h-auto max-w-full min-w-0 justify-start text-left break-all whitespace-normal"
                  aria-pressed={selected === node.id}
                  onClick={() => setSelected(node.id)}
                >
                  {node.label} · {node.occurrence_path}
                </Button>
                {node.child_scope && (
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => nextQuery({ scope: node.child_scope! })}
                  >
                    Open {node.label}
                  </Button>
                )}
              </li>
            ))}
          </ul>
          <details>
            <summary>Connections ({fragment.edges.length})</summary>
            <ul
              aria-label="Graph connections"
              className="grid gap-1 font-mono text-sm break-all"
            >
              {fragment.edges.map((edge, index) => (
                <li key={index}>
                  {edge.source}.{edge.source_port} → {edge.target}.
                  {edge.target_port}
                </li>
              ))}
            </ul>
          </details>
          {current && <GraphNodeDetails node={current} onSource={onSource} />}
          {fragment.nodes.length === 0 && <p>No nodes in this scope.</p>}
          {fragment.next_cursor && (
            <Button
              variant="outline"
              onClick={() =>
                onQuery({ ...query, cursor: fragment.next_cursor })
              }
            >
              Next graph page
            </Button>
          )}
        </>
      )}
    </section>
  )
}

export interface GraphControlsProps {
  query: GraphQuery
  onQuery: (patch: Partial<GraphQuery>) => void
  architectureView: boolean
}
/** Renders graph view and mode selectors and bounded root-scope navigation. */
function GraphControls({
  query,
  onQuery,
  architectureView,
}: GraphControlsProps) {
  return (
    <div className="flex flex-wrap gap-3">
      <Select
        value={query.view}
        onValueChange={(value) => {
          if (
            (value === "architecture" && architectureView) ||
            value === "operations"
          )
            onQuery({ view: value, scope: "root" })
        }}
      >
        <SelectTrigger aria-label="Graph view">
          <SelectValue>
            {(value) =>
              value === "architecture" ? "Architecture" : "Captured operations"
            }
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          {architectureView && (
            <SelectItem value="architecture">Architecture</SelectItem>
          )}
          <SelectItem value="operations">Captured operations</SelectItem>
        </SelectContent>
      </Select>
      <Select
        value={query.mode}
        onValueChange={(value) => {
          if (value === "train" || value === "eval")
            onQuery({ mode: value, scope: "root" })
        }}
      >
        <SelectTrigger aria-label="Graph mode">
          <SelectValue>
            {(value) => (value === "train" ? "Training" : "Evaluation")}
          </SelectValue>
        </SelectTrigger>
        <SelectContent>
          <SelectItem value="train">Training</SelectItem>
          <SelectItem value="eval">Evaluation</SelectItem>
        </SelectContent>
      </Select>
      {query.scope !== "root" && (
        <Button variant="ghost" onClick={() => onQuery({ scope: "root" })}>
          Root scope
        </Button>
      )}
    </div>
  )
}

/** Shows captured metadata exactly as supplied; missing measurements remain explicitly unavailable. */
export function GraphNodeDetails({ node, onSource }: GraphNodeDetailsProps) {
  return (
    <section
      aria-label={`Selected ${node.label}`}
      className="grid gap-2 border-t border-border py-4 text-sm"
    >
      <h3 className="font-medium">{node.label}</h3>
      <dl className="grid gap-2 break-all">
        <div>
          <dt>Execution occurrence</dt>
          <dd className="font-mono">{node.occurrence_path}</dd>
        </div>
        <div>
          <dt>Module instance</dt>
          <dd>{node.instance_path ?? "Unavailable"}</dd>
        </div>
        <div>
          <dt>Parameter owners</dt>
          <dd>{node.parameter_owner_ids?.join(", ") || "None declared"}</dd>
        </div>
        <div>
          <dt>Unique parameter count</dt>
          <dd>
            {node.unique_parameter_count?.toLocaleString() ?? "Unavailable"}
          </dd>
        </div>
        {node.operation_target && (
          <div>
            <dt>Operation</dt>
            <dd>{node.operation_target}</dd>
          </div>
        )}
        {Object.entries(node.parameters ?? {}).map(([name, value]) => (
          <div key={name}>
            <dt>{name}</dt>
            <dd>{String(value)}</dd>
          </div>
        ))}
        {Object.entries(node.outputs ?? {}).map(([name, value]) => (
          <div key={name}>
            <dt>Output {name}</dt>
            <dd>
              [{value.shape.join(", ")}] · {value.dtype}
            </dd>
          </div>
        ))}
      </dl>
      {node.source && (
        <Button
          variant="outline"
          className="w-fit"
          disabled={!onSource}
          onClick={() =>
            onSource?.(node.source!.revision_id, node.source!.line ?? null)
          }
        >
          {node.source.file}
          {node.source.line == null ? "" : `:${node.source.line}`}
        </Button>
      )}
    </section>
  )
}
