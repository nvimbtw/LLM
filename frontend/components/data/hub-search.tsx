"use client"

import * as React from "react"
import { Download, Search } from "lucide-react"

import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { FormError, LoadError } from "@/components/ui/error-state"
import { Input } from "@/components/ui/input"
import { Panel } from "@/components/ui/panel"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import {
  hubDetail,
  hubSample,
  pullDataset,
  searchHub,
  type DatasetKind,
  type HubDetail,
  type HubResult,
  type HubSample,
  type HubSplit,
  type Turn,
} from "@/lib/data"

/** Long enough that typing a repo name is one request, not eight. */
const DEBOUNCE_MS = 350

/** What a pull defaults to. Deliberately small: this writes rows into Postgres. */
const DEFAULT_ROWS = 5000

export interface HubSearchProps {
  onQueued: () => void
}

export function HubSearch({ onQueued }: HubSearchProps) {
  const [query, setQuery] = React.useState("")
  const [results, setResults] = React.useState<HubResult[]>([])
  const [isLoading, setIsLoading] = React.useState(true)
  const [error, setError] = React.useState<string | null>(null)
  const [chosen, setChosen] = React.useState<HubResult | null>(null)

  React.useEffect(() => {
    let active = true
    const timer = setTimeout(() => {
      setIsLoading(true)
      searchHub(query)
        .then((found) => {
          if (!active) return
          setResults(found)
          setError(null)
        })
        .catch(
          (e) =>
            active && setError(e instanceof Error ? e.message : "search failed")
        )
        .finally(() => active && setIsLoading(false))
    }, DEBOUNCE_MS)

    return () => {
      active = false
      clearTimeout(timer)
    }
  }, [query])

  return (
    <Panel
      variant="open"
      title="Hugging Face"
      description="most downloaded first"
      bodyClassName="p-0"
    >
      <div className="border-b border-border py-4">
        <div className="relative">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search datasets — tinystories, fineweb, smoltalk…"
            className="pl-8"
            aria-label="Search Hugging Face datasets"
          />
        </div>
      </div>

      {error ? (
        <div className="py-4">
          <LoadError title="The Hub didn't answer." hint={error} />
        </div>
      ) : isLoading ? (
        <div className="space-y-3 py-4">
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
          <Skeleton className="h-10 w-full" />
        </div>
      ) : results.length === 0 ? (
        <p className="py-6 text-sm text-muted-foreground">
          Nothing matched “{query}”.
        </p>
      ) : (
        <ul className="divide-y divide-border">
          {results.map((result) => (
            <li key={result.repo} className="flex items-center gap-4 py-3">
              {/* The row itself opens the dialog, which leads with a sample of
                  the rows: choosing a dataset means looking at it first, and
                  making that a separate affordance would hide it. */}
              <button
                type="button"
                className="min-w-0 flex-1 cursor-pointer text-left"
                onClick={() => setChosen(result)}
              >
                <p className="truncate text-sm font-medium">{result.repo}</p>
                <p className="mt-0.5 truncate text-xs text-muted-foreground">
                  {result.downloads !== null
                    ? `${result.downloads.toLocaleString()} downloads`
                    : "downloads unknown"}
                  {result.tags.length > 0
                    ? ` · ${result.tags.slice(0, 3).join(", ")}`
                    : ""}
                </p>
              </button>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setChosen(result)}
              >
                <Download data-icon="inline-start" className="size-3.5" />
                Pull
              </Button>
            </li>
          ))}
        </ul>
      )}

      {/* Keyed by repo so choosing another one remounts it with fresh fields,
          rather than an effect resetting six pieces of state on every change. */}
      <PullDialog
        key={chosen?.repo ?? "none"}
        repo={chosen?.repo ?? null}
        onClose={() => setChosen(null)}
        onQueued={() => {
          setChosen(null)
          onQueued()
        }}
      />
    </Panel>
  )
}

interface PullDialogProps {
  repo: string | null
  onClose: () => void
  onQueued: () => void
}

/**
 * What to take from a repo, and what to call it here.
 *
 * The row cap is required and pre-filled rather than optional: this writes into
 * a database, and "all of FineWeb" typed into a form at speed is not a request
 * anyone means.
 */
function PullDialog({ repo, onClose, onQueued }: PullDialogProps) {
  const [detail, setDetail] = React.useState<HubDetail | null>(null)
  const [name, setName] = React.useState(
    repo ? (repo.split("/").pop() ?? repo) : ""
  )
  const [kind, setKind] = React.useState<DatasetKind>("pretrain")
  const [prefix, setPrefix] = React.useState("")
  const [column, setColumn] = React.useState("")
  const [rows, setRows] = React.useState(DEFAULT_ROWS)
  const [submitting, setSubmitting] = React.useState(false)
  const [failure, setFailure] = React.useState<string | null>(null)

  const [where, setWhere] = React.useState<HubSplit | null>(null)
  /** Both answers carry the request they belong to, so switching splits shows
      the skeleton again without an effect clearing state on the way out. */
  const [loaded, setLoaded] = React.useState<{
    key: string
    sample: HubSample
  } | null>(null)
  const [refused, setRefused] = React.useState<{
    key: string
    message: string
  } | null>(null)
  /** Only until the user says otherwise — after that the sample stops steering
      fields they have already answered. */
  const [defaulted, setDefaulted] = React.useState(false)

  React.useEffect(() => {
    if (!repo) return
    let active = true
    hubDetail(repo)
      .then((found) => active && setDetail(found))
      .catch(
        (e) =>
          active &&
          setFailure(
            e instanceof Error ? e.message : "could not read the dataset"
          )
      )
    return () => {
      active = false
    }
  }, [repo])

  const asked = `${repo}::${where?.config ?? ""}/${where?.split ?? ""}`
  const sample = loaded?.key === asked ? loaded.sample : null
  const sampleError = refused?.key === asked ? refused.message : null

  React.useEffect(() => {
    if (!repo) return
    let active = true
    hubSample(repo, where)
      .then((found) => {
        if (!active) return
        setLoaded({ key: asked, sample: found })
        // The rows are the best evidence for the two fields nothing else can
        // answer — but they only pre-fill, never overwrite a choice.
        if (defaulted) return
        if (found.suggested_kind) setKind(found.suggested_kind)
        if (found.suggested_column) setColumn(found.suggested_column)
      })
      .catch((e) => {
        if (!active) return
        const message = e instanceof Error ? e.message : "no preview available"
        setRefused({ key: asked, message })
      })
    return () => {
      active = false
    }
    // `defaulted` is read, not tracked: flipping it must not refetch the rows,
    // and `asked` is derived from the two that do.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [repo, where])

  // The sample's columns first: they are read off real rows, while the card's
  // are declared and may describe a config nobody is pulling.
  const columns = React.useMemo(
    () =>
      Array.from(
        new Set([
          ...(sample?.columns ?? []),
          ...(detail?.configs ?? []).flatMap((c) => c.columns),
        ])
      ).filter(Boolean),
    [detail, sample]
  )

  async function submit() {
    if (!repo) return
    setSubmitting(true)
    setFailure(null)
    try {
      await pullDataset({
        repo,
        name: name.trim(),
        kind,
        prefix: prefix || null,
        column: column || null,
        limit_rows: rows,
      })
      toast.success(`Pulling ${name} from ${repo}.`)
      onQueued()
    } catch (e) {
      setFailure(e instanceof Error ? e.message : "the pull was refused")
    } finally {
      setSubmitting(false)
    }
  }

  return (
    <Dialog open={repo !== null} onOpenChange={(open) => !open && onClose()}>
      {/* Wider and taller than the default: the rows are read on the left while
          the form is answered on the right, and stacking them would put the
          evidence off-screen exactly when it is being used. */}
      <DialogContent className="max-h-[88vh] max-w-4xl">
        <DialogHeader>
          <DialogTitle>Pull {repo}</DialogTitle>
          <DialogDescription>
            A slice of it, stored here as documents. Cleaning comes after.
          </DialogDescription>
        </DialogHeader>

        {/* One column until there is room for two — a narrow window reads the
            sample first and the form under it, which is the same order. */}
        <div className="grid min-h-0 flex-1 gap-5 md:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
          <SamplePanel
            sample={sample}
            error={sampleError}
            column={column}
            onPick={(picked) => {
              setWhere(picked)
              setDefaulted(true)
            }}
          />

          <div className="min-h-0 space-y-4 overflow-y-auto md:pr-1">
            <Field label="Call it" htmlFor="pull-name">
              <Input
                id="pull-name"
                value={name}
                onChange={(e) => setName(e.target.value)}
                placeholder="tinystories"
              />
            </Field>

            <Field label="Kind" htmlFor="pull-kind">
              <Select
                value={kind}
                onValueChange={(v) => {
                  setKind(v as DatasetKind)
                  setDefaulted(true)
                }}
              >
                <SelectTrigger id="pull-kind">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pretrain">
                    Prose — for pretraining
                  </SelectItem>
                  <SelectItem value="sft">
                    Conversations — for fine-tuning
                  </SelectItem>
                </SelectContent>
              </Select>
            </Field>

            <Field
              label="Documents"
              htmlFor="pull-rows"
              hint="Every pull is capped. These are rows in a database, not files on disk."
            >
              <Input
                id="pull-rows"
                type="number"
                min={1}
                value={rows}
                onChange={(e) =>
                  setRows(Math.max(1, Number(e.target.value) || 1))
                }
              />
            </Field>

            {(detail?.prefixes.length ?? 0) > 1 && (
              <Field
                label="Which files"
                htmlFor="pull-prefix"
                hint="A repo that ships train and test separately would mix them otherwise."
              >
                <Select
                  value={prefix || "all"}
                  onValueChange={(v) =>
                    setPrefix(v === "all" || v === null ? "" : v)
                  }
                >
                  <SelectTrigger id="pull-prefix">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="all">Everything</SelectItem>
                    {(detail?.prefixes ?? []).map((p) => (
                      <SelectItem
                        key={p.prefix || "root"}
                        value={p.prefix || "all"}
                      >
                        {p.prefix || "(root)"} — {p.shards} shards
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            )}

            {columns.length > 0 && (
              <Field
                label="Column"
                htmlFor="pull-column"
                hint="Left alone, the first plausible one is used."
              >
                <Select
                  value={column || "auto"}
                  onValueChange={(v) => {
                    setColumn(v === "auto" || v === null ? "" : v)
                    setDefaulted(true)
                  }}
                >
                  <SelectTrigger id="pull-column">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="auto">Detect it</SelectItem>
                    {columns.map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </Field>
            )}

            {detail?.gated && (
              <p className="text-xs text-muted-foreground">
                This dataset is gated. The pull will fail unless the machine is
                logged in and the terms have been accepted.
              </p>
            )}

            {failure && <FormError>{failure}</FormError>}
          </div>
        </div>

        <DialogFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            onClick={submit}
            disabled={submitting || name.trim().length === 0}
          >
            {submitting ? "Queueing…" : "Pull it"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

/** How many characters of one cell reach the screen. Enough to tell prose from
    a template, short enough that five of them still fit above the form. */
const CELL_CHARS = 600

/**
 * The first few rows of the repo, as they actually are.
 *
 * This is the answer to "is this prose or conversations", which is otherwise a
 * question the pull form asks and nobody can answer without pulling first. A
 * repo the Hub has not indexed simply says so — the pull still works, it reads
 * parquet directly.
 */
function SamplePanel({
  sample,
  error,
  column,
  onPick,
}: {
  sample: HubSample | null
  error: string | null
  column: string
  onPick: (where: HubSplit) => void
}) {
  if (error) {
    return (
      <p className="py-3 text-xs text-muted-foreground">
        No preview: {error}. The pull reads the files directly and is
        unaffected.
      </p>
    )
  }

  if (!sample) {
    return (
      <div className="space-y-2">
        <Skeleton className="h-4 w-32" />
        <Skeleton className="h-24 w-full" />
      </div>
    )
  }

  const shown = column || widestColumn(sample)

  return (
    <div className="flex min-h-0 flex-col gap-2">
      <div className="flex items-center justify-between gap-3">
        <p className="text-sm font-medium">
          First {sample.rows.length} rows
          {shown ? (
            <span className="text-muted-foreground"> · {shown}</span>
          ) : null}
        </p>
        {sample.splits.length > 1 && (
          <Select
            value={`${sample.config} ${sample.split}`}
            onValueChange={(v) => {
              const [config, split] = (v ?? "").split(" ")
              if (config && split) onPick({ config, split })
            }}
          >
            <SelectTrigger
              className="h-7 w-auto text-xs"
              aria-label="Config and split"
            >
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {sample.splits.map((s) => (
                <SelectItem
                  key={`${s.config}/${s.split}`}
                  value={`${s.config} ${s.split}`}
                >
                  {s.config} / {s.split}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>

      <div className="min-h-48 flex-1 space-y-2 overflow-y-auto py-3">
        {sample.rows.length === 0 ? (
          <p className="text-xs text-muted-foreground">
            The split is indexed but empty.
          </p>
        ) : (
          sample.rows.map((row, i) => (
            <Cell key={i} value={shown ? row[shown] : row} />
          ))
        )}
      </div>

      {sample.truncated && (
        <p className="text-xs text-muted-foreground">
          Long values are cut short here.
        </p>
      )}
    </div>
  )
}

/** The column carrying the most text across the sample, which is the one worth
    showing when the form has not settled on one yet. */
function widestColumn(sample: HubSample): string | null {
  let best: string | null = null
  let width = -1
  for (const name of sample.columns) {
    const total = sample.rows.reduce((sum, row) => sum + weight(row[name]), 0)
    if (total > width) {
      width = total
      best = name
    }
  }
  return best
}

function weight(value: unknown): number {
  if (typeof value === "string") return value.length
  if (Array.isArray(value))
    return value.reduce<number>((sum, v) => sum + weight(v), 0)
  if (value && typeof value === "object") {
    return Object.values(value).reduce<number>((sum, v) => sum + weight(v), 0)
  }
  return 0
}

/** One cell, rendered as whatever it is: a conversation, prose, or raw JSON. */
function Cell({ value }: { value: unknown }) {
  const turns = asTurns(value)
  if (turns) {
    return (
      <div className="space-y-1 border-b border-border pb-2 last:border-0 last:pb-0">
        {turns.map((turn, i) => (
          <p key={i} className="text-xs">
            <span className="font-medium text-muted-foreground">
              {turn.role}:{" "}
            </span>
            <span className="whitespace-pre-wrap">{clamp(turn.content)}</span>
          </p>
        ))}
      </div>
    )
  }

  const text =
    typeof value === "string" ? value : JSON.stringify(value, null, 1)
  return (
    <p className="border-b border-border pb-2 text-xs whitespace-pre-wrap last:border-0 last:pb-0">
      {clamp(text ?? "")}
    </p>
  )
}

function asTurns(value: unknown): Turn[] | null {
  if (!Array.isArray(value) || value.length === 0) return null
  const turns = value.map((raw) => {
    if (!raw || typeof raw !== "object") return null
    const entry = raw as Record<string, unknown>
    // The same two key pairs the fetch understands: `role`/`content`, and the
    // ShareGPT `from`/`value`.
    const role = entry.role ?? entry.from
    const content = entry.content ?? entry.value
    if (typeof role !== "string" || typeof content !== "string") return null
    return { role, content }
  })
  return turns.every((t): t is Turn => t !== null) ? turns : null
}

function clamp(text: string): string {
  return text.length > CELL_CHARS ? `${text.slice(0, CELL_CHARS)}…` : text
}

/** Associates a pull-form label and optional hint with its supplied control. */
function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string
  htmlFor: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <label htmlFor={htmlFor} className="eyebrow text-muted-foreground">
        {label}
      </label>
      {children}
      {hint && <p className="text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}
