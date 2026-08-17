"use client"

import * as React from "react"
import Link from "next/link"
import { Trash2 } from "lucide-react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { Button } from "@/components/ui/button"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { Throbber } from "@/components/ui/throbber"
import { toast } from "@/components/ui/toast"
import { TokenizerComposer } from "@/components/tokenizers/tokenizer-composer"
import { useJobs } from "@/hooks/use-jobs"
import { useTokenizers } from "@/hooks/use-tokenizers"
import { jobHref, type Job } from "@/lib/jobs"
import { LoadError } from "@/components/ui/error-state"
import {
  deleteTokenizer,
  describeDatasets,
  type TokenizerInfo,
} from "@/lib/tokenizers"

function compact(n: number | null): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

function megabytes(bytes: number): string {
  if (!bytes) return "—"
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

/**
 * The vocabularies: build one, and see what has been built.
 *
 * Two lists, not one. A vocabulary built here is a choice a training run can be
 * pointed at; one a run derived for itself is the same kind of artifact and takes
 * the same disk, but nobody picked it — showing them together would put a column
 * of `auto 9c1e…` in front of the two names a person actually chose.
 */
export function TokenizersView() {
  const [refreshKey, setRefreshKey] = React.useState(0)
  const { tokenizers, isLoading, error } = useTokenizers(refreshKey)
  const refresh = () => setRefreshKey((k) => k + 1)

  // The composer hands off to the job page, so a build queued *here* is watched
  // there. This is for coming back: a vocabulary takes minutes, and a page that
  // lists what has been built should say when one is being built right now
  // instead of showing a stale list and no reason for it.
  const { running } = useJobs()
  const building = running?.kind === "tokenize" ? running : null

  const named = tokenizers.filter((t) => !t.implicit)
  const derived = tokenizers.filter((t) => t.implicit)

  const remove = async (tokenizer: TokenizerInfo) => {
    try {
      const corpora = await deleteTokenizer(tokenizer.id)
      toast.success(
        `Deleted ${tokenizer.name}.`,
        corpora > 0
          ? `${corpora} packed ${corpora === 1 ? "corpus" : "corpora"} went with it.`
          : undefined
      )
    } catch (e) {
      toast.error(
        "Couldn't delete it",
        e instanceof Error ? e.message : "Something went wrong."
      )
    } finally {
      refresh()
    }
  }

  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,22rem)_minmax(0,1fr)]">
      <div className="lg:sticky lg:top-6 lg:self-start">
        <TokenizerComposer onQueued={refresh} />
      </div>

      <div className="space-y-6">
        {building && <BuildingBand job={building} />}
        {error ? (
          <LoadError
            title="The vocabularies didn't load."
            hint="The engine may still be starting. Nothing has been lost — the vocabularies are on disk either way."
            onRetry={refresh}
          />
        ) : isLoading ? (
          <div className="space-y-3">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-20 rounded-lg" />
            ))}
          </div>
        ) : (
          <>
            <Panel
              title="Built"
              description={`${named.length} vocabular${named.length === 1 ? "y" : "ies"}`}
              bodyClassName="p-0"
            >
              {named.length === 0 ? (
                <p className="px-5 py-10 text-center text-sm text-muted-foreground">
                  None yet. Build one and every run can share it.
                </p>
              ) : (
                <ul className="divide-y divide-border/40">
                  {named.map((tokenizer) => (
                    <Row
                      key={tokenizer.id}
                      tokenizer={tokenizer}
                      onDelete={() => remove(tokenizer)}
                    />
                  ))}
                </ul>
              )}
            </Panel>

            {derived.length > 0 && (
              <Panel
                title="Learned by a run"
                description="Not offered as a choice"
                bodyClassName="p-0"
              >
                <p className="px-5 pt-4 text-xs text-muted-foreground">
                  A run that names no vocabulary learns one from its own
                  corpora. It is kept and reused by the next run asking for the
                  same corpora, which is why it takes space here.
                </p>
                <ul className="mt-2 divide-y divide-border/40">
                  {derived.map((tokenizer) => (
                    <Row
                      key={tokenizer.id}
                      tokenizer={tokenizer}
                      onDelete={() => remove(tokenizer)}
                    />
                  ))}
                </ul>
              </Panel>
            )}
          </>
        )}
      </div>
    </div>
  )
}

/**
 * A vocabulary being built, right now.
 *
 * One of the few places the mark is a *figure* rather than a dot: the waiting is
 * the subject of the band, and there is room for it. It carries `tokenizing` —
 * every patch splitting in two — which is the one state that says what this
 * particular wait is, rather than only that there is one.
 */
function BuildingBand({ job }: { job: Job }) {
  return (
    <Link
      href={jobHref(job)}
      className="flex items-center gap-4 rounded-lg border border-border bg-secondary/25 px-5 py-4 transition-colors hover:border-primary/40"
    >
      <Throbber mode="tokenizing" />
      <span className="min-w-0">
        <span className="block truncate font-display font-semibold tracking-tight">
          {job.label}
        </span>
        <span className="mt-0.5 block truncate text-xs text-muted-foreground">
          {job.stage ?? "Learning merges…"} — follow it on the job page
        </span>
      </span>
    </Link>
  )
}

function Row({
  tokenizer,
  onDelete,
}: {
  tokenizer: TokenizerInfo
  onDelete: () => void
}) {
  return (
    <li className="flex items-start justify-between gap-4 px-5 py-4">
      <div className="min-w-0 space-y-1">
        <p className="flex items-center gap-2 font-medium">
          <span className="truncate">{tokenizer.name}</span>
          {!tokenizer.hasVocab && (
            <span className="shrink-0 rounded-sm bg-destructive/10 px-2 py-0.5 text-xs text-destructive">
              missing on disk
            </span>
          )}
        </p>
        <p className="truncate text-xs text-muted-foreground">
          {describeDatasets(tokenizer)}
        </p>
        <p className="flex flex-wrap gap-x-3 text-xs text-muted-foreground tabular-nums">
          <span>{compact(tokenizer.vocabSize)} vocab</span>
          <span>{compact(tokenizer.merges)} merges</span>
          <span>{compact(tokenizer.files)} files</span>
          <span>{megabytes(tokenizer.bytes)}</span>
        </p>
      </div>

      <AlertDialog>
        <AlertDialogTrigger
          render={
            <Button variant="ghost" size="icon" aria-label="Delete">
              <Trash2 className="size-4" />
            </Button>
          }
        />
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete {tokenizer.name}?</AlertDialogTitle>
            <AlertDialogDescription>
              The corpora packed under it go too — without this vocabulary they
              are token ids nothing can read. Models trained with it keep their
              own copy and are unaffected.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={onDelete}
              className="bg-destructive text-white hover:bg-destructive/90"
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  )
}
