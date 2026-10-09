"use client"

import * as React from "react"
import Link from "next/link"
import { Plus } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { Throbber } from "@/components/ui/throbber"
import { toast } from "@/components/ui/toast"
import { TokenizerCatalogue } from "@/components/tokenizers/tokenizer-catalogue"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
  Dialog,
  DialogTrigger,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { TokenizerComposer } from "@/components/tokenizers/tokenizer-composer"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { useTokenizers } from "@/hooks/use-tokenizers"
import { jobHref, type Job } from "@/lib/jobs"
import { LoadError } from "@/components/ui/error-state"
import { deleteTokenizer, type TokenizerInfo } from "@/lib/tokenizers"

/**
 * A catalogue with category filters; named builds lead the combined list.
 * Building is a focused dialog, while automatic vocabularies remain inspectable
 * and deletable without competing with the form for a permanent column.
 */
export function TokenizersView() {
  const [refreshKey, setRefreshKey] = React.useState(0)
  const { tokenizers, isLoading, error } = useTokenizers(refreshKey)
  const refresh = () => setRefreshKey((k) => k + 1)

  // The composer hands off to the job page, so a build queued *here* is watched
  // there. This is for coming back: a vocabulary takes minutes, and a page that
  // lists what has been built should say when one is being built right now
  // instead of showing a stale list and no reason for it.
  //
  // Read from the app-wide subscription rather than a `useJobs()` of its own: a
  // second instance is a second poll of `GET /runs` for as long as this page is
  // open, saying the same thing as the one the provider already runs.
  const { running } = useJobsContext()
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
    <div className="space-y-6">
      {building && <BuildingBand job={building} />}
      <Tabs defaultValue="all">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <TabsList variant="line" aria-label="Vocabulary category">
            <TabsTrigger value="all">
              All{" "}
              <span className="text-xs text-muted-foreground">
                {tokenizers.length}
              </span>
            </TabsTrigger>
            <TabsTrigger value="named">
              Built{" "}
              <span className="text-xs text-muted-foreground">
                {named.length}
              </span>
            </TabsTrigger>
            <TabsTrigger value="derived">
              From runs{" "}
              <span className="text-xs text-muted-foreground">
                {derived.length}
              </span>
            </TabsTrigger>
          </TabsList>
          <Dialog>
            <DialogTrigger
              render={
                <Button size="sm">
                  <Plus className="size-3.5" />
                  Build vocabulary
                </Button>
              }
            />
            <DialogContent className="max-h-[calc(100dvh-2rem)] max-w-xl overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Build a vocabulary</DialogTitle>
              </DialogHeader>
              <TokenizerComposer onQueued={refresh} />
            </DialogContent>
          </Dialog>
        </div>
        {error ? (
          <LoadError
            title="The vocabularies didn't load."
            hint="The LLM engine may still be starting. Nothing has been lost — the vocabularies are on disk either way."
            onRetry={refresh}
          />
        ) : isLoading ? (
          <div className="space-y-3 py-4">
            {Array.from({ length: 3 }).map((_, i) => (
              <Skeleton key={i} className="h-12" />
            ))}
          </div>
        ) : (
          <>
            <TabsContent value="all" className="mt-4">
              <TokenizerCatalogue
                tokenizers={[...named, ...derived]}
                onDelete={remove}
              />
            </TabsContent>
            <TabsContent value="named" className="mt-4">
              <TokenizerCatalogue tokenizers={named} onDelete={remove} />
            </TabsContent>
            <TabsContent value="derived" className="mt-4">
              <TokenizerCatalogue tokenizers={derived} onDelete={remove} />
            </TabsContent>
          </>
        )}
      </Tabs>
      {derived.length > 0 && (
        <p className="max-w-prose text-xs leading-relaxed text-muted-foreground">
          Vocabularies learned by a run are not offered as a choice. A run that
          names no vocabulary learns one from its own corpora. It is kept and
          reused by the next run asking for the same corpora.
        </p>
      )}
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
      className="flex items-center gap-4 border-y border-border/60 py-4 transition-colors hover:bg-muted/30 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring"
    >
      <Throbber mode="tokenizing" />
      <span className="min-w-0">
        <span className="block truncate font-display text-lg">{job.label}</span>
        <span className="mt-0.5 block truncate text-xs text-muted-foreground">
          {job.stage ?? "Learning merges…"}
        </span>
      </span>
    </Link>
  )
}
