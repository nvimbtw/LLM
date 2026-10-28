"use client"

import * as React from "react"
import Link from "next/link"
import { Plus } from "lucide-react"

import { useSessionToken } from "@/hooks/use-auth"
import { ButtonLink } from "@/components/ui/button-link"
import { Skeleton } from "@/components/ui/skeleton"
import { Throbber } from "@/components/ui/throbber"
import { toast } from "@/components/ui/toast"
import { TokenizerCatalogue } from "@/components/tokenizers/tokenizer-catalogue"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { useJobsContext } from "@/components/jobs/jobs-provider"
import { useTokenizers } from "@/hooks/use-tokenizers"
import { jobHref, type Job } from "@/lib/jobs"
import { LoadError } from "@/components/ui/error-state"
import { deleteTokenizer, type TokenizerInfo } from "@/lib/tokenizers"
import { getToken } from "@/lib/api"

/**
 * A catalogue with category filters; named builds lead the combined list.
 * Creation has its own page; automatic vocabularies retain their provenance and confirmed deletion.
 */
export function TokenizersView() {
  const token = useSessionToken()
  return token ? <TokenizersCatalogue key={token} token={token} /> : null
}

/** Owns one session's vocabulary catalogue, category tabs and deletion refresh. */
function TokenizersCatalogue({ token }: { token: string }) {
  const active = React.useRef(false)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])
  const [refreshKey, setRefreshKey] = React.useState(0)
  const { tokenizers, isLoading, error } = useTokenizers(refreshKey)
  const refresh = () => setRefreshKey((k) => k + 1)

  const { running } = useJobsContext()
  const building = running?.kind === "tokenize" ? running : null

  const named = tokenizers.filter((t) => !t.implicit)
  const derived = tokenizers.filter((t) => t.implicit)

  /** Deletes a confirmed vocabulary, reports failures and refreshes this catalogue. */
  const remove = async (tokenizer: TokenizerInfo) => {
    if (getToken() !== token) return
    try {
      const corpora = await deleteTokenizer(tokenizer.id)
      if (!active.current || getToken() !== token) return
      toast.success(
        `Deleted ${tokenizer.name}.`,
        corpora > 0
          ? `${corpora} packed ${corpora === 1 ? "corpus" : "corpora"} went with it.`
          : undefined
      )
      refresh()
    } catch (e) {
      if (!active.current || getToken() !== token) return
      throw e
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
          <ButtonLink href="/tokenizers/new-tokenizer" size="sm">
            <Plus className="size-3.5" />
            Build vocabulary
          </ButtonLink>
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
      className="flex items-center gap-4 py-4 transition-colors hover:bg-muted/30 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring"
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
