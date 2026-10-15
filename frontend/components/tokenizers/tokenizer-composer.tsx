"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { PageHeader } from "@/components/ui/page-header"
import { useSessionToken } from "@/hooks/use-auth"
import { getToken } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { toast } from "@/components/ui/toast"
import { DatasetPicker, Pow2Input } from "@/components/training/dataset-picker"
import { useDatasets } from "@/hooks/use-datasets"
import { queueTokenizer, jobHref } from "@/lib/jobs"
import { validateTokenizerFields, TOKENIZER_NAME_LIMIT } from "@/lib/tokenizers"
import type { DatasetSelection } from "@/lib/training"

const DEFAULT_MAX_VOCAB = 8192

// The vocabulary can be larger than any other power-of-two field here, so the
// exponent ceiling is its own: 2^16 is the ceiling the corpus format sets, since
// ids are packed as uint16.
const MAX_VOCAB_EXP = 16

/**
 * Composing a vocabulary: which corpora to learn merges from, and how many.
 *
 * A form only. Submitting queues a `tokenize` job and hands off to its job page,
 * where the frames are — the same division every composer here has.
 */
export function TokenizerComposer() {
  const token = useSessionToken()
  return token ? <TokenizerComposerForm key={token} token={token} /> : null
}

/** Owns one session's vocabulary build draft and ignores stale queue completions. */
function TokenizerComposerForm({ token }: { token: string }) {
  const router = useRouter()
  const { datasets: available, isLoading } = useDatasets()
  const [name, setName] = React.useState("")
  const [maxVocab, setMaxVocab] = React.useState(DEFAULT_MAX_VOCAB)
  const [picked, setPicked] = React.useState<DatasetSelection[]>([])
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  const active = React.useRef(false)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Queues the chosen vocabulary and opens its durable build page for the originating session. */
  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    if (isSubmitting || getToken() !== token) return
    setIsSubmitting(true)
    try {
      const job = await queueTokenizer({
        tokenizer_id: validateTokenizerFields({ name, description: "" }).name,
        max_vocab: maxVocab,
        datasets: picked,
      })
      if (!active.current || getToken() !== token) return
      toast.success("Vocabulary build queued.")
      router.push(jobHref(job))
    } catch (e) {
      if (!active.current || getToken() !== token) return
      toast.error(
        "Couldn't queue the build",
        e instanceof Error ? e.message : "Something went wrong."
      )
      setIsSubmitting(false)
    }
  }

  return (
    <form onSubmit={submit} className="space-y-5">
      <PageHeader
        back={{ href: "/tokenizers", label: "Tokenizers" }}
        contentClassName="min-w-48 flex-1"
        title={
          <>
            <span className="sr-only">New tokenizer</span>
            <Input
              aria-label="Vocabulary name"
              autoFocus
              required
              maxLength={TOKENIZER_NAME_LIMIT * 2}
              variant="inline"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Vocabulary name"
              disabled={isSubmitting}
              className="h-auto px-0 py-0 font-display text-3xl font-semibold tracking-tight md:text-3xl"
            />
          </>
        }
        action={
          <Button type="submit" disabled={isSubmitting || !name.trim()}>
            {isSubmitting ? "Queueing…" : "Build vocabulary"}
          </Button>
        }
      />

      <div className="flex items-center justify-between gap-3 py-1.5">
        <span className="text-sm text-muted-foreground">Max vocab</span>
        <Pow2Input
          ariaLabel="Maximum vocabulary exponent"
          value={maxVocab}
          maxExp={MAX_VOCAB_EXP}
          disabled={isSubmitting}
          onChange={setMaxVocab}
        />
      </div>

      <DatasetPicker
        available={available}
        loading={isLoading}
        value={picked}
        disabled={isSubmitting}
        onChange={setPicked}
        emptyHint="No corpora yet. Pull one from the Data view first."
      />

      <p className="text-xs text-muted-foreground">
        {picked.length === 0
          ? "Nothing picked — it will learn from every corpus on disk."
          : "A run pointed at this vocabulary uses it whatever it trains on."}
      </p>
    </form>
  )
}
