"use client"

import * as React from "react"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Panel } from "@/components/ui/panel"
import { toast } from "@/components/ui/toast"
import { DatasetPicker, Pow2Input } from "@/components/training/dataset-picker"
import { useDatasets } from "@/hooks/use-datasets"
import { queueTokenizer } from "@/lib/jobs"
import type { DatasetSelection } from "@/lib/training"

const DEFAULT_MAX_VOCAB = 8192

// The vocabulary can be larger than any other power-of-two field here, so the
// exponent ceiling is its own: 2^16 is the ceiling the corpus format sets, since
// ids are packed as uint16.
const MAX_VOCAB_EXP = 16

export interface TokenizerComposerProps {
  /** Called once a build is queued, so the list beside this can refetch. */
  onQueued?: () => void
}

/**
 * Composing a vocabulary: which corpora to learn merges from, and how many.
 *
 * A form only. Submitting queues a `tokenize` job and hands off to its job page,
 * where the frames are — the same division every composer here has.
 */
export function TokenizerComposer({ onQueued }: TokenizerComposerProps) {
  const router = useRouter()
  const { datasets: available, isLoading } = useDatasets()
  const [name, setName] = React.useState("")
  const [maxVocab, setMaxVocab] = React.useState(DEFAULT_MAX_VOCAB)
  const [picked, setPicked] = React.useState<DatasetSelection[]>([])
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  const submit = async (event: React.FormEvent) => {
    event.preventDefault()
    setIsSubmitting(true)
    try {
      const job = await queueTokenizer({
        // The typed name is what to call it; the engine mints the id the
        // vocabulary is stored under and sends this back as its display name.
        tokenizer_id: name.trim(),
        max_vocab: maxVocab,
        datasets: picked,
      })
      toast.success("Vocabulary build queued.")
      onQueued?.()
      router.push(`/jobs/${job.id_job}`)
    } catch (e) {
      toast.error(
        "Couldn't queue the build",
        e instanceof Error ? e.message : "Something went wrong."
      )
      setIsSubmitting(false)
    }
  }

  return (
    <Panel
      title="Build a vocabulary"
      description="Learned once, reused by any run"
    >
      <form onSubmit={submit} className="space-y-5">
        <div className="space-y-1.5">
          <label className="text-xs text-muted-foreground">Name</label>
          <Input
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="Prose 8k"
            disabled={isSubmitting}
            className="h-10 font-heading text-lg font-semibold"
          />
        </div>

        <div className="flex items-center justify-between gap-3 py-1.5">
          <span className="text-sm text-muted-foreground">Max vocab</span>
          <Pow2Input
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
          emptyHint="No corpora on disk yet — fetch some into data/input/ first."
        />

        {/* Empty is a real request, not an unfinished one: the engine resolves it
            to every corpus on disk, at the moment the build runs. */}
        <p className="text-xs text-muted-foreground">
          {picked.length === 0
            ? "Nothing picked — it will learn from every corpus on disk."
            : "A run pointed at this vocabulary uses it whatever it trains on."}
        </p>

        <Button
          type="submit"
          className="w-full"
          disabled={isSubmitting || name.trim().length === 0}
        >
          Build the vocabulary
        </Button>
      </form>
    </Panel>
  )
}
