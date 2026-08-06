"use client"

import * as React from "react"
import { Sparkles } from "lucide-react"

import { ButtonLink } from "@/components/ui/button-link"
import { PageHeader } from "@/components/ui/page-header"
import { Panel } from "@/components/ui/panel"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { StatRun } from "@/components/training/stat-run"
import { useCorpus } from "@/hooks/use-corpus"
import { useModels } from "@/hooks/use-models"
import { CorpusFigures } from "./corpus-figures"

/**
 * The SFT corpus: what is in it, and how much of it the model can actually hold.
 *
 * Two questions, in that order. "How many do we have" is the count everyone asks
 * first and it is the less useful one — a corpus of four hundred conversations that
 * are each three times the context window trains on fragments. So the headline row
 * carries both, and the figure below is the one that decides whether a run is worth
 * starting.
 */
export function CorpusView() {
  const { models, isLoading: modelsLoading } = useModels()
  const [chosenId, setChosenId] = React.useState("")
  // Defaults to the newest model, which is nearly always the one about to be
  // fine-tuned — and the measurement is meaningless without *some* model, so
  // waiting for a pick would show an empty page for no reason.
  const modelId = chosenId || models[0]?.id || ""
  const { corpus, isLoading, error } = useCorpus(modelId || undefined)

  const fits = corpus ? countedTokens(corpus.files) - corpus.over_budget : 0
  const counted = corpus ? countedTokens(corpus.files) : 0

  return (
    <div className="space-y-6">
      <PageHeader
        title="SFT corpus"
        description="The conversations supervised fine-tuning trains on — hand-written and generated, measured against the window they have to fit."
        action={
          <ButtonLink href="/jobs/new/teach" className="gap-2">
            <Sparkles className="size-3.5" />
            Generate more
          </ButtonLink>
        }
      />

      <Panel
        title="Measured against"
        description="Whose tokenizer and context window"
      >
        <Select
          value={modelId}
          onValueChange={(value) => setChosenId(value ?? "")}
          disabled={modelsLoading}
        >
          <SelectTrigger className="w-full min-w-0 sm:w-80">
            <SelectValue>
              {(value: string | null) =>
                models.find((model) => model.id === value)?.name ??
                (modelsLoading ? "Loading models…" : "Select a model")
              }
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {models.map((model) => (
              <SelectItem key={model.id} value={model.id}>
                {model.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="mt-3 text-xs text-muted-foreground">
          Lengths are real tokens: every conversation is rendered through the
          same chat template and merges the training run will use, so the
          boundary drawn below is the one that will actually apply.
        </p>
      </Panel>

      {error && (
        <p className="rounded-lg border border-dashed border-border px-4 py-3 text-sm text-destructive">
          {error}
        </p>
      )}

      {isLoading && !corpus ? (
        <p className="py-16 text-center text-sm text-muted-foreground">
          Loading…
        </p>
      ) : corpus ? (
        <>
          <StatRun
            stats={[
              {
                label: "Conversations",
                value: corpus.conversations.toLocaleString(),
              },
              {
                label: "Fit the window",
                value: counted > 0 ? `${fits.toLocaleString()}` : "—",
              },
              {
                label: "Too long",
                value: counted > 0 ? corpus.over_budget.toLocaleString() : "—",
              },
              {
                label: "Context window",
                value: corpus.ctx_window ? `${corpus.ctx_window}` : "—",
              },
              { label: "Files", value: `${corpus.files.length}` },
            ]}
          />

          <Panel
            title="Length"
            description="Where the corpus sits"
            bodyClassName="pt-6"
          >
            <CorpusFigures corpus={corpus} />
          </Panel>

          <Panel title="Files" description="Every .jsonl the SFT build reads">
            {/* Wide enough to overflow on a phone, so it scrolls inside its own
                box rather than pushing the page sideways. */}
            <div className="overflow-x-auto">
              <table className="w-full min-w-[34rem] text-sm">
                <thead className="text-xs text-muted-foreground">
                  <tr className="border-b border-border text-left">
                    <th className="pr-4 pb-2 font-normal">File</th>
                    <th className="pr-4 pb-2 text-right font-normal">Convs</th>
                    <th className="pr-4 pb-2 text-right font-normal">Turns</th>
                    <th className="pr-4 pb-2 text-right font-normal">
                      Thinking
                    </th>
                    <th className="pr-4 pb-2 text-right font-normal">
                      Too long
                    </th>
                    <th className="pb-2 text-right font-normal">
                      Median reply
                    </th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {corpus.files.map((file) => (
                    <tr
                      key={file.file}
                      className="border-b border-border/40 last:border-0"
                    >
                      <td className="py-2 pr-4 font-mono text-xs">
                        {file.file}
                      </td>
                      <td className="py-2 pr-4 text-right">
                        {file.conversations}
                      </td>
                      <td className="py-2 pr-4 text-right">{file.turns}</td>
                      <td className="py-2 pr-4 text-right">
                        {file.with_thinking}
                      </td>
                      <td className="py-2 pr-4 text-right">
                        {corpus.ctx_window ? file.over_budget : "—"}
                      </td>
                      <td className="py-2 text-right">
                        {file.assistant_chars_median.toLocaleString()} ch
                      </td>
                    </tr>
                  ))}
                  {corpus.files.length === 0 && (
                    <tr>
                      <td
                        colSpan={6}
                        className="py-6 text-center text-muted-foreground"
                      >
                        Nothing in the SFT directory yet.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </Panel>
        </>
      ) : null}
    </div>
  )
}

// Conversations that actually got measured. Without a model there are no token
// lengths, and "0 fit" would be a lie rather than an absence.
function countedTokens(files: { tokens: number[] }[]): number {
  return files.reduce((n, f) => n + f.tokens.length, 0)
}
