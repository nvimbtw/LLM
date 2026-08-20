"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Panel } from "@/components/ui/panel"
import { toast } from "@/components/ui/toast"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useModels } from "@/hooks/use-models"
import {
  BENCHMARK_SUITES,
  getSuite,
  type BenchmarkSuite,
  type SuiteInfo,
} from "@/lib/benchmark"
import { queueBenchmark } from "@/lib/jobs"

/**
 * Composing a benchmark: pick a model, queue it.
 *
 * Was the run bar on /benchmark. It moved because starting a run is now queueing a
 * job, and that page is where you *read* scores — a control that puts work on the
 * GPU sitting above a history table was the reason a benchmark could previously be
 * fired while training held the device.
 */
export function BenchmarkComposer() {
  const router = useRouter()
  const { models, isLoading } = useModels()
  const [suite, setSuite] = React.useState<SuiteInfo | null>(null)
  const [chosenId, setChosenId] = React.useState("")
  // Which of the two benchmarks to run. They are separate runs on separate scales,
  // so this is a choice rather than a mode — queue both if you want both.
  const [which, setWhich] = React.useState<BenchmarkSuite>("language")
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  // Only the *explicit* choice is stored; the default is derived from the list.
  // Storing the default instead would mean writing state from an effect the moment
  // models load, for a value that is already computable from what we have.
  const modelId = chosenId || models[0]?.id || ""

  React.useEffect(() => {
    let alive = true
    getSuite()
      .then((info) => {
        if (alive) setSuite(info)
      })
      .catch(() => {})
    return () => {
      alive = false
    }
  }, [])

  // Success navigates to the new job, so the confirmation has to outlive the
  // page it was triggered from — which is what a toast is. A rejection is
  // likewise an event, not a property of the form, and the message is the
  // server's own 400 body.
  const submit = async () => {
    setIsSubmitting(true)
    try {
      const job = await queueBenchmark(modelId, which)
      toast.success("Benchmark queued.", job.label)
      router.push(`/jobs/${job.id_job}`)
    } catch (e) {
      toast.error(
        "Couldn't queue the benchmark",
        e instanceof Error ? e.message : "Something went wrong."
      )
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Panel title="Model" description="What to score">
        <div className="flex flex-wrap items-center gap-3">
          {/* The primitive can emit null on clear; the form always wants a
              concrete id, so a clear falls back to the empty string. */}
          <Select
            value={modelId}
            onValueChange={(value) => setChosenId(value ?? "")}
            disabled={isSubmitting || isLoading}
          >
            <SelectTrigger className="w-full min-w-0 sm:w-80">
              {/* Base UI's Select.Value renders the raw `value` — here a UUID — so
                  the label has to be looked up explicitly. The render function also
                  supersedes `placeholder`, so the empty case is handled here too. */}
              <SelectValue>
                {(value: string | null) =>
                  models.find((model) => model.id === value)?.name ??
                  (isLoading ? "Loading models…" : "Select a model")
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

          <Select
            value={which}
            onValueChange={(value) =>
              setWhich((value as BenchmarkSuite | null) ?? "language")
            }
            disabled={isSubmitting}
          >
            <SelectTrigger className="w-full min-w-0 sm:w-44">
              <SelectValue>
                {(value: string | null) =>
                  BENCHMARK_SUITES.find((s) => s.id === value)?.label ??
                  "Language"
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {BENCHMARK_SUITES.map((option) => (
                <SelectItem key={option.id} value={option.id}>
                  {option.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          <Button
            onClick={submit}
            disabled={isSubmitting || !modelId}
            className="gap-2"
          >
            <Play className="size-3.5" />
            {isSubmitting ? "Queueing…" : "Queue benchmark"}
          </Button>
        </div>
      </Panel>

      {/* The markup suite's pair stage needs nothing on disk, so an absent chat set
          is a partial score rather than a broken run — and it is written from your
          SFT conversations at the start of every benchmark, so it is usually a
          statement about the first run rather than about the install. */}
      {which === "markup" && suite && suite.chat_files === 0 && (
        <p className="rounded-(--radius-sheet) border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
          No held-out chat set in{" "}
          <code className="text-xs">{suite.chat_dir}</code> yet. It is exported
          from your SFT conversations when this run starts, so this run scores
          the marker pairs and the next one scores everything.
        </p>
      )}
    </div>
  )
}
