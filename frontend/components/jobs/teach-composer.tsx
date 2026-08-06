"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { GraduationCap } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
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
import { queueTeach } from "@/lib/jobs"
import { getTeachers, type TeacherFile } from "@/lib/synth"

const DEFAULT_COUNT = 100
const DEFAULT_CTX = 12288
const DEFAULT_TEMPERATURE = 1.0
// The tier boundaries, in target-model tokens. They sit in the *gaps* between the
// three word targets the prompt asks for (~30-75, ~330-570, ~960-1500 tokens), not
// inside a band — so an answer has to be genuinely the wrong length to be
// relabelled. The first run used 64/256 against narrower targets and the bands
// overlapped, which meant the tier was decided by BPE noise rather than by how much
// reasoning the block actually contained.

/**
 * Composing a local-teacher run: which GGUF to run, how many triples to collect,
 * and where the tier boundaries sit.
 *
 * The run asks the teacher one question and then three answers to it, each with a
 * longer thinking block. That matters more than it looks: sampling budgets
 * independently across separate conversations would give the same spread of
 * thinking lengths, but no two records would differ *only* in reasoning depth. The
 * matched triple is what makes the budget a lever the student can learn rather
 * than noise it has to average over.
 */
export function TeachComposer() {
  const router = useRouter()
  const { models, isLoading } = useModels()

  const [teachers, setTeachers] = React.useState<TeacherFile[] | null>(null)
  const [modelFile, setModelFile] = React.useState("")
  const [chosenId, setChosenId] = React.useState("")
  const [count, setCount] = React.useState(DEFAULT_COUNT)
  const [nCtx, setNCtx] = React.useState(DEFAULT_CTX)
  const [temperature, setTemperature] = React.useState(DEFAULT_TEMPERATURE)
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  React.useEffect(() => {
    let live = true
    getTeachers()
      .then((found) => {
        if (!live) return
        setTeachers(found)
        // Preselect the only one, which is the common case — a box with one model
        // in it should not need a choice made.
        if (found.length > 0) setModelFile((prev) => prev || found[0].file)
      })
      .catch(() => live && setTeachers([]))
    return () => {
      live = false
    }
  }, [])

  const targetId = chosenId || models[0]?.id || ""

  const submit = async () => {
    setIsSubmitting(true)
    try {
      const job = await queueTeach({
        teacher: { model_file: modelFile, n_ctx: nCtx },
        sample: {
          temperature,
          top_p: 0.95,
          min_p: 0.05,
          top_k: 60,
          // Three answers plus a thinking block each; a single answer's cap would
          // truncate the third.
          max_tokens: 6144,
          seed: 0,
        },
        count,
        // The tier boundaries are deliberately not sent. They are derived from the
        // student's context window, which is the only thing that can decide them:
        // a long thinking block has to leave room for the reply inside the window,
        // and a hand-set bound that ignores this makes the long tier unfillable.
        id_model: targetId || undefined,
      })
      // The device lane, unlike `synth` — worth saying, because it means this will
      // sit behind (and block) whatever else wants the GPU.
      toast.success("Teaching queued on the GPU line.", job.label)
      router.push(`/jobs/${job.id_job}`)
    } catch (e) {
      toast.error(
        "Couldn't queue the run",
        e instanceof Error ? e.message : "Something went wrong."
      )
      setIsSubmitting(false)
    }
  }

  const valid =
    modelFile.trim() !== "" && count >= 1 && nCtx >= 1024 && temperature > 0

  return (
    <div className="space-y-6">
      <Panel title="Teacher" description="Which local model writes the answers">
        {teachers != null && teachers.length === 0 ? (
          <p className="rounded-lg border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
            No <code className="text-xs">.gguf</code> in the teacher directory
            yet. Download a quantized instruct model into{" "}
            <code className="text-xs">data/artifacts/teacher/</code> and it will
            appear here. A 4B at Q4_K_M is around 2.4&nbsp;GB and fits this card
            comfortably; prefer a non-reasoning variant, since a model that
            emits its own thinking tags fights the ones we ask it for.
          </p>
        ) : (
          <Select
            value={modelFile}
            onValueChange={(value) => setModelFile(value ?? "")}
            disabled={isSubmitting || teachers == null}
          >
            <SelectTrigger className="w-full min-w-0 sm:w-[28rem]">
              <SelectValue>
                {(value: string | null) =>
                  value ?? (teachers == null ? "Looking…" : "Select a model")
                }
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {(teachers ?? []).map((t) => (
                <SelectItem key={t.file} value={t.file}>
                  {t.file} · {(t.bytes / 1e9).toFixed(1)} GB
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        <dl className="mt-4 space-y-3">
          <Field
            label="Context window"
            hint="The binding VRAM cost after the weights themselves — the KV cache scales linearly with it. It has to hold the exemplars, the instructions and three answers, so this is not a field to shrink casually."
          >
            <Input
              type="number"
              min={1024}
              step={1024}
              value={nCtx}
              onChange={(e) => setNCtx(Number(e.target.value))}
              disabled={isSubmitting}
              className="h-8 w-28"
            />
          </Field>
          <Field
            label="Temperature"
            hint="High on purpose: this is a diversity task, and duplicate questions are skipped anyway."
          >
            <Input
              type="number"
              min={0.1}
              step={0.1}
              value={temperature}
              onChange={(e) => setTemperature(Number(e.target.value))}
              disabled={isSubmitting}
              className="h-8 w-24"
            />
          </Field>
        </dl>
      </Panel>

      <Panel
        title="Target"
        description="Whose tokenizer measures the thinking budget"
      >
        <Select
          value={targetId}
          onValueChange={(value) => setChosenId(value ?? "")}
          disabled={isSubmitting || isLoading}
        >
          <SelectTrigger className="w-full min-w-0 sm:w-80">
            <SelectValue>
              {(value: string | null) =>
                models.find((m) => m.id === value)?.name ??
                (isLoading ? "Loading models…" : "Select a model")
              }
            </SelectValue>
          </SelectTrigger>
          <SelectContent>
            {models.map((m) => (
              <SelectItem key={m.id} value={m.id}>
                {m.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        <p className="mt-3 text-xs text-muted-foreground">
          A thinking budget is what <em>this</em> model spends at inference, so
          the tiers are counted in its tokens — not the teacher&apos;s, and not
          in words. They are derived from its context window, because a long
          thinking block still has to leave room for the reply beside it: a
          1024-token student cannot hold 700 tokens of reasoning and an answer,
          so a bound set by hand can make the long tier unfillable. The run
          reports the boundaries it resolved before it starts generating. Skip
          the model and answers are filed under whichever tier was asked for
          rather than the one they landed in, which makes the labels a record of
          the request instead of the content.
        </p>
      </Panel>

      <Panel title="Run size" description="How much to generate">
        <dl className="space-y-3">
          <Field
            label="Triples"
            hint="One question with three answers each. The run stops here, or when the call budget runs out."
          >
            <Input
              type="number"
              min={1}
              value={count}
              onChange={(e) => setCount(Number(e.target.value))}
              disabled={isSubmitting}
              className="h-8 w-24"
            />
          </Field>
        </dl>

        <div className="mt-5 flex items-center gap-3">
          <Button
            onClick={submit}
            disabled={isSubmitting || !valid}
            className="gap-2"
          >
            <GraduationCap className="size-3.5" />
            {isSubmitting ? "Queueing…" : "Queue teaching"}
          </Button>
          <span className="text-xs text-muted-foreground">
            Takes the GPU line — training and chat wait behind it.
          </span>
        </div>
      </Panel>

      <p className="rounded-lg border border-dashed border-border px-4 py-3 text-xs text-muted-foreground">
        Answers are appended to <code className="text-xs">think-low.jsonl</code>
        , <code className="text-xs">think-med.jsonl</code> and{" "}
        <code className="text-xs">think-high.jsonl</code> in the SFT directory —
        flat files rather than a subfolder, because everything that reads{" "}
        <code className="text-xs">raw/sft/</code> globs it one level deep and
        would silently not see nested data. Each record carries the id of the
        triple it came from, so the three can be put back together later.
      </p>
    </div>
  )
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint: string
  children: React.ReactNode
}) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-x-6 gap-y-1">
      <div className="min-w-0 flex-1">
        <dt className="text-sm">{label}</dt>
        <dd className="mt-0.5 text-xs text-muted-foreground">{hint}</dd>
      </div>
      {children}
    </div>
  )
}
