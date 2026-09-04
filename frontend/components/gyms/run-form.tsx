"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
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
import { createGymRun, LEVELS, type Gym } from "@/lib/gyms"

/** What one run may ask for — the gyms service refuses more (`MAX_EPISODES`). */
const MAX_EPISODES = 32
const DEFAULT_EPISODES = 5

/**
 * The smallest trained window a task actually fits in.
 *
 * Measured, not guessed: the shortest task plus its system block is ~72 tokens,
 * and `build_prompt` reserves half the window for the reply — so 256 is the
 * floor before the shell has answered anything, and a tool result has to fit
 * after that. Below it the prompt is truncated (the engine cuts it rather than
 * failing) and the model is asked a question it was never shown, which reads as
 * a model scoring zero when it is really a window that could not hold the
 * question.
 *
 * Raising the context multiplier in settings stretches a smaller model past
 * this; it is not a wall, which is why this is a warning and not a refusal.
 */
const MIN_CONTEXT = 256

/**
 * Starting a run: a model, the kinds of task to draw from, how many.
 *
 * Creating a run does not run it — it generates the tasks and seeds a sandbox
 * for each, which is the slow part and the part worth doing before anyone is
 * watching. The run page below drives it.
 */
export function RunForm({ gym }: { gym: Gym }) {
  const router = useRouter()
  const { models, isLoading } = useModels()
  const [chosenId, setChosenId] = React.useState("")
  // The easy rung of every family, not all ten types.
  //
  // A run mixing every level is the shape that produced ten runs of 0.00 and no
  // way to read them: a model that can list a directory and not read a file
  // scores the same as one that can do neither. Starting at the bottom of each
  // ladder is what makes the first non-zero score appear, and everything harder
  // is one checkbox away. Falls back to the whole set for a gym that declares
  // no levels.
  const [taskTypes, setTaskTypes] = React.useState<string[]>(() => {
    const easy = gym.task_types.filter((t) => t.level === "easy")
    return (easy.length > 0 ? easy : gym.task_types).map((t) => t.id)
  })
  const [count, setCount] = React.useState(DEFAULT_EPISODES)
  const [groupSize, setGroupSize] = React.useState(1)
  const [seed, setSeed] = React.useState("")
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  // The model's **id**, not its name. `model_name` is the engine's field and it
  // has always carried the id — that is the directory the weights are under
  // (`artifacts/models/<id>/model.pt`), and a display name has no path. Chat
  // sends the same thing under the same misleading key.
  // The task types as columns: one per family, each ordered easy to hard.
  // Registry order decides the columns and `LEVELS` decides the rows, so
  // neither is spelled out here — a new family or rung arrives from `GET /gyms`
  // and lands in the right place. A gym that declares no families collapses to
  // one unlabelled column, which is what the grouping degrades to rather than
  // breaking on.
  const families = React.useMemo(() => {
    const rung = (level: string) => LEVELS.findIndex((l) => l === level)
    const order = [...new Set(gym.task_types.map((t) => t.family))]
    return order.map((family) => ({
      family,
      types: gym.task_types
        .filter((t) => t.family === family)
        .sort((a, b) => rung(a.level) - rung(b.level)),
    }))
  }, [gym.task_types])

  const modelId = chosenId || models[0]?.id || ""
  // The service refuses a count that leaves a partial group, so the form does
  // too rather than letting a 400 be the way anyone finds out.
  const partialGroup = count % groupSize !== 0
  const ready = modelId && taskTypes.length > 0 && count > 0 && !partialGroup
  const chosen = models.find((model) => model.id === modelId)
  const cramped =
    chosen !== undefined && chosen.contextWindow > 0 && chosen.contextWindow < MIN_CONTEXT

  const toggle = (id: string) =>
    setTaskTypes((chosen) =>
      chosen.includes(id) ? chosen.filter((t) => t !== id) : [...chosen, id]
    )

  const submit = async () => {
    setIsSubmitting(true)
    try {
      const run = await createGymRun({
        gymId: gym.id,
        modelName: modelId,
        taskTypes,
        count,
        // 1 is an ordinary evaluation and is left off the wire entirely.
        ...(groupSize > 1 ? { groupSize } : {}),
        // A blank seed is a fresh draw; a given one makes the run repeatable.
        ...(seed.trim() === "" ? {} : { seedBase: Number(seed) }),
      })
      router.push(`/gyms/${gym.id}/runs/${run.id}`)
    } catch (e) {
      toast.error(
        "Couldn't create the run",
        e instanceof Error ? e.message : "Something went wrong."
      )
      setIsSubmitting(false)
    }
  }

  return (
    <Panel
      title="New run"
      description={`${gym.name} — one sandbox per task, seeded before anything runs.`}
    >
      <div className="space-y-5">
        <div className="grid gap-4 sm:grid-cols-[minmax(0,1fr)_6rem_6rem_6rem]">
          <Field label="Model">
            <Select value={modelId} onValueChange={(value) => setChosenId(value ?? "")}>
              <SelectTrigger disabled={isLoading || models.length === 0}>
                {/* The value is the model's id, so the label has to be looked
                    up — the same explicit render the benchmark composer needs,
                    and for the same reason. It supersedes `placeholder`, so the
                    empty case is handled here too. */}
                <SelectValue>
                  {(value: string | null) =>
                    models.find((model) => model.id === value)?.name ??
                    (isLoading ? "Loading models…" : "No models")
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
          </Field>

          <Field label="Tasks">
            <Input
              type="number"
              min={1}
              max={MAX_EPISODES}
              value={count}
              onChange={(e) => setCount(clamp(e.target.valueAsNumber))}
            />
          </Field>

          <Field label="Rollouts" hint="attempts per task">
            <Input
              type="number"
              min={1}
              max={MAX_EPISODES}
              value={groupSize}
              onChange={(e) => setGroupSize(clamp(e.target.valueAsNumber))}
            />
          </Field>

          <Field label="Seed" hint="blank draws one">
            <Input
              inputMode="numeric"
              placeholder="random"
              value={seed}
              onChange={(e) => setSeed(e.target.value.replace(/\D/g, ""))}
            />
          </Field>
        </div>

        <Field label="Task types">
          {/* A column per family, rungs in order down it.
              Grouping by *level* instead put ten checkboxes through a flex wrap
              and let a row of four orphan its last item onto a line of its own —
              and it broke the thing worth seeing, which is that each column is
              one skill getting harder. Read down for the ladder, across for the
              three skills. */}
          <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
            {families.map(({ family, types }) => (
              <div key={family} className="space-y-2.5">
                {family && (
                  <p className="text-xs uppercase tracking-wide text-muted-foreground">
                    {family}
                  </p>
                )}
                {types.map((type) => (
                  <label key={type.id} className="flex items-start gap-2.5 text-sm">
                    <Checkbox
                      className="mt-0.5 shrink-0"
                      checked={taskTypes.includes(type.id)}
                      onCheckedChange={() => toggle(type.id)}
                    />
                    <span className="min-w-0">
                      <span className="flex items-baseline gap-2">
                        {type.label}
                        <span className="text-[10px] uppercase tracking-wide text-muted-foreground">
                          {type.level}
                        </span>
                      </span>
                      <span className="block text-xs text-muted-foreground">
                        {type.description}
                      </span>
                    </span>
                  </label>
                ))}
              </div>
            ))}
          </div>
        </Field>

        {groupSize > 1 && (
          <p className="rounded-(--radius-sheet) border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
            <span className="text-foreground">
              {count / groupSize || 0} task{count / groupSize === 1 ? "" : "s"},
              each attempted {groupSize} times.
            </span>{" "}
            A group of repeated attempts at one task is the baseline a policy
            gradient measures a rollout against — so this only means anything at
            a temperature above zero and with no sampler seed, or the attempts
            are one attempt counted {groupSize} times.
          </p>
        )}

        {partialGroup && (
          <p className="rounded-(--radius-sheet) border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
            {count} tasks does not divide into groups of {groupSize}. The last
            group would be smaller than the rest, and a group of one has no
            siblings to be better than.
          </p>
        )}

        {cramped && (
          <p className="rounded-(--radius-sheet) border border-border bg-muted/40 px-3 py-2 text-xs text-muted-foreground">
            <span className="text-foreground">
              {chosen?.name} trained at {chosen?.contextWindow} tokens.
            </span>{" "}
            A task needs about {MIN_CONTEXT} to be asked in full — below that the
            prompt is cut and the model answers a question it was only shown part
            of. Raise the context multiplier in settings, or pick a wider model.
          </p>
        )}

        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <Button onClick={submit} disabled={!ready || isSubmitting}>
            <Play className="size-4" />
            {isSubmitting ? "Preparing…" : "Create run"}
          </Button>
          {/* A run created here happens once: it measures the model as it is,
              and nothing about it is fed back into the weights. The loop —
              roll out, grade, step, checkpoint, again — is a queued job, and
              this is the only place anyone would look for it. */}
          <p className="text-xs text-muted-foreground">
            This scores the model once.{" "}
            <Link
              href="/jobs/new/rl"
              className="text-primary underline underline-offset-2"
            >
              Practise against this gym
            </Link>{" "}
            to keep going: every iteration is a run like this one, and what it
            scores is fed back into the weights.
          </p>
        </div>
      </div>
    </Panel>
  )
}

function Field({
  label,
  hint,
  children,
}: {
  label: string
  hint?: string
  children: React.ReactNode
}) {
  return (
    <div className="space-y-1.5">
      <p className="eyebrow text-muted-foreground">
        {label}
        {hint && <span className="ml-2 normal-case opacity-70">{hint}</span>}
      </p>
      {children}
    </div>
  )
}

function clamp(value: number): number {
  if (!Number.isFinite(value)) return 1
  return Math.min(MAX_EPISODES, Math.max(1, Math.round(value)))
}
