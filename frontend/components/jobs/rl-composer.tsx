"use client"

import * as React from "react"
import { SourceWeights } from "@/components/gyms/source-weights"
import { selectedSourceWeights } from "@/lib/gyms"
import { useRouter } from "next/navigation"
import { Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { FormError, LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useModels } from "@/hooks/use-models"
import { listGyms, LEVELS, type Gym } from "@/lib/gyms"
import { jobHref, queueReinforcement } from "@/lib/jobs"
import { useJobSubmission } from "@/hooks/use-job-submission"
import { identityDigest } from "@/lib/architecture/decoder"
import { DemonstrationsForm } from "@/components/gyms/demonstrations-form"

/**
 * What one gym run may ask for — `MAX_EPISODES` in `gyms-engine/api/routes/gym.py`,
 * which exists because every episode holds a sandbox filesystem and just-bash
 * keeps 256 of them.
 *
 * An iteration *is* a gym run, so `tasks x rollouts` is bounded by it. The form
 * refuses a batch over the ceiling rather than letting a run queue happily and
 * then die on its first iteration with a 400 nobody is watching for.
 */
const MAX_EPISODES = 256

/**
 * Composing a practice run: the loop the gym page does not have.
 *
 * Creating a *gym run* evaluates a model — it draws tasks, the browser drives
 * them, and it ends. This queues the thing that repeats: each iteration is one
 * gym run of its own, graded, stepped on, and checkpointed, `iterations` times.
 * That difference is the whole reason this is a job on the queue and starting a
 * gym run is not.
 *
 * The fields are the LLM engine's `RlConfig` under its own names, for the reason
 * `queueReinforcement` gives: a second vocabulary for one form is a translation
 * layer to keep true forever.
 */
export function RlComposer({ gym: preselected }: { gym?: string }) {
  const router = useRouter()
  const { models, isLoading } = useModels()
  const [gyms, setGyms] = React.useState<Gym[] | null>(null)
  const [loadError, setLoadError] = React.useState<string | null>(null)

  const [name, setName] = React.useState("")
  const [chosenBase, setChosenBase] = React.useState("")
  const [chosenGym, setChosenGym] = React.useState(preselected ?? "")
  // `null` is "nothing chosen yet", which is not the same as "nothing chosen":
  // an empty list has to survive, or unchecking the last box silently ticks the
  // defaults back on and the form disagrees with itself about being ready.
  const [taskTypes, setTaskTypes] = React.useState<string[] | null>(null)
  const [gymWeights, setGymWeights] = React.useState<Record<string, number>>({})
  const [recoverySaves, setRecoverySaves] = React.useState(10)
  const [iterations, setIterations] = React.useState(50)
  const [tasksPerBatch, setTasksPerBatch] = React.useState(4)
  const [groupSize, setGroupSize] = React.useState(8)
  const [maxRounds, setMaxRounds] = React.useState(8)
  const [lr, setLr] = React.useState("1e-6")
  const [beta, setBeta] = React.useState("0.02")
  const [temperature, setTemperature] = React.useState("1.0")
  const [automatic, setAutomatic] = React.useState(true)
  const [checkEvery, setCheckEvery] = React.useState(10)
  const submission = useJobSubmission("rl-composer")
  const isSubmitting = submission.pending

  React.useEffect(() => {
    let alive = true
    listGyms()
      .then((all) => alive && setGyms(all))
      .catch(
        (e) =>
          alive &&
          setLoadError(
            e instanceof Error ? e.message : "Failed to load the gyms."
          )
      )
    return () => {
      alive = false
    }
  }, [])

  // Only an explicit choice is stored; the default is derived from the list, so
  // neither select writes state from an effect the moment its data lands.
  const baseModel = chosenBase || models[0]?.id || ""
  // A choice only counts once the list confirms it exists: `?gym=` comes off a
  // URL anyone can type, and an id nothing answers to would otherwise leave the
  // select showing "No gyms" with no way to pick another.
  const known = (id: string) => gyms?.some((g) => g.id === id) ?? false
  const gymId = (known(chosenGym) ? chosenGym : gyms?.[0]?.id) || ""
  const gym = gyms?.find((g) => g.id === gymId)
  const noTools = gym !== undefined && !gym.composite && gym.tools.length === 0
  const adaptive = gymId === "math" && automatic

  // The easy rung of every family, for the reason the run form starts there: a
  // model that can list a directory and not read a file scores the same as one
  // that can do neither, and a batch of all-fail groups has an exactly zero
  // gradient. Recomputed when the gym changes rather than kept in state, so
  // switching gyms cannot leave another gym's ids checked.
  const defaults = React.useMemo(() => {
    if (!gym) return []
    const easy = gym.task_types.filter((t) => t.level === "easy")
    return (easy.length > 0 ? easy : gym.task_types).map((t) => t.id)
  }, [gym])
  const mine = new Set(gym?.task_types.map((t) => t.id) ?? [])
  const selected = (
    adaptive
      ? (gym?.task_types.map((t) => t.id) ?? [])
      : (taskTypes ?? defaults)
  ).filter((id) => mine.has(id))

  const sourceWeights = selectedSourceWeights(
    gym?.task_types ?? [],
    selected,
    gymWeights
  )

  const families = React.useMemo(() => {
    if (!gym) return []
    const rung = (level: string) => LEVELS.findIndex((l) => l === level)
    const order = [...new Set(gym.task_types.map((t) => t.family))]
    return order.map((family) => ({
      family,
      types: gym.task_types
        .filter((t) => t.family === family)
        .sort((a, b) => rung(a.level) - rung(b.level)),
    }))
  }, [gym])

  const episodes = tasksPerBatch * groupSize
  const oversized = episodes > MAX_EPISODES
  const numbers = [lr, beta, temperature].map(Number)
  const [lrValue, betaValue, temperatureValue] = numbers
  const numeric =
    numbers.every((n) => Number.isFinite(n)) &&
    lrValue > 0 &&
    betaValue >= 0 &&
    temperatureValue > 0
  const ready =
    name.trim() !== "" &&
    baseModel !== "" &&
    gymId !== "" &&
    selected.length > 0 &&
    !oversized &&
    numeric

  const toggle = (id: string) =>
    setTaskTypes((current) => {
      const base = current ?? defaults
      return base.includes(id) ? base.filter((t) => t !== id) : [...base, id]
    })

  /** Queues an immutable practice configuration under one session/request identity, preserving unknown outcomes for an unchanged retry. */
  const submit = async () => {
    const config = {
      model_name: name.trim(),
      base_model: baseModel,
      gym_id: gymId,
      task_types: [...selected],
      ...(gym?.composite ? { gym_weights: { ...sourceWeights } } : {}),
      group_size: groupSize,
      tasks_per_batch: tasksPerBatch,
      iterations,
      checkpoint_target: recoverySaves,
      max_rounds: noTools ? 1 : maxRounds,
      ...(adaptive ? { curriculum: { check_every: checkEvery } } : {}),
      lr: lrValue,
      beta: betaValue,
      temperature: temperatureValue,
    }
    const job = await submission.owner?.submit(
      identityDigest(config),
      (request) => queueReinforcement(config, [], request)
    )
    if (job) {
      toast.success("Practice run queued.", job.label)
      router.push(jobHref(job))
    }
  }

  if (loadError) return <LoadError hint={loadError} />

  return (
    <div className="space-y-6">
      {submission.state?.error && (
        <FormError>{submission.state.error}</FormError>
      )}
      <Panel variant="open" title="The model">
        <div className="grid gap-4 sm:grid-cols-2">
          <Field label="Base model" hint="the run's frozen reference too">
            <Select
              value={baseModel}
              onValueChange={(value) => setChosenBase(value ?? "")}
              disabled={isSubmitting || isLoading}
            >
              <SelectTrigger disabled={isLoading || models.length === 0}>
                {/* The value is an id, so the label is looked up explicitly —
                    the render function supersedes `placeholder`, which is why
                    the empty case is handled here too. */}
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

          <Field label="Name" hint="what the result is called">
            <Input
              value={name}
              placeholder={`${gymId}-practised`}
              onChange={(e) => setName(e.target.value)}
            />
          </Field>
        </div>
      </Panel>

      <Panel variant="open" title="The gym">
        {gyms === null ? (
          <Skeleton className="h-40" />
        ) : (
          <div className="space-y-5">
            <Field label="Gym">
              <Select
                value={gymId}
                onValueChange={(value) => {
                  setChosenGym(value ?? "")
                  // Reset selection to the new gym's easy rung.
                  setTaskTypes(null)
                }}
                disabled={isSubmitting}
              >
                <SelectTrigger className="w-full sm:w-80">
                  <SelectValue>
                    {(value: string | null) =>
                      gyms.find((g) => g.id === value)?.name ?? "No gyms"
                    }
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {gyms.map((option) => (
                    <SelectItem key={option.id} value={option.id}>
                      {option.name}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </Field>

            {gymId === "math" && (
              <div className="space-y-3">
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={automatic}
                    onCheckedChange={(checked) =>
                      setAutomatic(checked === true)
                    }
                  />
                  Advance difficulty automatically
                </label>
                <p className="text-xs text-muted-foreground">
                  Automatic practice includes every rung, starts with placement,
                  and keeps earlier skills in review. Promotion requires 90% on
                  two checks of 100 distinct validation problems. Fixed tests
                  run before and after training. These checks add generation
                  time beyond the training iterations. No tools are available.
                </p>
                {adaptive && (
                  <Field label="Validate every" hint="training iterations">
                    <Input
                      type="number"
                      min={1}
                      value={checkEvery}
                      onChange={(e) =>
                        setCheckEvery(clamp(e.target.valueAsNumber, 1, 1000))
                      }
                    />
                  </Field>
                )}
              </div>
            )}

            <Field label="Task types">
              {/* A column per family, rungs in order down it — the run form's
                  layout, and for its reason: read down for one skill getting
                  harder, across for the skills. */}
              <div className="grid gap-x-6 gap-y-5 sm:grid-cols-2 lg:grid-cols-3">
                {families.map(({ family, types }) => (
                  <div key={family} className="space-y-2.5">
                    {family && (
                      <p className="text-xs tracking-wide text-muted-foreground uppercase">
                        {family}
                      </p>
                    )}
                    {types.map((type) => (
                      <label
                        key={type.id}
                        className="flex items-start gap-2.5 text-sm"
                      >
                        <Checkbox
                          className="mt-0.5 shrink-0"
                          checked={selected.includes(type.id)}
                          disabled={adaptive}
                          onCheckedChange={() => toggle(type.id)}
                        />
                        <span className="min-w-0">
                          <span className="flex items-baseline gap-2">
                            {type.label}
                            <span className="text-[10px] tracking-wide text-muted-foreground uppercase">
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
          </div>
        )}
      </Panel>

      {gym?.composite && (
        <Panel variant="open" title="The mixture">
          <SourceWeights
            tasks={gym.task_types}
            selected={selected}
            weights={gymWeights}
            onChange={setGymWeights}
          />
        </Panel>
      )}
      {gym?.demonstrations && (
        <DemonstrationsForm
          gymId={gymId}
          taskTypes={selected}
          gymWeights={gym.composite ? sourceWeights : undefined}
        />
      )}

      <Panel variant="open" title="The loop">
        <div className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-4">
            <Field
              label="Recovery saves"
              hint="Approximate saves per run; 0 disables"
            >
              <Input
                type="number"
                min={0}
                step={1}
                value={recoverySaves}
                onChange={(event) =>
                  setRecoverySaves(clamp(event.target.valueAsNumber, 0, 10000))
                }
              />
            </Field>
            <Field label="Iterations" hint="gym runs">
              <Input
                type="number"
                min={1}
                value={iterations}
                onChange={(e) =>
                  setIterations(clamp(e.target.valueAsNumber, 1, 10_000))
                }
              />
            </Field>
            <Field label="Tasks" hint="per iteration">
              <Input
                type="number"
                min={1}
                max={MAX_EPISODES}
                value={tasksPerBatch}
                onChange={(e) =>
                  setTasksPerBatch(
                    clamp(e.target.valueAsNumber, 1, MAX_EPISODES)
                  )
                }
              />
            </Field>
            <Field label="Rollouts" hint="attempts per task">
              <Input
                type="number"
                min={2}
                max={MAX_EPISODES}
                value={groupSize}
                onChange={(e) =>
                  setGroupSize(clamp(e.target.valueAsNumber, 2, MAX_EPISODES))
                }
              />
            </Field>
            <Field
              label="Rounds"
              hint={
                noTools ? "one answer, no tools" : "shell turns per episode"
              }
            >
              <Input
                type="number"
                min={1}
                max={32}
                value={noTools ? 1 : maxRounds}
                disabled={noTools}
                onChange={(e) =>
                  setMaxRounds(clamp(e.target.valueAsNumber, 1, 32))
                }
              />
            </Field>
          </div>

          <p
            className={
              "rounded-(--radius-sheet) border px-3 py-2 text-xs " +
              (oversized
                ? "border-border bg-muted/40 text-muted-foreground"
                : "border-dashed border-border text-muted-foreground")
            }
          >
            {oversized ? (
              <>
                <span className="text-foreground">
                  {episodes} episodes is more than one gym run may hold.
                </span>{" "}
                The gym service limits the episodes in one run, so tasks ×
                rollouts has to stay at or under {MAX_EPISODES}. Each iteration
                would be refused, and the run would fail on its first one.
              </>
            ) : (
              <>
                <span className="text-foreground">
                  {iterations} iterations of {tasksPerBatch} task
                  {tasksPerBatch === 1 ? "" : "s"} × {groupSize} attempts ={" "}
                  {episodes} episodes each.
                </span>{" "}
                Each iteration rolls that batch out, compares every attempt with
                its own siblings, takes one optimizer step, and saves a
                checkpoint you can chat with or run the gym against.
              </>
            )}
          </p>
        </div>
      </Panel>

      <Panel variant="open" title="The update">
        <div className="space-y-5">
          <div className="grid gap-4 sm:grid-cols-3">
            <Field label="Learning rate">
              <Input value={lr} onChange={(e) => setLr(e.target.value)} />
            </Field>
            <Field label="KL weight" hint="β, against the base model">
              <Input value={beta} onChange={(e) => setBeta(e.target.value)} />
            </Field>
            <Field label="Temperature" hint="above zero, always">
              <Input
                value={temperature}
                onChange={(e) => setTemperature(e.target.value)}
              />
            </Field>
          </div>

          {!numeric && (
            <p className="border-l border-border py-2 pl-3 text-xs text-muted-foreground">
              A learning rate and a temperature above zero, and a KL weight of
              zero or more. A temperature of zero makes a group of {groupSize}{" "}
              attempts one attempt counted {groupSize} times, and a group whose
              attempts are identical has no gradient in it.
            </p>
          )}
        </div>
      </Panel>

      <Button
        onClick={submit}
        disabled={!ready || isSubmitting}
        className="gap-2"
      >
        <Play className="size-3.5" />
        {isSubmitting ? "Queueing…" : "Queue practice run"}
      </Button>
    </div>
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

function clamp(value: number, low: number, high: number): number {
  if (!Number.isFinite(value)) return low
  return Math.min(high, Math.max(low, Math.round(value)))
}
