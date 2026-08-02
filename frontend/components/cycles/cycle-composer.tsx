"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { Play } from "lucide-react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Panel } from "@/components/ui/panel"
import { Switch } from "@/components/ui/switch"
import { toast } from "@/components/ui/toast"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { useModels } from "@/hooks/use-models"
import { useSignals } from "@/hooks/use-signals"
import { startCycle } from "@/lib/cycles"

const DEFAULT_ROUNDS = 5

/**
 * Starting a cycle: pick the model it begins from, say how many rounds at most,
 * queue it.
 *
 * Deliberately few knobs. Everything else — candidates per prompt, learning rates,
 * beta, the gain a round must show to be kept — takes the defaults and is frozen
 * into the cycle's row, where its page displays them. A form that exposed all of
 * them would suggest they are the interesting decisions, and they are not: the
 * interesting decision is whether the loop climbs.
 */
export function CycleComposer() {
  const router = useRouter()
  const { models, isLoading } = useModels()
  const { status } = useSignals()
  const [chosenId, setChosenId] = React.useState("")
  const [rounds, setRounds] = React.useState(DEFAULT_ROUNDS)
  const [explore, setExplore] = React.useState(false)
  const [skipPreference, setSkipPreference] = React.useState(false)
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  const modelId = chosenId || models[0]?.id || ""

  const submit = async () => {
    setIsSubmitting(true)
    try {
      const cycle = await startCycle({
        id_model: modelId,
        max_rounds: rounds,
        config: { explore, skip_preference: skipPreference },
      })
      toast.success(
        "Cycle started.",
        explore
          ? `All ${rounds} rounds — it will not stop early.`
          : `Up to ${rounds} rounds — leave it running.`
      )
      router.push(`/cycles/${cycle.id_cycle}`)
    } catch (e) {
      toast.error(
        "Couldn't start the cycle",
        e instanceof Error ? e.message : "Something went wrong."
      )
      setIsSubmitting(false)
    }
  }

  return (
    <div className="space-y-6">
      <Panel title="Model" description="what round 1 trains from">
        <Select
          value={modelId}
          onValueChange={(value) => setChosenId(value ?? "")}
          disabled={isSubmitting || isLoading}
        >
          <SelectTrigger className="w-full min-w-0 sm:w-80">
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
      </Panel>

      <Panel title="Rounds" description="an upper bound, not a target">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="max-w-md text-xs text-muted-foreground">
            The loop usually stops before this: a round that does not beat the
            best score halts it, and so does a catalogue that has gone one-sided
            — which is what happens once the model clears the bar most of the
            time.
          </p>
          <Input
            type="number"
            min={1}
            max={50}
            value={rounds}
            onChange={(e) => setRounds(Number(e.target.value))}
            disabled={isSubmitting}
            className="h-8 w-24"
          />
        </div>

        {/* Sits under the round count because it is the same decision: how many
            rounds actually run. The gate is unchanged — this only stops it from
            being a stopping condition, which is why the label says what still
            happens rather than promising the loop will climb. */}
        <label className="mt-5 flex items-start justify-between gap-4 border-t border-border pt-4">
          <span className="space-y-1">
            <span className="block text-sm">Explore</span>
            <span className="block max-w-md text-xs text-muted-foreground">
              Run all {rounds} rounds even when one does not improve. Every round
              is still scored and its verdict recorded, and the best model is
              still the best one measured — a round that fails is kept as{" "}
              <span className="font-mono">explored</span> and the loop carries on
              from it. For seeing where the trajectory goes, not for making it go
              up.
            </span>
          </span>
          <Switch
            checked={explore}
            onCheckedChange={(next) => setExplore(Boolean(next))}
            disabled={isSubmitting}
            aria-label="Run every round instead of halting on the first that does not improve"
          />
        </label>

        {/* Sits beside Explore because both change what a round *does* rather than
            how it is judged. This one is a measurement: with it on, the round's
            output is the self-trained model, so the benchmark scores that stage
            alone instead of the two stages combined. */}
        <label className="mt-4 flex items-start justify-between gap-4 border-t border-border pt-4">
          <span className="space-y-1">
            <span className="block text-sm">Skip preference tuning</span>
            <span className="block max-w-md text-xs text-muted-foreground">
              Run each round as build catalogue → self-train → benchmark, with no
              preference stage. Scoring the two stages separately put self-train at
              &minus;3.8 and preference at &minus;5.7 points of assistant score, and
              preference is the only stage with no prose anchor. Turning it off
              measures whether self-train holds on its own.
            </span>
          </span>
          <Switch
            checked={skipPreference}
            onCheckedChange={(next) => setSkipPreference(Boolean(next))}
            disabled={isSubmitting}
            aria-label="Skip the preference tuning stage in every round"
          />
        </label>

        <div className="mt-5 flex items-center gap-3">
          <Button
            onClick={submit}
            disabled={isSubmitting || !modelId || status?.seeded === false}
            className="gap-2"
          >
            <Play className="size-3.5" />
            {isSubmitting ? "Starting…" : "Start cycle"}
          </Button>
        </div>
      </Panel>

      {/* Said before rather than after: the first stage of round 1 calibrates
          against hand ratings, so a cycle started without them halts immediately
          on a fact that was knowable here. */}
      {status?.seeded === false && (
        <p className="rounded-lg border border-dashed border-border px-4 py-3 text-sm text-muted-foreground">
          {status.seed_reason} Rate some replies first — a cycle&rsquo;s first
          stage checks the auto-scorer against them, and refuses if it
          disagrees.
        </p>
      )}
    </div>
  )
}
