"use client"

import * as React from "react"
import { useRouter } from "next/navigation"
import { Sparkles } from "lucide-react"

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
import { queueSynth } from "@/lib/jobs"

// Any OpenAI-compatible `/chat/completions` endpoint works, so the presets are
// only shortcuts past typing a URL. A local Ollama is on the list because it is
// free, unmetered and needs no key at all — the right first thing to try.
//
// Google is listed first because Gemma is Google's model and AI Studio hands out a
// free key for it directly — going through a reseller to reach it is a detour. Note
// its OpenAI-compatible endpoint lives under a `/openai` suffix; the plain
// `/v1beta` root speaks Google's own protocol and will not work here.
const PROVIDERS = [
  {
    label: "Google AI Studio",
    base_url: "https://generativelanguage.googleapis.com/v1beta/openai",
    // Gemma 4. The 3-series names are gone from the API, so `gemma-3-27b-it` now
    // 404s — if this stops resolving again, `GET /v1beta/models` lists what the
    // key can actually reach.
    model: "gemma-4-31b-it",
  },
  {
    label: "OpenRouter",
    base_url: "https://openrouter.ai/api/v1",
    model: "google/gemma-3-27b-it:free",
  },
  {
    label: "DeepSeek",
    base_url: "https://api.deepseek.com/v1",
    model: "deepseek-chat",
  },
  {
    label: "Local Ollama",
    base_url: "http://localhost:11434/v1",
    model: "gemma3:12b",
  },
]

const DEFAULT_COUNT = 50
const DEFAULT_BATCH = 2
const DEFAULT_MIN_THINK = 120
const DEFAULT_TEMPERATURE = 1.0

/**
 * Composing a generation run: where to ask, what to ask for, and which of our
 * models the result has to fit.
 *
 * That last field is the one that matters and the one it would be easy to leave
 * out. Without it nothing measures length, and the conversations come back three
 * times longer than the window that will train on them — which does not fail, it
 * just quietly trains the model on the middles of replies.
 */
export function SynthComposer() {
  const router = useRouter()
  const { models, isLoading } = useModels()

  const [provider, setProvider] = React.useState(PROVIDERS[0].label)
  const [baseUrl, setBaseUrl] = React.useState(PROVIDERS[0].base_url)
  const [model, setModel] = React.useState(PROVIDERS[0].model)
  const [apiKeyEnv, setApiKeyEnv] = React.useState("SFT_API_KEY")
  const [chosenId, setChosenId] = React.useState("")
  const [count, setCount] = React.useState(DEFAULT_COUNT)
  const [batch, setBatch] = React.useState(DEFAULT_BATCH)
  const [minThink, setMinThink] = React.useState(DEFAULT_MIN_THINK)
  const [temperature, setTemperature] = React.useState(DEFAULT_TEMPERATURE)
  const [isSubmitting, setIsSubmitting] = React.useState(false)

  const targetId = chosenId || models[0]?.id || ""

  const pickProvider = (label: string) => {
    const found = PROVIDERS.find((p) => p.label === label)
    if (!found) return
    setProvider(label)
    setBaseUrl(found.base_url)
    setModel(found.model)
  }

  const submit = async () => {
    setIsSubmitting(true)
    try {
      const job = await queueSynth({
        base_url: baseUrl,
        model,
        count,
        batch,
        temperature,
        min_think_words: minThink,
        api_key_env: apiKeyEnv,
        id_model: targetId || undefined,
      })
      // The network lane, so this one really does run beside whatever is on the
      // GPU — worth saying, because the queue page will not show it waiting.
      toast.success("Generation queued on the network lane.", job.label)
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
    baseUrl.startsWith("http") &&
    model.trim() !== "" &&
    count >= 1 &&
    batch >= 1 &&
    temperature > 0

  return (
    <div className="space-y-6">
      <Panel
        title="Provider"
        description="Whose model writes the conversations"
      >
        <div className="flex flex-wrap items-center gap-3">
          <Select
            value={provider}
            onValueChange={(v) => pickProvider(v ?? "")}
            disabled={isSubmitting}
          >
            <SelectTrigger className="w-full min-w-0 sm:w-56">
              <SelectValue>
                {(value: string | null) => value ?? "Provider"}
              </SelectValue>
            </SelectTrigger>
            <SelectContent>
              {PROVIDERS.map((p) => (
                <SelectItem key={p.label} value={p.label}>
                  {p.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        <dl className="mt-4 space-y-3">
          <Field
            label="API base"
            hint="Any OpenAI-compatible /chat/completions endpoint."
          >
            <Input
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
              disabled={isSubmitting}
              className="h-8 w-full sm:w-80"
            />
          </Field>
          <Field label="Model id" hint="The provider's name for it, not ours.">
            <Input
              value={model}
              onChange={(e) => setModel(e.target.value)}
              disabled={isSubmitting}
              className="h-8 w-full sm:w-80"
            />
          </Field>
          <Field
            label="API key variable"
            hint="The name of a server-side environment variable. The key itself never reaches this form, the job config, or the database — the server reads it at call time. It has to be a key for the provider named above: a key from somewhere else is rejected as a 401, often with a message claiming no key was sent at all. Leave it unset for a local model."
          >
            <Input
              value={apiKeyEnv}
              onChange={(e) => setApiKeyEnv(e.target.value)}
              disabled={isSubmitting}
              className="h-8 w-40 font-mono text-xs"
            />
          </Field>
        </dl>
      </Panel>

      <Panel
        title="Target"
        description="Which model has to be able to train on this"
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
          Every generated conversation is rendered with this model&apos;s
          tokenizer and rejected if it does not fit its context window. Skip it
          and nothing checks length — which is how a corpus ends up several
          times too long to train on, without anything failing.
        </p>
      </Panel>

      <Panel
        title="Run"
        description="How much, and how hard the thinking bar is set"
      >
        <dl className="space-y-3">
          <Field
            label="Conversations"
            hint="Valid ones to add. The run stops here, or when the call budget runs out."
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
          <Field
            label="Per call"
            hint="Small on purpose: a batch with long thinking blocks runs past the response cap and arrives truncated, losing everything after the cut."
          >
            <Input
              type="number"
              min={1}
              max={10}
              value={batch}
              onChange={(e) => setBatch(Number(e.target.value))}
              disabled={isSubmitting}
              className="h-8 w-24"
            />
          </Field>
          <Field
            label="Minimum thinking words"
            hint="Blocks shorter than this are rejected. Large reasoning blocks are what generation adds over the hand-written set, which has almost none."
          >
            <Input
              type="number"
              min={0}
              value={minThink}
              onChange={(e) => setMinThink(Number(e.target.value))}
              disabled={isSubmitting}
              className="h-8 w-24"
            />
          </Field>
          <Field
            label="Temperature"
            hint="High on purpose: this is a diversity task, and identical conversations are deduplicated away anyway."
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

        <div className="mt-5 flex items-center gap-3">
          <Button
            onClick={submit}
            disabled={isSubmitting || !valid}
            className="gap-2"
          >
            <Sparkles className="size-3.5" />
            {isSubmitting ? "Queueing…" : "Queue generation"}
          </Button>
          <span className="text-xs text-muted-foreground">
            Runs on the network lane — it never occupies the GPU.
          </span>
        </div>
      </Panel>

      <p className="rounded-lg border border-dashed border-border px-4 py-3 text-xs text-muted-foreground">
        Generated conversations are appended to{" "}
        <code className="text-xs">synthetic_llm.jsonl</code>, beside the
        hand-curated <code className="text-xs">synthetic.jsonl</code> rather
        than into it — the gold set is what every call imitates, and it is never
        written to. Both files train.
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
