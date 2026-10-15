"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { ArrowLeft, Eye, EyeOff, Trash2 } from "lucide-react"

import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import { ArchitectureFlow } from "@/components/training/architecture-flow"
import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useModel, type ModelDetail as ModelData } from "@/hooks/use-model"
import { useModels } from "@/hooks/use-models"
import { AttentionViewer } from "./attention-viewer"
import { ModelBenchmarks } from "./model-benchmarks"
import { ModelCheckpoints } from "./model-checkpoints"
import { ModelHeader } from "./model-header"
import { ModelLineage } from "./model-lineage"
import {
  formatContext,
  formatCount,
  formatNumber,
  formatParams,
} from "./format"

export interface ModelDetailProps {
  modelId: string
}

/** Scopes model reads, editor drafts and diagram controls to the actual session and model identity. */
export function ModelDetail({ modelId }: ModelDetailProps) {
  const token = useSessionToken()
  return token ? (
    <ModelDetailContent key={`${token}:${modelId}`} modelId={modelId} />
  ) : null
}

/** Loads model facts and coordinates admin publication/deletion while preserving read-only access for regular users. */
function ModelDetailContent({ modelId }: ModelDetailProps) {
  const router = useRouter()
  const { isAdmin } = useAuth()
  const { model, isLoading, notFound, error, update, setPublished, remove } =
    useModel(modelId)
  const { models } = useModels()
  const families = Array.from(
    new Set(models.map((entry) => entry.family).filter(Boolean))
  ).sort()
  const [busy, setBusy] = React.useState(false)
  const active = React.useRef(true)
  React.useEffect(() => {
    active.current = true
    return () => {
      active.current = false
    }
  }, [])

  /** Persists publication changes and reports the result without replacing unsaved display fields. */
  async function togglePublished() {
    if (!model || busy) return
    setBusy(true)
    try {
      await setPublished(!model.published)
      if (!active.current) return
      toast.success(model.published ? "Model unpublished." : "Model published.")
    } catch (failure) {
      if (!active.current) return
      toast.error(
        "Couldn't update visibility",
        failure instanceof Error ? failure.message : "Something went wrong."
      )
    } finally {
      if (active.current) setBusy(false)
    }
  }

  /** Removes the selected model and returns to the registry only after a successful response. */
  async function deleteModel() {
    if (busy) return
    setBusy(true)
    try {
      await remove()
      if (!active.current) return
      toast.success("Model deleted.")
      router.push("/models")
    } catch (failure) {
      if (!active.current) return
      toast.error(
        "Couldn't delete the model",
        failure instanceof Error ? failure.message : "Something went wrong."
      )
      setBusy(false)
    }
  }

  if (isLoading || notFound || error || !model)
    return (
      <>
        <Link
          href="/models"
          className="mb-6 inline-flex items-center gap-1.5 text-xs text-muted-foreground hover:text-foreground"
        >
          <ArrowLeft className="size-3" />
          Models
        </Link>
        {isLoading ? (
          <div className="space-y-6">
            <Skeleton className="h-12 w-64" />
            <Skeleton className="h-64" />
          </div>
        ) : notFound ? (
          <p className="text-sm text-muted-foreground">
            This model doesn&apos;t exist.
          </p>
        ) : (
          <LoadError title="This model didn't load." />
        )}
      </>
    )

  return (
    <>
      <ModelHeader
        model={model}
        editable={isAdmin}
        busy={busy}
        families={families}
        onSave={update}
        actions={(saving) => (
          <>
            <Button
              type="button"
              variant="text"
              size="sm"
              disabled={busy || saving}
              onClick={togglePublished}
            >
              {model.published ? <EyeOff /> : <Eye />}
              {model.published ? "Unpublish" : "Publish"}
            </Button>
            <AlertDialog>
              <AlertDialogTrigger
                render={
                  <Button
                    type="button"
                    variant="text"
                    size="sm"
                    disabled={busy || saving}
                    className="text-muted-foreground hover:text-destructive"
                  >
                    <Trash2 />
                    Delete
                  </Button>
                }
              />
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete {model.name}?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This permanently removes the model and its weights. Existing
                    messages are kept but detached. This cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel disabled={busy}>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    disabled={busy}
                    onClick={deleteModel}
                    className="bg-destructive text-white hover:bg-destructive/90"
                  >
                    {busy ? "Deleting…" : "Delete model"}
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </>
        )}
      />
      <div className="flex flex-col gap-8">
        <dl className="grid grid-cols-2 gap-x-8 gap-y-5 border-y border-border py-5 sm:grid-cols-3">
          <HeroStat
            label="Parameters"
            value={model.params ? formatParams(model.params) : "—"}
          />
          <HeroStat
            label="Context"
            value={
              model.contextWindow ? formatContext(model.contextWindow) : "—"
            }
            hint="tokens"
          />
          <HeroStat
            label="Vocabulary"
            value={
              model.vocabularySize ? formatCount(model.vocabularySize) : "—"
            }
            hint="tokens"
          />
        </dl>
        <ModelArchitecture model={model} />
        <TrainingFacts model={model} />
        <ModelLineage model={model} />
        <ModelCheckpoints modelId={modelId} canContinue={isAdmin} />
        {isAdmin && <ModelBenchmarks modelId={modelId} />}
        {model.architecture && (
          <AttentionViewer
            modelId={modelId}
            layers={model.architecture.layers}
            heads={model.architecture.heads}
          />
        )}
      </div>
    </>
  )
}

/** Displays one measured model statistic with an optional unit. */
function HeroStat({
  label,
  value,
  hint,
}: {
  label: string
  value: string
  hint?: string
}) {
  return (
    <div>
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="mt-1.5 figure text-2xl">{value}</dd>
      {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

/** Presents resolved architecture with the same diagram implementation used by the training composer. */
function ModelArchitecture({ model }: { model: ModelData }) {
  const arch = model.architecture
  if (!arch)
    return (
      <Panel variant="open" title="Architecture">
        <p className="text-sm text-muted-foreground">
          Architecture metadata is unavailable.
        </p>
      </Panel>
    )
  const facts = [
    { label: "Dimensions", value: formatNumber(arch.dimensions) },
    { label: "Layers", value: formatNumber(arch.layers) },
    { label: "Heads", value: formatNumber(arch.heads) },
    { label: "FF multiplier", value: `${arch.ff_multiplier}×` },
  ]
  return (
    <Panel variant="open" title="Architecture">
      <FactList facts={facts} />
      <ArchitectureFlow
        view="generation"
        cfg={{ ...arch, mode: model.trainingMode }}
      />
    </Panel>
  )
}

/** Shows only this run's training data and optimization figures, including nested legacy dataset summaries. */
function TrainingFacts({ model }: { model: ModelData }) {
  const result = model.dataset
  const data = result?.dataset
  const figures = [
    { label: "Documents", count: data?.documents ?? result?.files },
    { label: "Conversations", count: data?.conversations },
    { label: "Characters", count: data?.chars ?? result?.chars },
    { label: "Corpus tokens", count: data?.corpus_len ?? result?.corpus_len },
    { label: "BPE merges", count: result?.merges },
    { label: "Epochs", count: model.config?.epochs },
  ]
  const facts = figures.flatMap(({ label, count }) =>
    count === undefined ? [] : [{ label, value: formatCount(count) }]
  )
  if (model.config?.lr)
    facts.push({
      label: "Learning rate",
      value: String(Number(model.config.lr.toPrecision(3))),
    })
  if (!facts.length) return null
  return (
    <Panel variant="open" title="Training">
      <FactList facts={facts} />
    </Panel>
  )
}

/** Renders a compact ledger of available model facts. */
function FactList({ facts }: { facts: { label: string; value: string }[] }) {
  return (
    <dl className="grid gap-x-8 sm:grid-cols-2">
      {facts.map(({ label, value }) => (
        <div
          key={label}
          className="flex items-baseline justify-between gap-3 border-b border-border/40 py-3"
        >
          <dt className="text-sm text-muted-foreground">{label}</dt>
          <dd className="text-sm tabular-nums">{value}</dd>
        </div>
      ))}
    </dl>
  )
}
