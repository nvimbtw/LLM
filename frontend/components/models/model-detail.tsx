"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { ArrowLeft, Eye, EyeOff, Trash2 } from "lucide-react"

import { DeleteConfirmation } from "@/components/ui/delete-confirmation"
import { ModelAnatomy, TrainingRecord } from "./model-anatomy"
import { Button } from "@/components/ui/button"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useModel } from "@/hooks/use-model"
import { useModels } from "@/hooks/use-models"
import { AttentionViewer } from "./attention-viewer"
import { ModelBenchmarks } from "./model-benchmarks"
import { ModelCheckpoints } from "./model-checkpoints"
import { ModelHeader } from "./model-header"
import { ModelLineage } from "./model-lineage"
import { SavedModelDefinition } from "./saved-definition"

export interface ModelDetailProps {
  modelId: string
}

/** Scopes model reads, editor drafts and diagram controls to the actual session and model identity. */
export function ModelDetail({ modelId }: ModelDetailProps) {
  const token = useSessionToken()
  return (
    <ModelDetailContent
      key={`${token ?? "anonymous"}:${modelId}`}
      modelId={modelId}
    />
  )
}

/** Loads model facts and coordinates admin publication/deletion while preserving read-only access for regular users. */
function ModelDetailContent({ modelId }: ModelDetailProps) {
  const router = useRouter()
  const { isAdmin, isAuthenticated } = useAuth()
  const { model, isLoading, notFound, error, update, setPublished, remove } =
    useModel(modelId)
  const { models } = useModels()
  const families = Array.from(
    new Set(models.map((entry) => entry.family).filter(Boolean))
  ).sort()
  const [busy, setBusy] = React.useState(false)
  const [confirmOpen, setConfirmOpen] = React.useState(false)
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
            <Button
              type="button"
              variant="text"
              size="sm"
              disabled={busy || saving}
              onClick={() => setConfirmOpen(true)}
            >
              <Trash2 />
              Delete
            </Button>
          </>
        )}
      />
      {confirmOpen && (
        <DeleteConfirmation
          open
          onOpenChange={setConfirmOpen}
          title={`Delete ${model.name}?`}
          description="This permanently removes the model and its weights. Existing messages are kept but detached. This cannot be undone."
          label="Delete model"
          onDelete={remove}
          onDeleted={() => {
            toast.success("Model deleted.")
            router.push("/models")
          }}
        />
      )}
      <div className="flex flex-col gap-8">
        <ModelAnatomy model={model} />
        {model.definitionId && isAdmin && (
          <SavedModelDefinition
            modelId={model.id}
            definitionId={model.definitionId}
          />
        )}
        <TrainingRecord model={model} />
        <ModelLineage model={model} />
        <ModelCheckpoints modelId={modelId} canContinue={isAdmin} />
        {isAdmin && <ModelBenchmarks modelId={modelId} />}
        {model.hasWeights && isAuthenticated && (
          <AttentionViewer modelId={modelId} />
        )}
        {!isAuthenticated && (
          <Link href="/login" className="text-sm text-primary hover:underline">
            Log in to chat or inspect attention
          </Link>
        )}
      </div>
    </>
  )
}
