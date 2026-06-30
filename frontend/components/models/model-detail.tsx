"use client"

import * as React from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { ArrowLeft, Eye, EyeOff, Pencil, Trash2 } from "lucide-react"

import { cn } from "@/lib/utils"

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
import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import { Skeleton } from "@/components/ui/skeleton"
import { useAuth } from "@/hooks/use-auth"
import { useModel } from "@/hooks/use-model"
import { useModels } from "@/hooks/use-models"
import { ModelGlyph } from "./model-glyph"
import { ModelEditSheet } from "./model-edit-sheet"
import { formatContext, formatCount, formatNumber, formatParams } from "./format"

function BackLink() {
  return (
    <Link
      href="/models"
      className="mb-6 inline-flex items-center gap-1.5 text-sm text-muted-foreground transition-colors hover:text-foreground"
    >
      <ArrowLeft className="size-4" />
      Models
    </Link>
  )
}

function Spec({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex flex-col gap-0.5">
      <dt className="text-xs text-muted-foreground">{label}</dt>
      <dd className="font-heading text-sm font-medium tabular-nums">{value}</dd>
    </div>
  )
}

function SpecCard({
  title,
  specs,
}: {
  title: string
  specs: { label: string; value: string }[]
}) {
  if (specs.length === 0) return null
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">{title}</CardTitle>
      </CardHeader>
      <CardContent>
        <dl className="grid grid-cols-2 gap-x-4 gap-y-4 sm:grid-cols-3">
          {specs.map((s) => (
            <Spec key={s.label} label={s.label} value={s.value} />
          ))}
        </dl>
      </CardContent>
    </Card>
  )
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "long",
    day: "numeric",
  })
}

export function ModelDetail({ modelId }: { modelId: string }) {
  const router = useRouter()
  const { isAdmin } = useAuth()
  const { model, isLoading, notFound, error, update, setPublished, remove } =
    useModel(modelId)
  // Existing families across all models, offered as suggestions in the editor.
  const { models } = useModels()
  const families = React.useMemo(
    () => Array.from(new Set(models.map((m) => m.family).filter(Boolean))).sort(),
    [models]
  )

  const [editing, setEditing] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)
  const [deleteError, setDeleteError] = React.useState<string | null>(null)
  const [isPublishing, setIsPublishing] = React.useState(false)
  const [actionError, setActionError] = React.useState<string | null>(null)

  async function handleTogglePublished(next: boolean) {
    setIsPublishing(true)
    setActionError(null)
    try {
      await setPublished(next)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : "Unable to update visibility.")
    } finally {
      setIsPublishing(false)
    }
  }

  async function handleDelete() {
    setIsDeleting(true)
    setDeleteError(null)
    try {
      await remove()
      router.push("/models")
    } catch (err) {
      setDeleteError(err instanceof Error ? err.message : "Unable to delete the model.")
      setIsDeleting(false)
    }
  }

  if (isLoading) {
    return (
      <>
        <BackLink />
        <div className="flex flex-col gap-4">
          <Skeleton className="h-56 rounded-xl" />
          <div className="grid gap-4 sm:grid-cols-2">
            <Skeleton className="h-40 rounded-xl" />
            <Skeleton className="h-40 rounded-xl" />
          </div>
        </div>
      </>
    )
  }

  if (notFound) {
    return (
      <>
        <BackLink />
        <p className="text-sm text-muted-foreground">This model doesn&apos;t exist.</p>
      </>
    )
  }

  if (error || !model) {
    return (
      <>
        <BackLink />
        <p className="text-sm text-destructive">
          Failed to load this model. Please try again.
        </p>
      </>
    )
  }

  const c = model.config
  const architecture: { label: string; value: string }[] = []
  if (model.estimatedParams) architecture.push({ label: "Parameters", value: formatParams(model.estimatedParams) })
  architecture.push({ label: "Context", value: `${formatContext(model.contextWindow)} tokens` })
  if (c?.dimensions) architecture.push({ label: "Dimensions", value: `${c.dimensions}` })
  if (c?.layers) architecture.push({ label: "Layers", value: `${c.layers}` })
  if (c?.heads) architecture.push({ label: "Heads", value: `${c.heads}` })
  if (c?.ff_multiplier) architecture.push({ label: "FF multiplier", value: `${c.ff_multiplier}×` })
  const vocab = model.dataset?.vocab_size ?? c?.max_vocab
  if (vocab) architecture.push({ label: "Vocabulary", value: formatCount(vocab) })

  const d = model.dataset
  const training: { label: string; value: string }[] = []
  if (d?.files) training.push({ label: "Training files", value: formatNumber(d.files) })
  if (d?.chars) training.push({ label: "Characters", value: formatCount(d.chars) })
  if (d?.corpus_len) training.push({ label: "Corpus tokens", value: formatCount(d.corpus_len) })
  if (d?.merges) training.push({ label: "BPE merges", value: formatNumber(d.merges) })
  if (c?.epochs) training.push({ label: "Epochs", value: `${c.epochs}` })
  if (c?.lr) training.push({ label: "Learning rate", value: `${c.lr}` })

  return (
    <>
      <BackLink />

      {(deleteError || actionError) && (
        <p className="mb-4 text-sm text-destructive">{deleteError ?? actionError}</p>
      )}

      <div className="flex flex-col gap-4">
        <Card className="gap-0 overflow-hidden pt-0">
          <ModelGlyph tier={model.tier} className="h-32" />
          <CardHeader className="pt-5">
            <div className="flex flex-wrap items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <h1 className="font-heading text-xl font-semibold tracking-tight">
                    {model.name}
                  </h1>
                  <span className="rounded-full bg-muted px-2 py-0.5 text-xs font-medium text-muted-foreground capitalize">
                    {model.tier}
                  </span>
                  <span
                    className={cn(
                      "rounded-full px-2 py-0.5 text-xs font-medium",
                      model.published
                        ? "bg-chart-2/15 text-chart-2"
                        : "bg-muted text-muted-foreground"
                    )}
                  >
                    {model.published ? "Published" : "Private"}
                  </span>
                </div>
                <p className="mt-1 text-sm text-muted-foreground">
                  {model.description}
                </p>
              </div>

              {isAdmin && (
                <div className="flex shrink-0 items-center gap-2">
                  <Button
                    variant={model.published ? "outline" : "default"}
                    size="sm"
                    disabled={isPublishing}
                    onClick={() => handleTogglePublished(!model.published)}
                  >
                    {model.published ? (
                      <EyeOff className="size-3.5" />
                    ) : (
                      <Eye className="size-3.5" />
                    )}
                    {model.published ? "Unpublish" : "Publish"}
                  </Button>
                  <Button variant="outline" size="sm" onClick={() => setEditing(true)}>
                    <Pencil className="size-3.5" />
                    Edit
                  </Button>
                  <AlertDialog>
                    <AlertDialogTrigger
                      render={
                        <Button variant="outline" size="sm">
                          <Trash2 className="size-3.5" />
                          Delete
                        </Button>
                      }
                    />
                    <AlertDialogContent>
                      <AlertDialogHeader>
                        <AlertDialogTitle>Delete {model.name}?</AlertDialogTitle>
                        <AlertDialogDescription>
                          This permanently removes the model from the registry and
                          deletes its weights from disk. Existing messages are kept
                          but detached. This cannot be undone.
                        </AlertDialogDescription>
                      </AlertDialogHeader>
                      <AlertDialogFooter>
                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                        <AlertDialogAction
                          onClick={handleDelete}
                          disabled={isDeleting}
                          className="bg-destructive text-white hover:bg-destructive/90"
                        >
                          {isDeleting ? "Deleting…" : "Delete model"}
                        </AlertDialogAction>
                      </AlertDialogFooter>
                    </AlertDialogContent>
                  </AlertDialog>
                </div>
              )}
            </div>

            <Separator className="my-4" />
            <dl className="grid grid-cols-2 gap-4 sm:grid-cols-3">
              <Spec label="Family" value={model.family} />
              <Spec label="Created" value={formatDate(model.createdAt)} />
            </dl>
          </CardHeader>
        </Card>

        <div className="grid gap-4 sm:grid-cols-2">
          <SpecCard title="Architecture" specs={architecture} />
          <SpecCard title="Training data" specs={training} />
        </div>
      </div>

      {isAdmin && (
        <ModelEditSheet
          open={editing}
          onOpenChange={setEditing}
          initialName={model.name}
          initialDescription={model.description}
          initialFamily={model.family}
          families={families}
          onSave={update}
        />
      )}
    </>
  )
}
