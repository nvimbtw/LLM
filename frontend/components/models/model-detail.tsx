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
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { useAuth } from "@/hooks/use-auth"
import { useModel } from "@/hooks/use-model"
import { useModels } from "@/hooks/use-models"
import { LoadError } from "@/components/ui/error-state"
import { ModelBenchmarks } from "./model-benchmarks"
import { ModelSizeMeter } from "./model-glyph"
import { ModelEditSheet } from "./model-edit-sheet"
import {
  formatContext,
  formatCount,
  formatNumber,
  formatParams,
} from "./format"

function BackLink() {
  return (
    <Link
      href="/models"
      className="mb-8 inline-flex items-center gap-1.5 text-xs font-medium tracking-[0.14em] text-muted-foreground uppercase transition-colors hover:text-foreground"
    >
      <ArrowLeft className="size-3.5" />
      Models
    </Link>
  )
}

// A small dot separator for the editorial metadata line.
function Dot() {
  return <span className="text-muted-foreground/30">·</span>
}

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
    <div className="bg-background px-5 py-4">
      <dt className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="mt-1.5 font-display text-2xl font-semibold tracking-tight tabular-nums">
        {value}
      </dd>
      {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

function Spec({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-xs font-medium tracking-[0.12em] text-muted-foreground uppercase">
        {label}
      </dt>
      <dd className="mt-0.5 font-display text-sm font-semibold tabular-nums">
        {value}
      </dd>
    </div>
  )
}

function SpecPanel({
  title,
  specs,
}: {
  title: string
  specs: { label: string; value: string }[]
}) {
  if (specs.length === 0) return null
  return (
    <section className="rounded-lg border border-border bg-background">
      <header className="border-b border-border px-5 py-3.5">
        <h2 className="text-xs font-medium tracking-[0.16em] text-muted-foreground uppercase">
          {title}
        </h2>
      </header>
      <dl className="grid grid-cols-2 gap-x-6 gap-y-5 p-5 sm:grid-cols-3">
        {specs.map((s) => (
          <Spec key={s.label} label={s.label} value={s.value} />
        ))}
      </dl>
    </section>
  )
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

// Learning rates arrive as raw floats that stringify with representation noise
// (e.g. 0.00012999999999999999). Trim to 3 significant figures, and render small
// values in scientific notation so they stay short and never clip the cell.
function formatLr(lr: number): string {
  const clean = Number(lr.toPrecision(3))
  return clean < 0.001 ? clean.toExponential() : String(clean)
}

export function ModelDetail({ modelId }: { modelId: string }) {
  const router = useRouter()
  const { isAdmin } = useAuth()
  const { model, isLoading, notFound, error, update, setPublished, remove } =
    useModel(modelId)
  // Existing families across all models, offered as suggestions in the editor.
  const { models } = useModels()
  const families = React.useMemo(
    () =>
      Array.from(new Set(models.map((m) => m.family).filter(Boolean))).sort(),
    [models]
  )

  const [editing, setEditing] = React.useState(false)
  const [isDeleting, setIsDeleting] = React.useState(false)
  const [isPublishing, setIsPublishing] = React.useState(false)

  async function handleTogglePublished(next: boolean) {
    setIsPublishing(true)
    try {
      await setPublished(next)
      toast.success(next ? "Model published." : "Model unpublished.")
    } catch (err) {
      toast.error(
        "Couldn't update visibility",
        err instanceof Error ? err.message : "Something went wrong."
      )
    } finally {
      setIsPublishing(false)
    }
  }

  async function handleDelete() {
    setIsDeleting(true)
    try {
      await remove()
      // The page is about to unmount, so the confirmation has to be the toast —
      // anything rendered here goes with it.
      toast.success("Model deleted.")
      router.push("/models")
    } catch (err) {
      toast.error(
        "Couldn't delete the model",
        err instanceof Error ? err.message : "Something went wrong."
      )
      setIsDeleting(false)
    }
  }

  if (isLoading) {
    return (
      <>
        <BackLink />
        <div className="flex flex-col gap-8">
          <div className="space-y-3">
            <Skeleton className="h-9 w-64" />
            <Skeleton className="h-4 w-80" />
          </div>
          <Skeleton className="h-24 rounded-lg" />
          <div className="grid gap-4 sm:grid-cols-2">
            <Skeleton className="h-44 rounded-lg" />
            <Skeleton className="h-44 rounded-lg" />
          </div>
        </div>
      </>
    )
  }

  if (notFound) {
    return (
      <>
        <BackLink />
        <p className="text-sm text-muted-foreground">
          This model doesn&apos;t exist.
        </p>
      </>
    )
  }

  if (error || !model) {
    return (
      <>
        <BackLink />
        <LoadError title="This model didn't load." />
      </>
    )
  }

  const c = model.config
  const vocab = model.dataset?.vocab_size ?? c?.max_vocab
  // Parameters / Context / Vocabulary are surfaced in the hero strip; this panel
  // carries the finer architectural shape.
  const architecture: { label: string; value: string }[] = []
  if (c?.dimensions)
    architecture.push({ label: "Dimensions", value: `${c.dimensions}` })
  if (c?.layers) architecture.push({ label: "Layers", value: `${c.layers}` })
  if (c?.heads) architecture.push({ label: "Heads", value: `${c.heads}` })
  if (c?.ff_multiplier)
    architecture.push({ label: "FF multiplier", value: `${c.ff_multiplier}×` })

  const d = model.dataset
  const training: { label: string; value: string }[] = []
  if (d?.files)
    training.push({ label: "Training files", value: formatNumber(d.files) })
  if (d?.chars)
    training.push({ label: "Characters", value: formatCount(d.chars) })
  if (d?.corpus_len)
    training.push({ label: "Corpus tokens", value: formatCount(d.corpus_len) })
  if (d?.merges)
    training.push({ label: "BPE merges", value: formatNumber(d.merges) })
  if (c?.epochs) training.push({ label: "Epochs", value: `${c.epochs}` })
  if (c?.lr) training.push({ label: "Learning rate", value: formatLr(c.lr) })

  return (
    <>
      <BackLink />

      <div className="flex flex-col gap-8">
        <header className="border-b border-border pb-6">
          <div className="flex flex-wrap items-start justify-between gap-4">
            <div className="min-w-0">
              <div className="flex items-center gap-3">
                <h1 className="truncate font-display text-3xl font-semibold tracking-tight">
                  {model.name}
                </h1>
                <ModelSizeMeter tier={model.tier} className="mb-1 shrink-0" />
              </div>
              <div className="mt-2.5 flex flex-wrap items-center gap-x-2.5 gap-y-1 text-xs font-medium tracking-[0.14em] text-muted-foreground uppercase">
                <span>{model.family}</span>
                <Dot />
                <span>{model.tier}</span>
                <Dot />
                <span className={cn(model.published && "text-primary")}>
                  {model.published ? "Published" : "Private"}
                </span>
                <Dot />
                <span>Created {formatDate(model.createdAt)}</span>
              </div>
              <p className="mt-4 max-w-prose text-sm text-muted-foreground">
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
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => setEditing(true)}
                >
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
                        deletes its weights from disk. Existing messages are
                        kept but detached. This cannot be undone.
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
        </header>

        {/* Headline figures — the divided ledger shared with the usage page. */}
        <dl className="grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-border bg-border/60 sm:grid-cols-3">
          <HeroStat
            label="Parameters"
            value={
              model.estimatedParams ? formatParams(model.estimatedParams) : "—"
            }
          />
          <HeroStat
            label="Context"
            value={formatContext(model.contextWindow)}
            hint="tokens"
          />
          <HeroStat
            label="Vocabulary"
            value={vocab ? formatCount(vocab) : "—"}
          />
        </dl>

        <div className="grid gap-4 sm:grid-cols-2">
          <SpecPanel title="Architecture" specs={architecture} />
          <SpecPanel title="Training data" specs={training} />
        </div>

        {/* What it is, then how well it does it. Admin-only because the benchmark
            API is: a normal session would collect a 403 per model page. */}
        {isAdmin && <ModelBenchmarks modelId={modelId} />}
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
