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
import { PageHeader } from "@/components/ui/page-header"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
import { toast } from "@/components/ui/toast"
import { useAuth } from "@/hooks/use-auth"
import { useModel } from "@/hooks/use-model"
import { useModels } from "@/hooks/use-models"
import { LoadError } from "@/components/ui/error-state"
import { AttentionViewer } from "./attention-viewer"
import { ModelBenchmarks } from "./model-benchmarks"
import { ModelSizeMeter } from "./model-glyph"
import { ModelEditSheet } from "./model-edit-sheet"
import {
  formatContext,
  formatCount,
  formatNumber,
  formatParams,
} from "./format"

const BACK = { href: "/models", label: "Models" }

// The same back link `PageHeader` renders, for the three states that have no
// header to hang it on — loading, missing, failed. Spelled once here rather than
// letting a fourth style of "← Models" into the app.
function BackLink() {
  return (
    <Link
      href={BACK.href}
      className="mb-8 inline-flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground"
    >
      <ArrowLeft className="size-3" />
      {BACK.label}
    </Link>
  )
}

// A small dot separator for the metadata line.
function Dot() {
  return (
    <span aria-hidden className="text-muted-foreground/30">
      ·
    </span>
  )
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
    <div className="pr-6">
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd className="mt-1.5 figure text-2xl">{value}</dd>
      {hint && <p className="mt-0.5 text-xs text-muted-foreground">{hint}</p>}
    </div>
  )
}

function Spec({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd className="mt-0.5 figure text-sm">{value}</dd>
    </div>
  )
}

// A sheet, via `Panel` — it was a hand-rolled `border-border bg-background` box
// with an eyebrow for a heading, which is the flat outline the elevation pass
// replaced everywhere else and the reason this page read as a different product
// from the one that links to it.
function SpecPanel({
  title,
  specs,
}: {
  title: string
  specs: { label: string; value: string }[]
}) {
  if (specs.length === 0) return null
  return (
    <Panel title={title} bodyClassName="p-5">
      <dl className="grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3">
        {specs.map((s) => (
          <Spec key={s.label} label={s.label} value={s.value} />
        ))}
      </dl>
    </Panel>
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
          <Skeleton className="h-24 rounded-(--radius-sheet)" />
          <div className="grid gap-4 sm:grid-cols-2">
            <Skeleton className="h-44 rounded-(--radius-sheet)" />
            <Skeleton className="h-44 rounded-(--radius-sheet)" />
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
      {/* `PageHeader`, not a masthead of its own: this page had the only remaining
          hand-spelled one, and it had drifted — an uppercase eyebrow where every
          other page sets its provenance line in `text-sm`, and three outline
          buttons with no hierarchy between publishing a model and deleting it.
          No rule under it, because the ledger below opens with one of its own. */}
      <PageHeader
        className="mb-6 border-b-0 pb-0"
        back={BACK}
        title={
          <span className="flex items-center gap-3">
            <span className="truncate">{model.name}</span>
            <ModelSizeMeter tier={model.tier} className="mb-1 shrink-0" />
          </span>
        }
        description={
          <>
            <p className="flex flex-wrap items-center gap-x-2 gap-y-1">
              {/* The family is left exactly as it was typed — it is a name, and
                  `sft` is not `Sft`. The tier is one of ours, stored lowercase,
                  so it takes a capital like the two facts after it. Both used to
                  be set in an uppercase eyebrow, which hid the difference. */}
              <span>{model.family}</span>
              <Dot />
              <span className="capitalize">{model.tier}</span>
              <Dot />
              <span className={cn(model.published && "text-primary")}>
                {model.published ? "Published" : "Private"}
              </span>
              <Dot />
              <span>Created {formatDate(model.createdAt)}</span>
            </p>
            {model.description && <p>{model.description}</p>}
          </>
        }
        action={
          isAdmin && (
            <div className="flex items-center gap-2">
              {/* Publishing is what an admin comes here to do, so it is the one
                  filled button — and it drops to outline once done, because
                  unpublishing is a retraction rather than the next step. */}
              <Button
                variant={model.published ? "outline" : "default"}
                size="sm"
                disabled={isPublishing}
                onClick={() => handleTogglePublished(!model.published)}
                className="gap-1.5"
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
                className="gap-1.5"
              >
                <Pencil className="size-3.5" />
                Edit
              </Button>
              <AlertDialog>
                <AlertDialogTrigger
                  render={
                    // Ghost, and only red on hover — the same weight the job page
                    // gives its delete. A destructive action drawn as loud as the
                    // two beside it is a page that reads as three equal risks.
                    <Button
                      variant="ghost"
                      size="sm"
                      className="gap-1.5 text-muted-foreground hover:text-destructive"
                    >
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
          )
        }
      />

      <div className="flex flex-col gap-8">
        {/* Headline figures — the ruled ledger shared with the usage page. */}
        <dl className="grid grid-cols-2 gap-x-8 gap-y-5 border-y border-border py-4 sm:grid-cols-3">
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

        {/* Not admin-gated, unlike the benchmarks above: the route behind it is
            `AuthUser`, and anyone who can hold a conversation with this model
            can already ask it for a forward pass. The panel needs the layer and
            head counts to offer a choice, so a model whose meta.json predates
            them simply has no viewer rather than an empty one. */}
        {c?.layers && c?.heads && (
          <AttentionViewer
            modelId={modelId}
            layers={c.layers}
            heads={c.heads}
          />
        )}
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
