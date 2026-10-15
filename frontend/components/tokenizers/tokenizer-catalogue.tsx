"use client"

import { Trash2 } from "lucide-react"
import { Button } from "@/components/ui/button"
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
import { describeDatasets, type TokenizerInfo } from "@/lib/tokenizers"

/** Formats a measured count compactly, keeping unavailable values explicit. */
function compact(n: number | null): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

/** Formats vocabulary storage bytes in megabytes, preserving the empty marker. */
function megabytes(bytes: number): string {
  if (!bytes) return "—"
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export interface TokenizerCatalogueProps {
  tokenizers: TokenizerInfo[]
  onDelete: (tokenizer: TokenizerInfo) => void
}

/** Lists read-only vocabularies with their provenance, figures and confirmed deletion. */
export function TokenizerCatalogue({
  tokenizers,
  onDelete,
}: TokenizerCatalogueProps) {
  if (tokenizers.length === 0)
    return (
      <p className="border-y border-border/60 py-6 text-sm text-muted-foreground">
        No vocabularies in this category.
      </p>
    )
  return (
    <ul
      aria-label="Vocabulary catalogue"
      className="divide-y divide-border/60 border-y border-border/60"
    >
      {tokenizers.map((tokenizer) => (
        <Row
          key={tokenizer.id}
          tokenizer={tokenizer}
          onDelete={() => onDelete(tokenizer)}
        />
      ))}
    </ul>
  )
}

/** Presents one vocabulary and invokes deletion only after the existing confirmation. */
function Row({
  tokenizer,
  onDelete,
}: {
  tokenizer: TokenizerInfo
  onDelete: () => void
}) {
  return (
    <li className="flex items-start justify-between gap-4 py-5">
      <div className="min-w-0">
        <h2 className="font-display text-xl leading-snug break-words">
          {tokenizer.name}
        </h2>
        <p className="mt-2 flex flex-wrap gap-x-2 gap-y-1 text-xs text-muted-foreground">
          <span>{tokenizer.implicit ? "Learned by a run" : "Named build"}</span>
          {!tokenizer.hasVocab && (
            <span className="text-destructive">Missing on disk</span>
          )}
        </p>
        <p className="mt-1 text-xs break-words text-muted-foreground">
          {describeDatasets(tokenizer)}
        </p>
        <p className="mt-3 text-sm text-muted-foreground tabular-nums">
          {compact(tokenizer.vocabSize)} vocab · {compact(tokenizer.merges)}{" "}
          merges
          {" · "}
          {compact(tokenizer.files)} files · {megabytes(tokenizer.bytes)}
        </p>
      </div>
      <div className="shrink-0">
        <AlertDialog>
          <AlertDialogTrigger
            render={
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Delete ${tokenizer.name}`}
              >
                <Trash2 className="size-4" />
              </Button>
            }
          />
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Delete {tokenizer.name}?</AlertDialogTitle>
              <AlertDialogDescription>
                The corpora packed under it go too — without this vocabulary
                they are token ids nothing can read. Models trained with it keep
                their own copy and are unaffected.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                onClick={onDelete}
                className="bg-destructive text-white hover:bg-destructive/90"
              >
                Delete
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </li>
  )
}
