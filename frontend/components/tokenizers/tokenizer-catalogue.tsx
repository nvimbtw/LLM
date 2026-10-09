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

function compact(n: number | null): string {
  if (n == null) return "—"
  return Intl.NumberFormat("en-US", { notation: "compact" }).format(n)
}

function megabytes(bytes: number): string {
  if (!bytes) return "—"
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

export interface TokenizerCatalogueProps {
  tokenizers: TokenizerInfo[]
  onDelete: (tokenizer: TokenizerInfo) => void
}

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
    <table className="w-full table-fixed">
      <caption className="sr-only">Vocabulary catalogue</caption>
      <thead className="border-y border-border/60 text-xs text-muted-foreground">
        <tr>
          <th scope="col" className="py-3 pr-3 text-left font-normal">
            Vocabulary
          </th>
          <th
            scope="col"
            className="hidden py-3 pr-3 text-left font-normal lg:table-cell"
          >
            Datasets
          </th>
          <th
            scope="col"
            className="w-16 py-3 pr-3 text-right font-normal md:w-20"
          >
            Vocab
          </th>
          <th
            scope="col"
            className="hidden w-20 py-3 pr-3 text-right font-normal md:table-cell"
          >
            Merges
          </th>
          <th
            scope="col"
            className="hidden w-16 py-3 pr-3 text-right font-normal md:table-cell"
          >
            Files
          </th>
          <th
            scope="col"
            className="hidden w-20 py-3 pr-3 text-right font-normal md:table-cell"
          >
            Size
          </th>
          <th scope="col" className="w-8 py-3">
            <span className="sr-only">Actions</span>
          </th>
        </tr>
      </thead>
      <tbody>
        {tokenizers.map((tokenizer) => (
          <Row
            key={tokenizer.id}
            tokenizer={tokenizer}
            onDelete={() => onDelete(tokenizer)}
          />
        ))}
      </tbody>
    </table>
  )
}

function Row({
  tokenizer,
  onDelete,
}: {
  tokenizer: TokenizerInfo
  onDelete: () => void
}) {
  return (
    <tr className="border-b border-border/40">
      <th scope="row" className="min-w-0 py-3 pr-3 text-left font-normal">
        <p className="truncate text-sm font-medium">{tokenizer.name}</p>
        <p className="mt-1 flex flex-wrap gap-x-2 gap-y-1 text-xs text-muted-foreground">
          <span>{tokenizer.implicit ? "Learned by a run" : "Named build"}</span>
          {!tokenizer.hasVocab && (
            <span className="text-destructive">Missing on disk</span>
          )}
        </p>
        <p className="mt-1 truncate text-xs text-muted-foreground lg:hidden">
          {describeDatasets(tokenizer)}
        </p>
        <p className="mt-1 text-xs text-muted-foreground tabular-nums md:hidden">
          {compact(tokenizer.merges)} merges · {compact(tokenizer.files)} files
          · {megabytes(tokenizer.bytes)}
        </p>
      </th>
      <td className="hidden min-w-0 py-3 pr-3 text-xs text-muted-foreground lg:table-cell">
        <p className="truncate">{describeDatasets(tokenizer)}</p>
      </td>
      <td className="py-3 pr-3 text-right text-sm tabular-nums">
        {compact(tokenizer.vocabSize)}
      </td>
      <td className="hidden py-3 pr-3 text-right text-xs text-muted-foreground tabular-nums md:table-cell">
        {compact(tokenizer.merges)}
      </td>
      <td className="hidden py-3 pr-3 text-right text-xs text-muted-foreground tabular-nums md:table-cell">
        {compact(tokenizer.files)}
      </td>
      <td className="hidden py-3 pr-3 text-right text-xs text-muted-foreground tabular-nums md:table-cell">
        {megabytes(tokenizer.bytes)}
      </td>
      <td className="py-3 text-right">
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
      </td>
    </tr>
  )
}
