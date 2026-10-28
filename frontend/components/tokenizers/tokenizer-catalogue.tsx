"use client"

import { EmptyState } from "@/components/ui/empty-state"

import {
  ResourceList,
  ResourceRow,
  ResourceTitle,
  ResourceLink,
} from "@/components/ui/resource-list"
import { TokenizerDeleteButton } from "./tokenizer-delete-button"
import { describeDatasets, type TokenizerInfo } from "@/lib/tokenizers"
import { useAuth } from "@/hooks/use-auth"

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
  onDelete: (tokenizer: TokenizerInfo) => Promise<void>
}

/** Lists read-only vocabularies with their provenance, figures and confirmed deletion. */
export function TokenizerCatalogue({
  tokenizers,
  onDelete,
}: TokenizerCatalogueProps) {
  if (tokenizers.length === 0)
    return <EmptyState title="No vocabularies in this category" />
  return (
    <ResourceList aria-label="Vocabulary catalogue">
      {tokenizers.map((tokenizer) => (
        <Row
          key={tokenizer.id}
          tokenizer={tokenizer}
          onDelete={() => onDelete(tokenizer)}
        />
      ))}
    </ResourceList>
  )
}

/** Presents one vocabulary and invokes deletion only after the existing confirmation. */
function Row({
  tokenizer,
  onDelete,
}: {
  tokenizer: TokenizerInfo
  onDelete: () => Promise<void>
}) {
  const { isAdmin } = useAuth()
  return (
    <ResourceRow className="flex items-start justify-between gap-4">
      <ResourceLink href={`/tokenizers/${tokenizer.id}`}>
        <ResourceTitle>{tokenizer.name}</ResourceTitle>
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
      </ResourceLink>
      <div className="shrink-0">
        {isAdmin && (
          <TokenizerDeleteButton tokenizer={tokenizer} onDelete={onDelete} />
        )}
      </div>
    </ResourceRow>
  )
}
