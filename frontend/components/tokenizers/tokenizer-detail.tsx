"use client"

import Link from "next/link"
import { useRouter } from "next/navigation"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useTokenizer } from "@/hooks/use-tokenizer"
import {
  TOKENIZER_NAME_LIMIT,
  TOKENIZER_DESCRIPTION_LIMIT,
} from "@/lib/tokenizers"
import { formatBytes } from "@/lib/data"
import { PageHeader } from "@/components/ui/page-header"
import { Panel } from "@/components/ui/panel"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { LoadError, FormError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"
import { TokenizerDeleteButton } from "./tokenizer-delete-button"

export interface TokenizerDetailViewProps {
  tokenizerId: string
}

/** Isolates vocabulary metadata and draft lifetime by the current account and resource identity. */
export function TokenizerDetailView({ tokenizerId }: TokenizerDetailViewProps) {
  const token = useSessionToken()
  return token ? (
    <TokenizerDetailContent
      key={`${token}:${tokenizerId}`}
      tokenizerId={tokenizerId}
    />
  ) : null
}

/** Presents editable display metadata, immutable build facts, recorded datasets and model references. */
function TokenizerDetailContent({ tokenizerId }: TokenizerDetailViewProps) {
  const router = useRouter()
  const { isAdmin } = useAuth()
  const editor = useTokenizer(tokenizerId, () => router.push("/tokenizers"))
  const { tokenizer, fields } = editor
  if (editor.loading) return <Skeleton className="h-64 w-full" />
  if (editor.error || !tokenizer)
    return <LoadError title={editor.error ?? "Vocabulary not found."} />
  return (
    <>
      <form
        onSubmit={(event) => {
          event.preventDefault()
          void editor.save()
        }}
      >
        <PageHeader
          back={{ href: "/tokenizers", label: "Tokenizers" }}
          contentClassName="min-w-48 flex-1"
          title={
            isAdmin ? (
              <>
                <span className="sr-only">{tokenizer.name}</span>
                <Input
                  aria-label="Vocabulary name"
                  variant="inline"
                  value={fields.name}
                  maxLength={TOKENIZER_NAME_LIMIT * 2}
                  required
                  disabled={editor.busy}
                  onChange={(event) =>
                    editor.change("name", event.target.value)
                  }
                  className="h-auto px-0 py-0 font-display text-3xl font-semibold tracking-tight md:text-3xl"
                />
              </>
            ) : (
              tokenizer.name
            )
          }
          description={
            isAdmin ? (
              <Textarea
                aria-label="Vocabulary description"
                rows={1}
                placeholder="Add a description…"
                value={fields.description}
                maxLength={TOKENIZER_DESCRIPTION_LIMIT * 2}
                disabled={editor.busy}
                onChange={(event) =>
                  editor.change("description", event.target.value)
                }
                className="min-h-5 resize-none rounded-none border-0 bg-transparent px-0 py-0 text-sm text-muted-foreground shadow-none focus-visible:ring-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring/50 dark:bg-transparent"
              />
            ) : (
              tokenizer.description || undefined
            )
          }
          action={
            isAdmin && (
              <div className="flex flex-wrap items-center gap-2">
                {editor.changed && (
                  <>
                    <Button
                      type="submit"
                      disabled={editor.busy || !fields.name.trim()}
                    >
                      {editor.busy ? "Saving…" : "Save changes"}
                    </Button>
                    <Button
                      type="button"
                      variant="ghost"
                      disabled={editor.busy}
                      onClick={editor.cancel}
                    >
                      Cancel
                    </Button>
                  </>
                )}
                <TokenizerDeleteButton
                  tokenizer={tokenizer}
                  onDelete={editor.remove}
                  iconOnly={false}
                  disabled={editor.busy}
                />
              </div>
            )
          }
        />
        {editor.failure && (
          <FormError className="mb-6">{editor.failure}</FormError>
        )}
      </form>
      <div className="space-y-8">
        <dl className="grid gap-x-8 gap-y-5 border-y border-border/60 py-4 sm:grid-cols-2">
          <Fact
            label="Vocabulary"
            value={tokenizer.vocabSize?.toLocaleString() ?? "—"}
          />
          <Fact
            label="Merges"
            value={tokenizer.merges?.toLocaleString() ?? "—"}
          />
          <Fact
            label="Files"
            value={tokenizer.files?.toLocaleString() ?? "—"}
          />
          <Fact
            label="Size"
            value={tokenizer.hasVocab ? formatBytes(tokenizer.bytes) : "—"}
          />
          <Fact
            label="Learning ceiling"
            value={tokenizer.maxVocab?.toLocaleString() ?? "—"}
          />
          <Fact
            label="Origin"
            value={tokenizer.implicit ? "Learned by a run" : "Named build"}
          />
          <Fact
            label="Built"
            value={
              tokenizer.builtAt
                ? new Date(tokenizer.builtAt).toLocaleString()
                : "—"
            }
          />
          <Fact
            label="Status"
            value={tokenizer.hasVocab ? "Available" : "Missing on disk"}
          />
          <Fact label="Identity" value={tokenizer.id} />
          <Fact label="Fingerprint" value={tokenizer.fingerprint ?? "—"} />
        </dl>
        <Panel variant="open" title="Datasets">
          {tokenizer.datasets.length ? (
            <ul className="divide-y divide-border/60">
              {tokenizer.datasets.map((dataset) => (
                <li
                  key={dataset.id}
                  className="flex flex-wrap justify-between gap-3 py-3"
                >
                  <Link
                    href={`/data/${encodeURIComponent(dataset.id)}`}
                    className="text-sm hover:underline"
                  >
                    {dataset.name}
                  </Link>
                  <span className="text-xs text-muted-foreground tabular-nums">
                    {dataset.documents.toLocaleString()} documents
                    {dataset.count !== null
                      ? ` · limit ${dataset.count.toLocaleString()}`
                      : ""}
                  </span>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">
              No datasets recorded.
            </p>
          )}
        </Panel>
        <Panel variant="open" title="Models">
          {tokenizer.models.length ? (
            <ul className="divide-y divide-border/60">
              {tokenizer.models.map((model) => (
                <li key={model.id} className="py-3">
                  <Link
                    href={`/models/${model.id}`}
                    className="font-display text-lg hover:underline"
                  >
                    {model.name}
                  </Link>
                </li>
              ))}
            </ul>
          ) : (
            <p className="text-sm text-muted-foreground">
              No model references recorded.
            </p>
          )}
        </Panel>
      </div>
    </>
  )
}

/** Renders one recorded build fact with a quiet label and tabular figures. */
function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <dt className="eyebrow text-muted-foreground">{label}</dt>
      <dd className="mt-1 text-sm break-words tabular-nums">{value}</dd>
    </div>
  )
}
