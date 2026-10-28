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
import { ResourceList, ResourceRow } from "@/components/ui/resource-list"
import { Panel } from "@/components/ui/panel"
import { EditableTitle } from "@/components/ui/editable-title"
import { EditActions } from "@/components/ui/edit-actions"
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
          className="mb-6 border-b-0 pb-0"
          back={{ href: "/tokenizers", label: "Tokenizers" }}
          contentClassName="w-full min-w-0 flex-auto sm:w-auto sm:min-w-48 sm:flex-1"
          title={
            isAdmin ? (
              <>
                <span className="sr-only">{tokenizer.name}</span>
                <EditableTitle
                  aria-label="Vocabulary name"
                  variant="inline"
                  value={fields.name}
                  maxLength={TOKENIZER_NAME_LIMIT * 2}
                  required
                  disabled={editor.busy}
                  onChange={(event) =>
                    editor.change("name", event.target.value)
                  }
                />
              </>
            ) : (
              tokenizer.name
            )
          }
          description={
            isAdmin ? (
              <Textarea
                variant="inline"
                aria-label="Vocabulary description"
                rows={1}
                placeholder="Add a description…"
                value={fields.description}
                maxLength={TOKENIZER_DESCRIPTION_LIMIT * 2}
                disabled={editor.busy}
                onChange={(event) =>
                  editor.change("description", event.target.value)
                }
                className="text-muted-foreground"
              />
            ) : (
              tokenizer.description || undefined
            )
          }
          action={
            isAdmin && (
              <div className="flex flex-wrap items-center gap-2">
                <EditActions
                  dirty={editor.changed}
                  busy={editor.busy}
                  disabled={!fields.name.trim()}
                  onCancel={editor.cancel}
                />
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
        <dl className="grid gap-x-8 gap-y-5 py-4 sm:grid-cols-2">
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
            <ResourceList density="compact">
              {tokenizer.datasets.map((dataset) => (
                <ResourceRow
                  density="compact"
                  key={dataset.id}
                  className="flex flex-wrap justify-between gap-3"
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
                </ResourceRow>
              ))}
            </ResourceList>
          ) : (
            <p className="text-sm text-muted-foreground">
              No datasets recorded.
            </p>
          )}
        </Panel>
        <Panel variant="open" title="Models">
          {tokenizer.models.length ? (
            <ResourceList density="compact">
              {tokenizer.models.map((model) => (
                <ResourceRow density="compact" key={model.id}>
                  <Link
                    href={`/models/${model.id}`}
                    className="font-display text-lg hover:underline"
                  >
                    {model.name}
                  </Link>
                </ResourceRow>
              ))}
            </ResourceList>
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
