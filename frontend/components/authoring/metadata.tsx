"use client"

import type { ReactNode } from "react"
import { useArchitectureMetadata } from "@/hooks/use-architecture-metadata"
import type { ArchitectureKind } from "@/hooks/use-architecture-catalog"
import type {
  ResourceDetail,
  ResourceSummary,
} from "@/lib/architecture/contracts.generated"
import { EditableTitle } from "@/components/ui/editable-title"
import { EditActions } from "@/components/ui/edit-actions"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import { FormError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"

export interface ArchitectureMetadataProps {
  kind: ArchitectureKind
  summary: ResourceSummary
  onSaved: (detail: ResourceDetail) => void
  sourceStatus: ReactNode
  actions: ReactNode
  back: { href: string; label: string }
}

/** Edits name, description and archive state in the resource header; its explicit metadata save/cancel remains independent of source-draft actions. */
export function ArchitectureMetadata({
  kind,
  summary,
  onSaved,
  sourceStatus,
  actions,
  back,
}: ArchitectureMetadataProps) {
  const form = useArchitectureMetadata(kind, summary, onSaved)
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        void form.save()
      }}
    >
      <PageHeader
        back={back}
        contentClassName="min-w-0 flex-1 basis-64"
        title={
          <EditableTitle
            aria-label={`${kind === "block" ? "Block" : "Architecture"} name`}
            value={form.name}
            required
            maxLength={120}
            disabled={form.busy}
            onChange={(event) => form.setName(event.target.value)}
          />
        }
        description={
          <>
            <Textarea
              variant="inline"
              aria-label={`${kind === "block" ? "Block" : "Architecture"} description`}
              rows={1}
              value={form.description}
              placeholder="Add a description…"
              maxLength={2000}
              disabled={form.busy}
              onChange={(event) => form.setDescription(event.target.value)}
            />
            <label className="flex items-center gap-2 text-xs">
              <Switch
                checked={form.archived}
                disabled={form.busy}
                onCheckedChange={form.setArchived}
              />
              Archived
            </label>
          </>
        }
        meta={sourceStatus}
        action={
          <div className="flex flex-wrap items-center gap-2">
            <EditActions
              dirty={form.dirty}
              busy={form.busy}
              disabled={!form.name.trim()}
              onCancel={form.reset}
            />
            {actions}
          </div>
        }
      />
      {form.error && <FormError className="mb-6">{form.error}</FormError>}
    </form>
  )
}
