"use client"

import { useArchitectureMetadata } from "@/hooks/use-architecture-metadata"
import type { ArchitectureKind } from "@/hooks/use-architecture-catalog"
import type {
  ResourceDetail,
  ResourceSummary,
} from "@/lib/architecture/contracts.generated"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Switch } from "@/components/ui/switch"
import { FormError } from "@/components/ui/error-state"
import { Panel } from "@/components/ui/panel"

export interface ArchitectureMetadataProps {
  kind: ArchitectureKind
  summary: ResourceSummary
  onSaved: (detail: ResourceDetail) => void
}

/** Edits display/archive metadata independently from accepted revisions and unsaved source, with optimistic conflict handling. */
export function ArchitectureMetadata({
  kind,
  summary,
  onSaved,
}: ArchitectureMetadataProps) {
  const form = useArchitectureMetadata(kind, summary, onSaved)
  return (
    <Panel variant="open" title="Resource details">
      <form
        className="grid gap-4"
        onSubmit={(event) => {
          event.preventDefault()
          void form.save()
        }}
      >
        <label className="grid gap-1 text-sm">
          Name
          <Input
            value={form.name}
            required
            maxLength={120}
            disabled={form.busy}
            onChange={(event) => form.setName(event.target.value)}
          />
        </label>
        <label className="grid gap-1 text-sm">
          Description
          <Textarea
            value={form.description}
            maxLength={2000}
            disabled={form.busy}
            onChange={(event) => form.setDescription(event.target.value)}
          />
        </label>
        <label className="flex items-center gap-2 text-sm">
          <Switch
            checked={form.archived}
            disabled={form.busy}
            onCheckedChange={form.setArchived}
          />
          Archived
        </label>
        {form.error && <FormError>{form.error}</FormError>}
        <Button
          type="submit"
          className="justify-self-start"
          disabled={!form.dirty || form.busy || !form.name.trim()}
        >
          {form.busy ? "Saving…" : "Save details"}
        </Button>
      </form>
    </Panel>
  )
}
