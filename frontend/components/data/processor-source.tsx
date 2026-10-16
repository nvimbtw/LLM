"use client"

import dynamic from "next/dynamic"
import * as React from "react"

import { Button } from "@/components/ui/button"
import { FormError, LoadError } from "@/components/ui/error-state"
import { useAuth } from "@/hooks/use-auth"
import { useProcessorSource } from "@/hooks/use-processor-source"

const PythonSourceEditor = dynamic(
  () =>
    import("./python-source-editor").then(
      (module) => module.PythonSourceEditor
    ),
  { ssr: false }
)

export interface ProcessorSourceProps {
  name: string
  onSaved: () => void
  height?: string
  actions?: React.ReactNode
}

/** Shows saved source, active identity and the administrator's edit/save controls. */
export function ProcessorSource({
  name,
  onSaved,
  height,
  actions,
}: ProcessorSourceProps) {
  const { isAdmin } = useAuth()
  const source = useProcessorSource(name, isAdmin, onSaved)
  const [editing, setEditing] = React.useState(false)
  if (source.loading)
    return <p className="py-3 text-sm text-muted-foreground">Loading source…</p>
  if (!source.info)
    return (
      <LoadError
        title="Source did not load."
        hint={source.error ?? undefined}
        onRetry={source.reload}
      />
    )
  const info = source.info
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3 text-xs text-muted-foreground">
        <span>{info.file}</span>
        <code title={info.version_hash}>
          saved {info.version_hash.slice(0, 12)}
        </code>
        {info.active_version_hash !== info.version_hash && (
          <code title={info.active_version_hash}>
            active {info.active_version_hash.slice(0, 12)}
          </code>
        )}
        {isAdmin &&
          info.editable &&
          (!editing ? (
            <Button
              variant="outline"
              size="sm"
              onClick={() => setEditing(true)}
            >
              Edit source
            </Button>
          ) : (
            <>
              <Button
                size="sm"
                disabled={source.saving || source.draft === info.source}
                onClick={source.save}
              >
                {source.saving ? "Saving…" : "Save source"}
              </Button>
              <Button
                variant="ghost"
                size="sm"
                disabled={source.saving}
                onClick={() => {
                  source.reset()
                  setEditing(false)
                }}
              >
                Close editor
              </Button>
            </>
          ))}
      </div>
      {actions}
      {info.reload_error && (
        <FormError>
          Saved. Preview uses the active version. {info.reload_error}
        </FormError>
      )}
      {source.error && <FormError>{source.error}</FormError>}
      <PythonSourceEditor
        value={source.draft}
        readOnly={!editing || source.saving || !isAdmin || !info.editable}
        onChange={source.setDraft}
        height={height}
      />
    </div>
  )
}
