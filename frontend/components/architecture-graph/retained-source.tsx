"use client"

import dynamic from "next/dynamic"
import { useSessionToken } from "@/hooks/use-auth"
import { useRetainedSource } from "@/hooks/use-retained-source"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { LoadError } from "@/components/ui/error-state"
import { Skeleton } from "@/components/ui/skeleton"

const SourceEditor = dynamic(
  () =>
    import("@/components/authoring/source-editor").then(
      (module) => module.SourceEditor
    ),
  { ssr: false }
)

export interface RetainedSourceSelection {
  revision: string
  line: number | null
  contentHash?: string
}

export interface RetainedSourceDialogProps {
  kind: "models" | "validations"
  id: string
  definitionId?: string
  selection: RetainedSourceSelection | null
  onClose: () => void
}

/** Mounts a read-only source dialog with fresh ownership for every actual session/resource/selection. */
export function RetainedSourceDialog(props: RetainedSourceDialogProps) {
  const token = useSessionToken()
  const selection = props.selection
  return (
    <Dialog
      open={Boolean(token && selection)}
      onOpenChange={(open) => {
        if (!open) props.onClose()
      }}
    >
      {token && selection && (
        <DialogContent className="max-w-5xl">
          <DialogHeader>
            <DialogTitle>Retained source</DialogTitle>
          </DialogHeader>
          <SourceContent
            key={`${token}:${props.kind}:${props.id}:${props.definitionId}:${selection.revision}:${selection.line}:${selection.contentHash}`}
            {...props}
            selection={selection}
          />
        </DialogContent>
      )}
    </Dialog>
  )
}

/** Reads exact retained bytes and cancels observation on replacement; this never starts or cancels durable work. */
function SourceContent({
  kind,
  id,
  definitionId,
  selection,
}: RetainedSourceDialogProps & {
  selection: RetainedSourceSelection
}) {
  const { value, error, retry } = useRetainedSource({
    kind,
    id,
    definitionId,
    revision: selection.revision,
    contentHash: selection.contentHash,
  })
  if (error)
    return (
      <LoadError
        title="Retained source did not load."
        hint={error}
        onRetry={retry}
      />
    )
  if (!value) return <Skeleton className="h-96 w-full" />
  const line =
    value.language === "json" ? undefined : (selection.line ?? undefined)
  return (
    <div className="grid min-w-0 gap-3">
      <p className="font-mono text-xs break-all">
        {value.file} · {value.content_hash}
      </p>
      {value.language === "json" ? (
        <>
          <p className="text-sm text-muted-foreground">
            Frozen builtin manifest. Python implementation source was not
            retained in this package.
          </p>
          <pre className="max-h-96 overflow-auto text-xs">{value.source}</pre>
        </>
      ) : (
        <SourceEditor
          language={value.language}
          ariaLabel="Frozen source, read only"
          value={value.source}
          readOnly
          onChange={() => {}}
          line={line}
        />
      )}
    </div>
  )
}
