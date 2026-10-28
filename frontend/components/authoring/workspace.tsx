"use client"

import dynamic from "next/dynamic"
import Link from "next/link"
import { AdminBoundary } from "@/components/admin/admin-boundary"
import { ArchitectureMetadata } from "./metadata"
import { DraftValidation } from "./draft-validation"
import { RevisionHistory } from "./revision-history"
import { DependencyPicker } from "./dependency-picker"
import { useArchitectureWorkspace } from "@/hooks/use-architecture-workspace"
import type { ArchitectureKind } from "@/hooks/use-architecture-catalog"
import { Button } from "@/components/ui/button"
import { FormError, LoadError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import { Panel } from "@/components/ui/panel"
import { Skeleton } from "@/components/ui/skeleton"
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

const SourceEditor = dynamic(
  () => import("./source-editor").then((module) => module.SourceEditor),
  { ssr: false }
)

export interface ArchitectureWorkspaceProps {
  kind: ArchitectureKind
  id: string
  initialRevision?: string
  line?: number
}

/** Mounts the editor only for an administrator and scopes all private draft state to its actual session/resource. */
export function ArchitectureWorkspace(props: ArchitectureWorkspaceProps) {
  return (
    <AdminBoundary>
      <WorkspaceContent key={`${props.kind}:${props.id}`} {...props} />
    </AdminBoundary>
  )
}

/** Edits exact invalid/valid draft bytes while displaying accepted identity and requiring explicit discard before a server reload. */
function WorkspaceContent({
  kind,
  id,
  initialRevision,
  line,
}: ArchitectureWorkspaceProps) {
  const workspace = useArchitectureWorkspace(kind, id)
  const { detail, draft, document } = workspace
  if (workspace.loading) return <Skeleton className="h-96 w-full" />
  if (workspace.error || !detail || !draft || !document)
    return (
      <LoadError
        title="Workspace did not load."
        hint={workspace.error ?? undefined}
        onRetry={workspace.reload}
      />
    )
  const root = kind === "block" ? "/arch/blocks" : "/arch"
  const summary = detail.summary
  const readOnly = summary.origin === "builtin" || summary.archived
  const dirty = draft.text !== draft.savedText
  const back = {
    href: root,
    label: kind === "block" ? "Blocks" : "Architectures",
  }
  const sourceStatus = `Source draft ${draft.saved.draft_version}${dirty ? " · Unsaved changes" : " · Saved"}`
  const actions = (
    <div className="flex flex-wrap gap-2">
      {summary.accepted_revision && (
        <Button
          nativeButton={false}
          variant="outline"
          render={
            <Link
              href={`${root}/${kind === "block" ? "new-block" : "new-arch"}?copy=${id}&revision=${summary.accepted_revision.revision_id}`}
            />
          }
        >
          Copy accepted {kind}
        </Button>
      )}
      <AlertDialog>
        <AlertDialogTrigger
          render={
            <Button type="button" variant="outline" disabled={draft.saving} />
          }
        >
          Reload server draft
        </AlertDialogTrigger>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Reload server draft?</AlertDialogTitle>
            <AlertDialogDescription>
              Local unsaved source edits will be discarded. Metadata edits are
              saved separately.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep editing</AlertDialogCancel>
            <AlertDialogAction onClick={workspace.reload}>
              Reload source
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {!readOnly && (
        <Button
          type="button"
          disabled={!dirty || draft.saving || draft.conflict}
          onClick={() => void document.save()}
        >
          {draft.saving ? "Saving source…" : "Save draft"}
        </Button>
      )}
    </div>
  )
  return (
    <>
      {summary.origin === "custom" ? (
        <ArchitectureMetadata
          key={summary.metadata_version}
          kind={kind}
          summary={summary}
          onSaved={workspace.updateMetadata}
          back={back}
          sourceStatus={sourceStatus}
          actions={actions}
        />
      ) : (
        <PageHeader
          title={summary.name}
          back={{
            href: root,
            label: kind === "block" ? "Blocks" : "Architectures",
          }}
          description={summary.description || undefined}
          meta={sourceStatus}
          action={actions}
        />
      )}
      <div className="grid gap-8">
        <dl className="grid gap-3 py-4 text-sm sm:grid-cols-2">
          <div>
            <dt className="eyebrow text-muted-foreground">Accepted revision</dt>
            <dd className="mt-1 font-mono break-all">
              {summary.accepted_revision?.content_hash ?? "None"}
            </dd>
          </div>
          <div>
            <dt className="eyebrow text-muted-foreground">Saved draft</dt>
            <dd className="mt-1 font-mono break-all">
              {draft.saved.content_hash}
            </dd>
          </div>
        </dl>
        <Panel
          variant="open"
          title={kind === "block" ? "Python draft" : "YAML draft"}
        >
          <div className="grid gap-3">
            {readOnly && (
              <p className="text-sm text-muted-foreground">
                {summary.origin === "builtin"
                  ? "Builtin source is read-only. Copy an accepted revision to create your own resource."
                  : "This resource is archived."}
              </p>
            )}
            {draft.error && <FormError>{draft.error}</FormError>}
            {workspace.storageError && (
              <FormError>{workspace.storageError}</FormError>
            )}
            {draft.conflict && (
              <p className="text-sm text-muted-foreground">
                Your local bytes remain in the editor. Reload only after keeping
                any edits you need.
              </p>
            )}
            <SourceEditor
              language={kind === "block" ? "python" : "yaml"}
              ariaLabel={
                kind === "block"
                  ? "Block Python draft"
                  : "Architecture YAML draft"
              }
              value={draft.text}
              readOnly={readOnly}
              onChange={(text) => document.edit(text)}
              height="60vh"
            />
          </div>
        </Panel>
        {kind === "architecture" && !readOnly && <DependencyPicker />}
        <DraftValidation
          key={`${draft.saved.draft_version}:${draft.saved.content_hash}`}
          kind={kind}
          id={id}
          document={document}
          draft={draft}
          readOnly={readOnly}
          onAccepted={workspace.updateMetadata}
        />
        <RevisionHistory
          key={`${initialRevision}:${line}`}
          summary={summary}
          initialRevision={initialRevision}
          line={line}
        />
      </div>
    </>
  )
}
