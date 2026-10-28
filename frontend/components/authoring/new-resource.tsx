"use client"

import dynamic from "next/dynamic"
import { useRouter } from "next/navigation"
import { AdminBoundary } from "@/components/admin/admin-boundary"
import { useNewArchitectureResource } from "@/hooks/use-new-architecture-resource"
import type { ArchitectureKind } from "@/hooks/use-architecture-catalog"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { FormError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import { Skeleton } from "@/components/ui/skeleton"

const SourceEditor = dynamic(
  () => import("./source-editor").then((module) => module.SourceEditor),
  { ssr: false }
)
export interface NewArchitectureResourceProps {
  kind: ArchitectureKind
  copy?: string
  revision?: string
}

/** Gates resource creation/copy before mounting any private requests and scopes local input to the selected immutable copy. */
export function NewArchitectureResource(props: NewArchitectureResourceProps) {
  return (
    <AdminBoundary>
      <NewResourceForm
        key={`${props.kind}:${props.copy}:${props.revision}`}
        {...props}
      />
    </AdminBoundary>
  )
}

/** Creates an editable draft from exact entered/copied bytes without claiming syntax, capture or revision acceptance. */
function NewResourceForm({
  kind,
  copy,
  revision,
}: NewArchitectureResourceProps) {
  const router = useRouter()
  const root = kind === "block" ? "/arch/blocks" : "/arch"
  const form = useNewArchitectureResource(kind, copy, revision, (id) =>
    router.replace(`${root}/${id}`)
  )
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        void form.submit()
      }}
    >
      <PageHeader
        title={`${copy ? "Copy" : "New"} ${kind}`}
        back={{
          href: root,
          label: kind === "block" ? "Blocks" : "Architectures",
        }}
        action={
          <Button
            type="submit"
            disabled={
              !form.copyReady ||
              !form.name.trim() ||
              form.submitting ||
              form.uncertain
            }
          >
            {form.submitting ? "Creating…" : "Create draft"}
          </Button>
        }
      />
      <div className="grid gap-5">
        {form.error && <FormError>{form.error}</FormError>}
        {!form.copyReady ? (
          <Skeleton className="h-64 w-full" />
        ) : (
          <>
            <label className="grid gap-1 text-sm">
              Name
              <Input
                value={form.name}
                maxLength={120}
                required
                disabled={form.submitting}
                onChange={(event) => form.setName(event.target.value)}
              />
            </label>
            <label className="grid gap-1 text-sm">
              Description
              <Textarea
                value={form.description}
                maxLength={2000}
                disabled={form.submitting}
                onChange={(event) => form.setDescription(event.target.value)}
              />
            </label>
            <section
              className="grid gap-3"
              aria-label={kind === "block" ? "Python source" : "YAML recipe"}
            >
              <h2 className="font-display text-xl font-semibold">
                {kind === "block" ? "Python source" : "YAML recipe"}
              </h2>
              <p className="text-sm text-muted-foreground">
                Invalid source can be saved as a draft. Validation and
                acceptance happen separately.
              </p>
              <SourceEditor
                language={kind === "block" ? "python" : "yaml"}
                ariaLabel={
                  kind === "block"
                    ? "New block Python source"
                    : "New architecture YAML recipe"
                }
                value={form.text}
                readOnly={form.submitting}
                onChange={form.setText}
                height="60vh"
              />
            </section>
          </>
        )}
      </div>
    </form>
  )
}
