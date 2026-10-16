"use client"

import dynamic from "next/dynamic"
import { useRouter } from "next/navigation"

import { Button } from "@/components/ui/button"
import { FormError, LoadError } from "@/components/ui/error-state"
import { PageHeader } from "@/components/ui/page-header"
import { Skeleton } from "@/components/ui/skeleton"
import { useAuth, useSessionToken } from "@/hooks/use-auth"
import { useNewProcessor } from "@/hooks/use-new-processor"

const PythonSourceEditor = dynamic(
  () =>
    import("./python-source-editor").then(
      (module) => module.PythonSourceEditor
    ),
  { ssr: false }
)

/** Presents a page-native creation flow, scoped to the actual administrator session. */
export function NewProcessor() {
  const token = useSessionToken()
  const { isAdmin, isLoading } = useAuth()
  if (isLoading) return <Skeleton className="h-64 w-full" />
  if (!isAdmin) return <LoadError title="Administrator access required." />
  return <NewProcessorForm key={token} />
}

/** Authors the new processor as one file, retaining invalid drafts for correction. */
function NewProcessorForm() {
  const router = useRouter()
  const form = useNewProcessor((name) =>
    router.replace(`/processors/${encodeURIComponent(name)}`)
  )
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        void form.submit()
      }}
    >
      <PageHeader
        title="New processor"
        back={{ href: "/processors", label: "Processors" }}
        action={
          <Button
            type="submit"
            disabled={form.submitting || !form.source.trim()}
          >
            {form.submitting ? "Creating…" : "Create processor"}
          </Button>
        }
      />
      <section aria-labelledby="processor-source-heading" className="space-y-3">
        <h2
          id="processor-source-heading"
          className="font-display text-xl font-semibold"
        >
          Python source
        </h2>
        <p className="text-sm text-muted-foreground">
          Set the identifier and title in @processor, then write the
          transformation in run.
        </p>
        {form.error && <FormError>{form.error}</FormError>}
        <PythonSourceEditor
          value={form.source}
          readOnly={form.submitting}
          onChange={form.setSource}
          height="60vh"
        />
      </section>
    </form>
  )
}
