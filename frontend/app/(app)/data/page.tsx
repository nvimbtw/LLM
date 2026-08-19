import { DataView } from "@/components/data/data-view"
import { PageHeader } from "@/components/ui/page-header"
import { PageShell } from "@/components/ui/page-shell"

export default function DataPage() {
  return (
    <PageShell width="board">
      <PageHeader
        title="Data"
        description={
          <p>
            Datasets before they are training data. Pull a slice of something
            from Hugging Face, look at what actually arrived, then clean it —
            watching each step on a handful of documents before running it over
            all of them.
          </p>
        }
      />
      <DataView />
    </PageShell>
  )
}
