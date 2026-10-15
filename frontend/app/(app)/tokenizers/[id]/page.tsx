import { TokenizerDetailView } from "@/components/tokenizers/tokenizer-detail"
import { PageShell } from "@/components/ui/page-shell"

/** Opens one vocabulary's inline metadata editor and recorded build facts. */
export default async function TokenizerPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return (
    <PageShell>
      <TokenizerDetailView tokenizerId={id} />
    </PageShell>
  )
}
