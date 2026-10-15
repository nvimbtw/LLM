import { ConversationsView } from "@/components/conversations/conversations-view"
import { PageShell } from "@/components/ui/page-shell"

/** Displays the conversation ledger at the shared board width. */
export default function ConversationsPage() {
  return (
    <PageShell>
      <ConversationsView />
    </PageShell>
  )
}
