import { ConversationsView } from "@/components/conversations/conversations-view"
import { PageShell } from "@/components/ui/page-shell"

/** Displays the conversation ledger at the same reading width as projects. */
export default function ConversationsPage() {
  return (
    <PageShell width="reading">
      <ConversationsView />
    </PageShell>
  )
}
