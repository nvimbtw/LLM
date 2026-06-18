import { ConversationsView } from "@/components/conversations/conversations-view";

export default function ConversationsPage() {
  return (
    <div className="mx-auto w-full max-w-3xl flex-1 px-4 py-8 sm:px-6">
      <header className="mb-8">
        <h1 className="font-heading text-2xl font-semibold tracking-tight">
          Conversations
        </h1>
        <p className="mt-1 text-sm text-muted-foreground">
          All of your chats in one place.
        </p>
      </header>
      <ConversationsView />
    </div>
  );
}
