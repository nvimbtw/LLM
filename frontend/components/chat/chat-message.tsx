import type { ChatMessage } from "@/hooks/use-chat"
import { parseUserMarkdown } from "@/lib/user-parser"
import { RichContent, RichMessage } from "./rich-message"

export function ChatMessageBubble({ message }: { message: ChatMessage }) {
  // User messages use a markdown subset; assistant messages use the model's markup.
  if (message.role === "user") {
    const tree = parseUserMarkdown(message.content)
    return (
      <div className="flex w-full justify-end">
        <div className="max-w-[85%] rounded-2xl bg-card px-4 py-3 text-card-foreground ring-1 ring-border/50">
          <RichContent root={tree} />
        </div>
      </div>
    )
  }

  return (
    <div className="w-full">
      <RichMessage content={message.content} />
    </div>
  )
}
