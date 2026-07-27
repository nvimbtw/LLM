import { ChatThread } from "@/components/chat/chat-thread"

export default async function ChatPage({
  params,
}: {
  params: Promise<{ id: string }>
}) {
  const { id } = await params
  return <ChatThread chatId={id} />
}
