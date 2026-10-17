import { apiFetch, getToken, SESSION_CHANGED_EVENT } from "./api"
import type { AttachmentWire } from "./attachments"
import type { Feedback } from "./feedback"
import { parseChatSummary, type ChatSummary } from "./chats"

export interface MessageWire {
  message_id: string
  role: "user" | "assistant" | "tool"
  content: string
  model_name: string | null
  feedback: Feedback | null
  attachments: AttachmentWire[]
}

const HISTORY_TTL_MS = 30_000
const HISTORY_CACHE_LIMIT = 10

interface HistoryEntry {
  expiresAt: number
  messages?: MessageWire[]
  summary?: ChatSummary
  pending: Promise<MessageWire[]>
}

// Shared by hover and navigation; bounded and session-local so private messages
// cannot survive a logout or be reused by another account.
const history = new Map<string, HistoryEntry>()
let session: string | null = null
let subscribed = false

function currentEntry(chatId: string): HistoryEntry | undefined {
  if (!subscribed && typeof window !== "undefined") {
    window.addEventListener(SESSION_CHANGED_EVENT, () => history.clear())
    window.addEventListener("storage", () => history.clear())
    subscribed = true
  }
  const token = getToken()
  if (session !== token) {
    history.clear()
    session = token
  }
  const entry = history.get(chatId)
  if (entry && entry.expiresAt <= Date.now()) {
    history.delete(chatId)
    return undefined
  }
  return entry
}

export function cachedChatHistory(chatId: string): MessageWire[] | undefined {
  return currentEntry(chatId)?.messages
}

/** Returns metadata from the same account-scoped request that loaded or prefetched a chat. */
export function cachedChatSummary(chatId: string): ChatSummary | undefined {
  return currentEntry(chatId)?.summary
}

export function invalidateChatHistory(chatId: string): void {
  history.delete(chatId)
}

export function loadChatHistory(chatId: string): Promise<MessageWire[]> {
  const cached = currentEntry(chatId)
  if (cached) return cached.pending
  const entry: HistoryEntry = {
    expiresAt: Date.now() + HISTORY_TTL_MS,
    pending: apiFetch(`/chats/${chatId}`)
      .then(async (response) => {
        if (!response.ok) throw new Error("Unable to load this chat.")
        const data: unknown = await response.json()
        if (
          typeof data !== "object" ||
          data === null ||
          !("messages" in data) ||
          !Array.isArray(data.messages)
        )
          throw new Error("The server returned an invalid conversation.")
        entry.summary = parseChatSummary(data)
        entry.messages = data.messages
        return data.messages
      })
      .catch((error: unknown) => {
        if (history.get(chatId) === entry) history.delete(chatId)
        throw error
      }),
  }
  if (history.size >= HISTORY_CACHE_LIMIT) {
    const oldest = history.keys().next().value
    if (oldest !== undefined) history.delete(oldest)
  }
  history.set(chatId, entry)
  return entry.pending
}

export function prefetchChatHistory(chatId: string): void {
  if (!getToken()) return
  void loadChatHistory(chatId).catch(() => {
    // Speculation must stay quiet; clicking retries through the normal load UI.
  })
}
