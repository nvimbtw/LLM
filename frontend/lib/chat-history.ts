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

export interface ChatHistory {
  messages: MessageWire[]
  summary: ChatSummary
}

interface HistoryEntry {
  expiresAt: number
  result?: ChatHistory
  pending: Promise<ChatHistory>
}

/** Validates interpreted attachment metadata without fetching or retaining file bytes. */
function parseHistoryAttachment(value: unknown): AttachmentWire {
  if (typeof value !== "object" || value === null)
    throw new Error("Invalid chat attachment.")
  const raw = value as Record<string, unknown>
  if (
    typeof raw.attachment_id !== "string" ||
    typeof raw.filename !== "string" ||
    typeof raw.media_type !== "string" ||
    typeof raw.size !== "number" ||
    !Number.isSafeInteger(raw.size) ||
    raw.size < 0
  )
    throw new Error("Invalid chat attachment.")
  return {
    attachment_id: raw.attachment_id,
    filename: raw.filename,
    media_type: raw.media_type,
    size: raw.size,
  }
}

/** Validates one stored message while retaining compatibility with omitted nullable metadata. */
function parseHistoryMessage(value: unknown): MessageWire {
  if (typeof value !== "object" || value === null)
    throw new Error("Invalid chat message.")
  const raw = value as Record<string, unknown>
  if (
    typeof raw.message_id !== "string" ||
    typeof raw.content !== "string" ||
    (raw.role !== "user" && raw.role !== "assistant" && raw.role !== "tool") ||
    (raw.model_name != null && typeof raw.model_name !== "string") ||
    (raw.feedback != null &&
      raw.feedback !== "up" &&
      raw.feedback !== "down") ||
    (raw.attachments != null && !Array.isArray(raw.attachments))
  )
    throw new Error("Invalid chat message.")
  return {
    message_id: raw.message_id,
    role: raw.role,
    content: raw.content,
    model_name: raw.model_name ?? null,
    feedback: raw.feedback ?? null,
    attachments: Array.isArray(raw.attachments)
      ? raw.attachments.map(parseHistoryAttachment)
      : [],
  }
}

// Shared by hover and navigation; bounded and session-local so private messages
// cannot survive a logout or be reused by another account.
const history = new Map<string, HistoryEntry>()
let session: string | null = null
let subscribed = false

/** Finds an unexpired entry after clearing snapshots owned by a different browser session. */
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

/** Returns cached messages and metadata together, or undefined before the request finishes or after expiry. */
export function cachedChatHistory(chatId: string): ChatHistory | undefined {
  return currentEntry(chatId)?.result
}

/** Invalidates both messages and their metadata before a write or authoritative reload. */
export function invalidateChatHistory(chatId: string): void {
  history.delete(chatId)
}

/** Shares one history/detail request between prefetch and navigation; failed requests remain retryable. */
export function loadChatHistory(chatId: string): Promise<ChatHistory> {
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
        entry.result = {
          summary: parseChatSummary(data),
          messages: data.messages.map(parseHistoryMessage),
        }
        return entry.result
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

/** Warms session-owned history without surfacing speculative errors; navigation retries normally. */
export function prefetchChatHistory(chatId: string): void {
  if (!getToken()) return
  void loadChatHistory(chatId).catch(() => {})
}
