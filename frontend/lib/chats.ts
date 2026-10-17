import { apiFetch, getToken } from "./api"

export interface ChatSummary {
  id: string
  title: string
  createdAt: string
  tokens: number
  projectId: string | null
}

/** Validates a UUID from a network boundary; malformed identities reject before routing or writes. */
export function parseResourceId(value: unknown): string {
  if (
    typeof value !== "string" ||
    !/^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i.test(
      value
    )
  )
    throw new Error("The server returned an invalid resource identity.")
  return value
}

/** Validates and adapts a stored chat summary, retaining compatibility with omitted token/project fields. */
export function parseChatSummary(value: unknown): ChatSummary {
  if (typeof value !== "object" || value === null || Array.isArray(value))
    throw new Error("The server returned an invalid conversation.")
  const raw = value as Record<string, unknown>
  if (
    typeof raw.title !== "string" ||
    typeof raw.created_at !== "string" ||
    !Number.isFinite(Date.parse(raw.created_at))
  )
    throw new Error("The server returned an invalid conversation.")
  const tokens = raw.tokens ?? 0
  if (typeof tokens !== "number" || !Number.isSafeInteger(tokens) || tokens < 0)
    throw new Error("The server returned an invalid conversation token count.")
  return {
    id: parseResourceId(raw.chat_id),
    title: raw.title,
    createdAt: raw.created_at,
    tokens,
    projectId: raw.project_id == null ? null : parseResourceId(raw.project_id),
  }
}

/** Fetches this account's archive or a bounded recent slice and rejects malformed interpreted rows. */
export async function loadChats(
  token: string,
  signal?: AbortSignal,
  limit?: number
): Promise<ChatSummary[]> {
  const response = await apiFetch(
    limit === undefined ? "/chats" : `/chats?limit=${limit}`,
    {
      headers: { Authorization: `Bearer ${token}` },
      signal,
    }
  )
  if (!response.ok) throw new Error("Unable to load your chats.")
  const data: unknown = await response.json()
  if (
    typeof data !== "object" ||
    data === null ||
    !("chats" in data) ||
    !Array.isArray(data.chats)
  )
    throw new Error("The server returned an invalid conversation archive.")
  return data.chats.map(parseChatSummary)
}

/** Moves or removes a conversation's project for its current session; expired ownership or HTTP failures reject. */
export async function assignChatProject(
  token: string,
  chatId: string,
  projectId: string | null
): Promise<ChatSummary> {
  if (getToken() !== token)
    throw new Error("Your account changed. Please try again.")
  const response = await apiFetch(`/chats/${parseResourceId(chatId)}/project`, {
    method: "PUT",
    headers: { Authorization: `Bearer ${token}` },
    body: JSON.stringify({
      project_id: projectId === null ? null : parseResourceId(projectId),
    }),
  })
  if (!response.ok)
    throw new Error("Unable to update this conversation's project.")
  return parseChatSummary(await response.json())
}
