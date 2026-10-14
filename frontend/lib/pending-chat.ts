import type { Attachment } from "./attachments"
import type { ThinkBudget } from "./think-budget"

export interface PendingMessage {
  content: string
  modelId: string
  budget: ThinkBudget
  attachments: Attachment[]
}

/** Owns a session's one-shot first-message handoff, preserving the chosen model, budget and uploaded rows across navigation. */
export class PendingChatMessages {
  private token: string | null = null
  private messages = new Map<string, PendingMessage>()

  /** Discards the previous account's drafts when session identity changes. */
  setSession(token: string | null): void {
    if (token === this.token) return
    this.token = token
    this.messages.clear()
  }

  /** Stores a created conversation's first message only for its active owning session. */
  put(token: string, chatId: string, message: PendingMessage): void {
    if (token !== this.token)
      throw new Error("Your account changed. Please try again.")
    if (this.messages.size >= 10)
      this.messages.delete(this.messages.keys().next().value!)
    this.messages.set(chatId, message)
  }

  /** Takes and removes one pending message, returning nothing for another account or a repeated read. */
  take(token: string | null, chatId: string): PendingMessage | null {
    this.setSession(token)
    const message = this.messages.get(chatId) ?? null
    this.messages.delete(chatId)
    return message
  }
}
