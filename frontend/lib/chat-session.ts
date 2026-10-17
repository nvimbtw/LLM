import type { Attachment } from "./attachments"
import type { ChatMessage } from "./chat-messages"
import {
  createChatTurn,
  reconcileChatTurn,
  reduceChatTurn,
  type ChatTurn,
  type ChatTurnEvent,
} from "./chat-turn"
import {
  streamChat,
  type ChatStreamHandle,
  type ChatStreamHandlers,
  type SamplingSettings,
} from "./chat-socket"
import { ArtStream } from "./live-throbber/stream"
import type { ThinkBudget } from "./think-budget"

export interface QueuedMessage {
  id: string
  content: string
  modelId: string
  budget: ThinkBudget
}

export interface ChatSessionSnapshot {
  isSending: boolean
  queuedAhead: number | null
  queued: QueuedMessage[]
  error: string | null
}

export type ChatMessagesChange =
  | { type: "start"; content: string; attachments: Attachment[] }
  | { type: "stream"; message: ChatMessage }
  | { type: "history"; messages: ChatMessage[] }

export interface ChatSessionEffects {
  changeMessages: (change: ChatMessagesChange) => void
  loadMessages: () => Promise<ChatMessage[]>
  refreshChats: () => Promise<void>
}

export interface ChatSendOptions {
  system: string | null
  sampling: SamplingSettings
  historyFloor: string | null
}

interface ActiveTurn {
  state: ChatTurn
  resolve: () => void
  handle: ChatStreamHandle | null
}

const IDLE: ChatSessionSnapshot = {
  isSending: false,
  queuedAhead: null,
  queued: [],
  error: null,
}

/** Applies an immutable session notification to the visible message list; no transport effects occur here. */
export function applyChatMessagesChange(
  messages: ChatMessage[],
  change: ChatMessagesChange
): ChatMessage[] {
  if (change.type === "history") return change.messages
  if (change.type === "start") {
    return [
      ...messages,
      {
        role: "user",
        content: change.content,
        attachments: change.attachments.length ? change.attachments : undefined,
      },
      { role: "assistant", content: "" },
    ]
  }
  return messages.at(-1)?.role === "assistant"
    ? [...messages.slice(0, -1), change.message]
    : messages
}

/**
 * Owns one chat's socket and interjection queue, independent of React renders.
 * Terminal reconciliation and cancelled/disposed callbacks are guarded by turn identity.
 */
export class ChatSession {
  readonly tokenArt: ArtStream
  private view = IDLE
  private listeners = new Set<() => void>()
  private active: ActiveTurn | null = null
  private disposed = false
  private sendOptions: ((modelId: string) => ChatSendOptions) | null = null

  /** Creates a dormant chat session with explicit history/render effects and an injectable socket adapter. */
  constructor(
    private readonly chatId: string,
    private readonly effects: ChatSessionEffects,
    private readonly openStream: typeof streamChat = streamChat
  ) {
    this.tokenArt = new ArtStream(chatId)
  }

  /** Supplies current send-time preferences after mounting; configuration never starts a stream. */
  configure(sendOptions: (modelId: string) => ChatSendOptions): void {
    this.sendOptions = sendOptions
  }

  /** Returns the stable external-store snapshot, replaced only when session status changes. */
  getSnapshot = (): ChatSessionSnapshot => this.view

  /** Subscribes a renderer to status updates and returns its unsubscribe operation. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /** Replaces status fields and notifies current subscribers after the new snapshot is available. */
  private publish(change: Partial<ChatSessionSnapshot>): void {
    this.view = { ...this.view, ...change }
    for (const listener of this.listeners) listener()
  }

  /** Returns whether a callback still belongs to the live turn of an undisposed session. */
  private owns(turn: ActiveTurn): boolean {
    return !this.disposed && this.active === turn
  }

  /**
   * Starts a new turn or queues text steering for the active one, preserving its model/budget choice.
   * The returned promise settles at completion, error, cancellation, or disposal; interjections settle immediately.
   */
  send = (
    content: string,
    modelId: string,
    budget: ThinkBudget,
    attachments: Attachment[]
  ): Promise<void> => {
    if (this.disposed) return Promise.resolve()
    if (this.active) {
      const queued = [
        ...this.view.queued,
        { id: crypto.randomUUID(), content, modelId, budget },
      ]
      this.publish({ queued })
      this.active.handle?.interject(content)
      return Promise.resolve()
    }
    const options = this.sendOptions?.(modelId)
    if (!options)
      throw new Error("Chat session must be configured before sending.")
    this.tokenArt.reset(`${this.chatId}:${modelId}:${content}`)
    this.publish({ isSending: true, error: null })
    this.effects.changeMessages({ type: "start", content, attachments })
    return new Promise((resolve) => {
      const turn: ActiveTurn = {
        state: createChatTurn(),
        resolve,
        handle: null,
      }
      this.active = turn
      turn.handle = this.openStream(
        this.chatId,
        { text: content, attachmentIds: attachments.map((a) => a.id) },
        modelId,
        options.system,
        budget,
        options.sampling,
        options.historyFloor,
        this.handlers(turn)
      )
    })
  }

  /** Translates socket callbacks into typed turn events and terminal reconciliation. */
  private handlers(turn: ActiveTurn): ChatStreamHandlers {
    return {
      onToken: (value, tokenId) =>
        this.receive(turn, { type: "token", value, tokenId }),
      onToolCall: () => this.receive(turn, { type: "tool-call" }),
      onToolResult: (result) =>
        this.receive(turn, { type: "tool-result", result }),
      onInterjected: (text) =>
        this.receive(turn, { type: "interjected", text }),
      onDone: (info) => {
        void this.completeRound(turn, info)
      },
      onError: (error) => {
        if (this.owns(turn)) this.finish(turn, error)
      },
    }
  }

  /** Advances the live turn once per frame, removing only the first matching confirmed interjection. */
  private receive(turn: ActiveTurn, event: ChatTurnEvent): void {
    if (!this.owns(turn)) return
    if (event.type === "token") {
      this.tokenArt.push(event.value, event.tokenId)
      if (this.view.queuedAhead !== 0) this.publish({ queuedAhead: 0 })
    }
    if (event.type === "interjected") {
      const at = this.view.queued.findIndex(
        (message) => message.content === event.text
      )
      if (at !== -1)
        this.publish({
          queued: this.view.queued.filter((_, index) => index !== at),
        })
    }
    turn.state = reduceChatTurn(turn.state, event)
    this.effects.changeMessages({ type: "stream", message: turn.state.message })
  }

  /**
   * Accumulates round durations and reconciles only final replies with committed history.
   * Failed history reads retain streamed content; a failed sidebar refresh retains its prior totals.
   * Stale completions cannot write or drain another turn.
   */
  private async completeRound(
    turn: ActiveTurn,
    info: Parameters<ChatStreamHandlers["onDone"]>[0]
  ): Promise<void> {
    if (!this.owns(turn)) return
    turn.state = reduceChatTurn(turn.state, {
      type: "round-done",
      elapsedSecs: info.elapsedSecs,
    })
    if (!info.final) return
    const refresh = Promise.resolve()
      .then(() => this.effects.refreshChats())
      .catch(() => {})
    try {
      const messages = await this.effects.loadMessages()
      if (!this.owns(turn)) return
      this.effects.changeMessages({
        type: "history",
        messages: reconcileChatTurn(messages, turn.state),
      })
    } catch {
      if (!this.owns(turn)) return
    }
    await refresh
    if (!this.owns(turn)) return
    this.finish(turn)
  }

  /** Settles one turn before draining its unconfirmed queue, preserving send-time options and FIFO order. */
  private finish(turn: ActiveTurn, error: string | null = null): void {
    this.active = null
    this.publish({ isSending: false, queuedAhead: null, error })
    turn.resolve()
    const [next, ...queued] = this.view.queued
    if (!next) return
    this.publish({ queued })
    void this.send(next.content, next.modelId, next.budget, [])
  }

  /** Stops the active socket and sends any unconfirmed steering as the next ordinary turn. */
  cancel = (): void => {
    const turn = this.active
    if (!turn) return
    this.active = null
    turn.handle?.cancel()
    this.finish(turn)
  }

  /** Enables callbacks for a mounted subscriber, including React StrictMode effect reattachment. */
  activate(): void {
    this.disposed = false
  }

  /**
   * Releases this chat's socket without draining messages into a conversation being left.
   * Any later socket or history callbacks are ignored and an outstanding send is settled.
   */
  dispose(): void {
    this.disposed = true
    const turn = this.active
    this.active = null
    turn?.handle?.cancel()
    turn?.resolve()
    this.listeners.clear()
    this.view = IDLE
  }
}
