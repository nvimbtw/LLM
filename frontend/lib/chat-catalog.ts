import { loadChats, type ChatSummary } from "./chats"

export const RECENT_CHAT_LIMIT = 5

export interface ChatCatalogSnapshot {
  token: string | null
  recent: ChatSummary[]
  archive: ChatSummary[] | null
  selected: ChatSummary | null
  recentLoading: boolean
  recentLoaded: boolean
  archiveLoading: boolean
  recentError: string | null
  archiveError: string | null
}

type ChatLoader = typeof loadChats
type ListKind = "recent" | "archive"

interface ListRequest {
  controller: AbortController
  promise: Promise<void>
}

/** Owns account chat metadata and cancellable recent/archive reads; only mounted archive views request the full list. */
export class ChatCatalog {
  private snapshot: ChatCatalogSnapshot = {
    token: null,
    recent: [],
    archive: null,
    selected: null,
    recentLoading: false,
    recentLoaded: false,
    archiveLoading: false,
    recentError: null,
    archiveError: null,
  }
  private listeners = new Set<() => void>()
  private requests: Partial<Record<ListKind, ListRequest>> = {}
  private archiveReaders = 0
  private version = 0
  private sessionVersion = 0
  private modifiedAt = new Map<string, number>()

  /** Uses the production HTTP adapter unless a test supplies a controlled loader. */
  constructor(private readonly load: ChatLoader = loadChats) {}

  /** Returns the immutable current account snapshot for React subscriptions. */
  getSnapshot = (): ChatCatalogSnapshot => this.snapshot

  /** Subscribes one view and returns its cleanup function. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /** Captures the current metadata revision so history loaded before a write cannot undo it. */
  getVersion = (): number => this.version

  /** Publishes one coherent metadata transition to mounted views. */
  private publish(change: Partial<ChatCatalogSnapshot>): void {
    this.snapshot = { ...this.snapshot, ...change }
    this.listeners.forEach((listener) => listener())
  }

  /** Replaces account identity, discards private metadata and loads only lists with active consumers. */
  setSession(token: string | null): void {
    if (token === this.snapshot.token) {
      if (token && !this.requests.recent && this.snapshot.recentLoading)
        void this.refresh()
      return
    }
    this.dispose()
    this.modifiedAt.clear()
    this.publish({
      token,
      recent: [],
      archive: null,
      selected: null,
      recentLoading: Boolean(token),
      recentLoaded: false,
      archiveLoading: Boolean(token) && this.archiveReaders > 0,
      recentError: null,
      archiveError: null,
    })
    if (token) void this.refresh()
  }

  /** Retains the full archive while a page or picker uses it; the final release cancels speculative work. */
  retainArchive = (): (() => void) => {
    this.archiveReaders++
    if (this.archiveReaders === 1) void this.loadArchive()
    return () => {
      this.archiveReaders--
      if (this.archiveReaders === 0) {
        this.requests.archive?.controller.abort()
        delete this.requests.archive
        this.publish({
          archive: null,
          archiveLoading: false,
          archiveError: null,
        })
      }
    }
  }

  /** Loads or retries the full archive, sharing an existing request from the same session. */
  loadArchive = (): Promise<void> => this.read("archive", false)

  /** Refreshes recent chats and any currently mounted archive, awaiting both for mutation callers. */
  refresh = async (): Promise<void> => {
    await Promise.all([
      this.read("recent", true),
      this.archiveReaders > 0 ? this.read("archive", true) : Promise.resolve(),
    ])
  }

  /** Reads one list; replaced sessions, cancelled requests and older writes cannot publish stale rows. */
  private read(kind: ListKind, replace: boolean): Promise<void> {
    const token = this.snapshot.token
    if (!token) return Promise.resolve()
    const existing = this.requests[kind]
    if (existing && !replace) return existing.promise
    existing?.controller.abort()
    const controller = new AbortController()
    const version = this.version
    this.publish(
      kind === "recent"
        ? { recentLoading: true, recentError: null }
        : { archiveLoading: true, archiveError: null }
    )
    const request: ListRequest = {
      controller,
      promise: this.load(
        token,
        controller.signal,
        kind === "recent" ? RECENT_CHAT_LIMIT : undefined
      )
        .then(async (rows) => {
          if (this.requests[kind] !== request || controller.signal.aborted)
            return
          if (version !== this.version) return this.read(kind, true)
          if (kind === "recent")
            this.publish({
              recent: rows,
              recentLoading: false,
              recentLoaded: true,
            })
          else this.publish({ archive: rows, archiveLoading: false })
        })
        .catch((failure: unknown) => {
          if (this.requests[kind] !== request || controller.signal.aborted)
            return
          const error =
            failure instanceof Error
              ? failure.message
              : "Unable to load your chats."
          this.publish(
            kind === "recent"
              ? { recentLoading: false, recentLoaded: true, recentError: error }
              : { archiveLoading: false, archiveError: error }
          )
        })
        .finally(() => {
          if (this.requests[kind] === request) delete this.requests[kind]
        }),
    }
    this.requests[kind] = request
    return request.promise
  }

  /** Returns the selected chat's authoritative detail or a known list summary. */
  getSummary = (id: string): ChatSummary | undefined =>
    this.snapshot.selected?.id === id
      ? this.snapshot.selected
      : (this.snapshot.recent.find((chat) => chat.id === id) ??
        this.snapshot.archive?.find((chat) => chat.id === id))

  /** Accepts a guarded history read and synchronizes its metadata across loaded views. */
  rememberSummary = (summary: ChatSummary, version: number): void => {
    if (
      version < this.sessionVersion ||
      (this.modifiedAt.get(summary.id) ?? 0) > version ||
      !this.snapshot.token
    )
      return
    this.patchSummary(summary, true)
  }

  /** Replaces known copies of one summary while preserving the independently selected chat on writes. */
  private patchSummary(summary: ChatSummary, select: boolean): void {
    this.publish({
      selected:
        select || this.snapshot.selected?.id === summary.id
          ? summary
          : this.snapshot.selected,
      recent: this.snapshot.recent.map((chat) =>
        chat.id === summary.id ? summary : chat
      ),
      archive:
        this.snapshot.archive?.map((chat) =>
          chat.id === summary.id ? summary : chat
        ) ?? null,
    })
  }

  /** Applies committed metadata immediately and invalidates reads that started before the write. */
  updateSummary(summary: ChatSummary): void {
    this.version++
    this.modifiedAt.set(summary.id, this.version)
    this.patchSummary(summary, false)
  }

  /** Removes committed deletions immediately and invalidates older metadata reads. */
  remove(id: string): void {
    this.version++
    this.modifiedAt.set(id, this.version)
    this.publish({
      selected:
        this.snapshot.selected?.id === id ? null : this.snapshot.selected,
      recent: this.snapshot.recent.filter((chat) => chat.id !== id),
      archive: this.snapshot.archive?.filter((chat) => chat.id !== id) ?? null,
    })
  }

  /** Cancels all outstanding reads and invalidates history completions during cleanup or account replacement. */
  dispose(): void {
    this.version++
    this.sessionVersion = this.version
    this.requests.recent?.controller.abort()
    this.requests.archive?.controller.abort()
    this.requests = {}
  }
}
