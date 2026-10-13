import {
  discardAttachment,
  uploadAttachment,
  MAX_ATTACHMENTS_PER_MESSAGE,
  type Attachment,
} from "./attachments"

export interface PendingAttachment {
  key: string
  file: File
  kind: "image" | "file"
  previewUrl?: string
  status: "uploading" | "ready" | "failed"
  attachmentId?: string
  error?: string
}

interface AttachmentEffects {
  upload: (file: File) => Promise<Attachment>
  discard: (id: string) => Promise<void>
  preview: (file: File) => string
  revoke: (url: string) => void
}

const BROWSER_EFFECTS: AttachmentEffects = {
  upload: uploadAttachment,
  discard: discardAttachment,
  preview: (file) => URL.createObjectURL(file),
  revoke: (url) => URL.revokeObjectURL(url),
}

/** Projects only completed uploads into a question's metadata, preserving pick order. */
export function readyAttachments(
  pending: readonly PendingAttachment[]
): Attachment[] {
  return pending.flatMap((item) =>
    item.status === "ready" && item.attachmentId
      ? [
          {
            id: item.attachmentId,
            filename: item.file.name,
            mediaType: item.file.type,
            size: item.file.size,
          },
        ]
      : []
  )
}

/**
 * Owns one draft's uploads and preview URLs outside React's replayable state updates.
 * Release keeps claimed server rows; discard/remove also delete unsent rows, including uploads that finish later.
 */
export class ComposerAttachments {
  private items: PendingAttachment[] = []
  private listeners = new Set<() => void>()
  private abandoned = new Set<string>()

  /** Creates an empty draft, with browser effects by default and explicit fakes available to lifecycle tests. */
  constructor(private readonly effects: AttachmentEffects = BROWSER_EFFECTS) {}

  /** Returns a stable snapshot until an upload or draft action changes it. */
  getSnapshot = (): PendingAttachment[] => this.items

  /** Registers a draft observer and returns its unsubscribe operation. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  /** Publishes a replacement list after effects have been performed by the action that owns them. */
  private publish(items: PendingAttachment[]): void {
    this.items = items
    for (const listener of this.listeners) listener()
  }

  /** Revokes local previews and optionally discards ready server rows; failed deletes leave server cleanup responsible. */
  private releaseItems(items: PendingAttachment[], discard: boolean): void {
    for (const item of items) {
      if (item.previewUrl) this.effects.revoke(item.previewUrl)
      if (discard && item.status === "uploading") this.abandoned.add(item.key)
      if (discard && item.attachmentId)
        void this.effects.discard(item.attachmentId).catch(() => {})
    }
  }

  /** Stores selected files immediately; excess picks remain visible as failures and consume no upload requests. */
  addFiles = (files: FileList | File[]): void => {
    const room = MAX_ATTACHMENTS_PER_MESSAGE - this.items.length
    const staged = Array.from(files).map(
      (file, index): PendingAttachment => ({
        key: crypto.randomUUID(),
        file,
        kind: file.type.startsWith("image/") ? "image" : "file",
        previewUrl: file.type.startsWith("image/")
          ? this.effects.preview(file)
          : undefined,
        status: index < room ? "uploading" : "failed",
        error:
          index < room
            ? undefined
            : `Up to ${MAX_ATTACHMENTS_PER_MESSAGE} files per message.`,
      })
    )
    this.publish([...this.items, ...staged])
    for (const item of staged.slice(0, Math.max(0, room))) {
      void this.effects.upload(item.file).then(
        (stored) => this.complete(item.key, stored),
        (error: unknown) => this.fail(item.key, error)
      )
    }
  }

  /** Settles a successful upload or removes its unclaimed row if the draft abandoned it while uploading. */
  private complete(key: string, stored: Attachment): void {
    if (this.abandoned.delete(key)) {
      void this.effects.discard(stored.id).catch(() => {})
      return
    }
    this.publish(
      this.items.map((item) =>
        item.key === key
          ? { ...item, status: "ready", attachmentId: stored.id }
          : item
      )
    )
  }

  /** Retains a failed upload's tile with its actionable message; removed tiles stay removed. */
  private fail(key: string, error: unknown): void {
    this.abandoned.delete(key)
    this.publish(
      this.items.map((item) =>
        item.key === key
          ? {
              ...item,
              status: "failed",
              error: error instanceof Error ? error.message : undefined,
            }
          : item
      )
    )
  }

  /** Removes one staged file, revoking its preview and discarding its current or future server row. */
  remove = (key: string): void => {
    this.releaseItems(
      this.items.filter((item) => item.key === key),
      true
    )
    this.publish(this.items.filter((item) => item.key !== key))
  }

  /** Empties a sent draft, releasing previews while retaining server rows claimed by the turn. */
  release = (): void => {
    this.releaseItems(this.items, false)
    this.publish([])
  }

  /** Abandons a draft and deletes its unsent uploads without waiting for best-effort cleanup. */
  discard = (): void => {
    this.releaseItems(this.items, true)
    this.publish([])
  }

  /** Releases previews on unmount; ready server rows retain the existing server cleanup policy. */
  dispose(): void {
    this.releaseItems(this.items, false)
    this.items = this.items.map((item) => ({ ...item, previewUrl: undefined }))
  }
}
