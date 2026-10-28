import { sha256 } from "@noble/hashes/sha256"
import { bytesToHex } from "@noble/hashes/utils"
import type { DraftSaved } from "./contracts.generated"
import { ArchitectureApiError } from "./http"

export type DraftSnapshot = Readonly<{
  saved: Readonly<DraftSaved>
  savedText: string
  text: string
  editRevision: number
  saving: boolean
  conflict: boolean
  error: string | null
}>

export type PersistDraft = (
  version: number,
  text: string,
  signal: AbortSignal
) => Promise<DraftSaved>

/** Hashes exact UTF-8 editor bytes without parsing or normalizing invalid Python/YAML drafts. */
export function draftDigest(text: string): string {
  return bytesToHex(sha256(new TextEncoder().encode(text)))
}

/** Owns one loaded draft and its optimistic save lifetime, preserving edits typed during an earlier save. */
export class ArchitectureDraft {
  private state: DraftSnapshot
  private readonly listeners = new Set<() => void>()
  private controller: AbortController | null = null
  private disposed = false

  /** Binds exact loaded bytes to their engine identity and one session/resource-specific persistence operation. */
  constructor(
    saved: DraftSaved,
    text: string,
    private readonly persist: PersistDraft
  ) {
    if (saved.content_hash !== draftDigest(text))
      throw new Error("Draft bytes differ from their saved identity.")
    this.state = Object.freeze({
      saved: Object.freeze({ ...saved }),
      savedText: text,
      text,
      editRevision: 0,
      saving: false,
      conflict: false,
      error: null,
    })
  }

  /** Returns the stable immutable snapshot used by React's external-store subscription. */
  getSnapshot = (): DraftSnapshot => this.state

  /** Subscribes one mounted observer; observer detachment alone does not imply save rollback. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** Publishes a complete state transition without transport effects inside React updaters. */
  private publish(next: DraftSnapshot): void {
    if (this.disposed) return
    this.state = Object.freeze({
      ...next,
      saved: Object.freeze({ ...next.saved }),
    })
    this.listeners.forEach((listener) => listener())
  }

  /** Edits local bytes and invalidates the configuration generation, retaining a known concurrent-version conflict. */
  edit(text: string): void {
    if (this.disposed || text === this.state.text) return
    this.publish({
      ...this.state,
      text,
      editRevision: this.state.editRevision + 1,
      error: this.state.conflict ? this.state.error : null,
    })
  }

  /** Saves captured bytes/version while preserving newer typing; stale server versions never retry as an overwrite. */
  async save(): Promise<void> {
    if (
      this.disposed ||
      this.state.saving ||
      this.state.conflict ||
      this.state.text === this.state.savedText
    )
      return
    const captured = this.state
    if (new TextEncoder().encode(captured.text).length > 262144) {
      this.publish({
        ...this.state,
        error: "Draft exceeds the 256 KiB UTF-8 limit.",
      })
      return
    }
    const controller = new AbortController()
    this.controller = controller
    this.publish({ ...this.state, saving: true, error: null })
    try {
      const saved = await this.persist(
        captured.saved.draft_version,
        captured.text,
        controller.signal
      )
      if (this.disposed) return
      if (
        saved.resource_id !== captured.saved.resource_id ||
        saved.draft_version !== captured.saved.draft_version + 1 ||
        saved.content_hash !== draftDigest(captured.text)
      )
        throw new Error(
          "Save response differs from the submitted draft. Reload to check the server version."
        )
      this.publish({
        ...this.state,
        saved,
        savedText: captured.text,
        saving: false,
        error: null,
      })
    } catch (error) {
      if (this.disposed) return
      const conflict =
        error instanceof ArchitectureApiError && error.status === 409
      this.publish({
        ...this.state,
        saving: false,
        conflict,
        error: conflict
          ? "This draft changed on the server. Copy your local edits before reloading the server version."
          : error instanceof Error
            ? error.message
            : "Draft did not save.",
      })
    } finally {
      if (this.controller === controller) this.controller = null
    }
  }

  /** Ends this session/resource owner, aborting transport and suppressing late acknowledgements without claiming remote rollback. */
  dispose(): void {
    this.disposed = true
    this.controller?.abort()
    this.controller = null
    this.listeners.clear()
  }
}
