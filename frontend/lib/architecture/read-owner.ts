export interface ResourceIdentity {
  token: string
  resource: string
}
export interface DraftIdentity extends ResourceIdentity {
  draftVersion: number
  contentHash: string
}
export interface ConfigurationIdentity extends DraftIdentity {
  configurationHash: string
}
export interface RunIdentity extends ResourceIdentity {
  runId: string
}
/** Owns cancellable reads for one mounted resource and suppresses delayed completions after replacement. */
export class ArchitectureReadOwner {
  private controller: AbortController | null = null
  private generation = 0
  private disposed = false
  /** Captures an authenticated resource identity for this owner's lifetime. */
  constructor(public readonly identity: ResourceIdentity) {}
  /** Aborts the previous read and returns only a result still owned by this resource generation. */
  async read<T>(
    load: (signal: AbortSignal) => Promise<T>
  ): Promise<T | undefined> {
    if (this.disposed) return undefined
    this.controller?.abort()
    const controller = new AbortController()
    this.controller = controller
    const generation = ++this.generation
    try {
      const value = await load(controller.signal)
      return !this.disposed && generation === this.generation
        ? value
        : undefined
    } catch (error) {
      if (
        this.disposed ||
        generation !== this.generation ||
        controller.signal.aborted
      )
        return undefined
      throw error
    }
  }
  /** Ends ownership and prevents any pending result from entering a replacement resource. */
  dispose(): void {
    this.disposed = true
    this.cancel()
  }
  /** Aborts a superseded selection without disposing this authenticated resource owner. */
  cancel(): void {
    this.generation++
    this.controller?.abort()
    this.controller = null
  }
}
