import type { ValidationRequest, ValidationRun } from "./contracts.generated"
import { decodeArchitecture, identityDigest } from "./decoder"
import { ArchitectureApiError } from "./http"

export interface ValidationSubmissionState {
  key: string | null
  request: ValidationRequest | null
  run: ValidationRun | null
  pending: boolean
  uncertain: boolean
  error: string | null
}

/** Owns explicit submission for one editor session; uncertain retries reuse the same actor/request deduplication identity and cleanup never cancels durable work. */
export class ValidationSubmission {
  private state: Readonly<ValidationSubmissionState> = Object.freeze({
    key: null,
    request: null,
    run: null,
    pending: false,
    uncertain: false,
    error: null,
  })
  private readonly listeners = new Set<() => void>()
  private controller: AbortController | null = null
  private disposed = false

  /** Captures one family/resource-specific queue transport without starting work. */
  constructor(
    private readonly enqueue: (
      request: ValidationRequest,
      signal: AbortSignal
    ) => Promise<ValidationRun>
  ) {}

  /** Returns one stable submission snapshot for external-store subscribers. */
  getSnapshot = (): Readonly<ValidationSubmissionState> => this.state

  /** Subscribes the current editor; listener cleanup has no durable execution effects. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** Publishes a complete submission transition only while this editor owns the actual session. */
  private publish(next: ValidationSubmissionState): void {
    if (this.disposed) return
    this.state = Object.freeze(next)
    this.listeners.forEach((listener) => listener())
  }

  /** Queues exact interpreted configuration, reusing an uncertain request only for unchanged input; rejects foreign acknowledgements and suppresses disposed replies. */
  async submit(
    key: string,
    value: Omit<ValidationRequest, "request_id">
  ): Promise<void> {
    if (this.disposed || this.controller) return
    const previous = this.state
    const request = decodeArchitecture("ValidationRequest", {
      ...value,
      request_id: crypto.randomUUID(),
    })
    if (
      previous.uncertain &&
      previous.key === key &&
      previous.request &&
      identityDigest({ ...previous.request, request_id: null }) ===
        identityDigest({ ...request, request_id: null })
    )
      request.request_id = previous.request.request_id
    const controller = new AbortController()
    this.controller = controller
    this.publish({
      key,
      request,
      run: null,
      pending: true,
      uncertain: false,
      error: null,
    })
    try {
      const run = decodeArchitecture(
        "ValidationRun",
        await this.enqueue(request, controller.signal)
      )
      if (identityDigest(run.config) !== identityDigest(request))
        throw new Error(
          "Queued validation differs from the submitted saved configuration."
        )
      this.publish({
        key,
        request,
        run,
        pending: false,
        uncertain: false,
        error: null,
      })
    } catch (error) {
      const rejected =
        error instanceof ArchitectureApiError &&
        error.status >= 400 &&
        error.status < 500
      this.publish({
        key,
        request,
        run: null,
        pending: false,
        uncertain: !rejected,
        error: rejected
          ? error.message.slice(0, 2000)
          : "Submission outcome is unknown. Retry the unchanged configuration to recover the same queued request, or check the queue.",
      })
    } finally {
      if (this.controller === controller) this.controller = null
    }
  }

  /** Aborts only this editor's transport and ignores late acknowledgements; the engine retains any committed queue request. */
  dispose(): void {
    this.disposed = true
    this.controller?.abort()
    this.listeners.clear()
  }
}
