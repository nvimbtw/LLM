import { JobSubmissionError, type Job, type JobSubmissionRequest } from "./jobs"
import { parseResourceId } from "./chats"

export interface JobSubmissionState {
  key: string | null
  requestId: string | null
  pending: boolean
  uncertain: boolean
  job: Job | null
  error: string | null
}

/** Owns one session's explicit queue operations; ambiguous retries reuse their UUID and disposed observation never cancels durable work. */
export class JobSubmission {
  private state: Readonly<JobSubmissionState> = Object.freeze({
    key: null,
    requestId: null,
    pending: false,
    uncertain: false,
    job: null,
    error: null,
  })
  private readonly listeners = new Set<() => void>()
  private controller: AbortController | null = null
  private disposed = false

  /** Captures the actual bearer session used for all requests from this mounted composer/control. */
  constructor(private readonly token: string) {}

  /** Returns a stable immutable snapshot suitable for React subscriptions. */
  snapshot = (): Readonly<JobSubmissionState> => this.state

  /** Registers one observer without taking ownership of the server job's lifetime. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** Publishes complete submission state only while the original session owns this operation. */
  private publish(state: JobSubmissionState): void {
    if (this.disposed) return
    this.state = Object.freeze(state)
    this.listeners.forEach((listener) => listener())
  }

  /** Queues frozen caller inputs under their exact key; duplicates share committed identity and unknown outcomes remain safe to retry unchanged. */
  async submit(
    key: string,
    enqueue: (request: JobSubmissionRequest) => Promise<Job>
  ): Promise<Job | undefined> {
    if (this.disposed || this.controller) return undefined
    const previous = this.state
    if (previous.key === key && previous.job) return previous.job
    const requestId =
      previous.key === key && previous.uncertain && previous.requestId
        ? previous.requestId
        : crypto.randomUUID()
    const controller = new AbortController()
    this.controller = controller
    this.publish({
      key,
      requestId,
      pending: true,
      uncertain: false,
      job: null,
      error: null,
    })
    try {
      const job = await enqueue({
        token: this.token,
        requestId,
        signal: controller.signal,
      })
      parseResourceId(job.id_job)
      if (
        !["train", "rl", "benchmark", "tokenize", "model_validation"].includes(
          job.kind
        )
      )
        throw new Error("Queued execution kind is invalid.")
      if (this.disposed) return undefined
      this.publish({
        key,
        requestId,
        pending: false,
        uncertain: false,
        job,
        error: null,
      })
      return job
    } catch (error) {
      const rejected =
        error instanceof JobSubmissionError &&
        error.status >= 400 &&
        error.status < 500
      this.publish({
        key,
        requestId,
        pending: false,
        uncertain: !rejected,
        job: null,
        error: rejected
          ? error.message
          : "Submission outcome is unknown. Retry unchanged inputs to recover the same queued request, or check the queue.",
      })
      return undefined
    } finally {
      if (this.controller === controller) this.controller = null
    }
  }

  /** Aborts only the current HTTP observation and suppresses stale results/toasts after session or resource replacement. */
  dispose(): void {
    this.disposed = true
    this.controller?.abort()
    this.listeners.clear()
  }

  /** Releases an observed successful operation so a later explicit action may intentionally queue the same inputs again. */
  acknowledge(): void {
    if (this.state.pending || !this.state.job) return
    this.publish({
      key: null,
      requestId: null,
      pending: false,
      uncertain: false,
      job: null,
      error: null,
    })
  }
}
