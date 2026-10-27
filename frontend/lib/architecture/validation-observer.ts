import type {
  Diagnostic,
  StageOutcome,
  ValidationEvent,
  ValidationReport,
  ValidationRun,
} from "./contracts.generated"
import { decodeArchitecture, identityDigest } from "./decoder"
import { ArchitectureApiError } from "./http"
import { requireValidationReport } from "../model-validation"

export interface ValidationTransport {
  run: (signal: AbortSignal) => Promise<ValidationRun>
  report: (signal: AbortSignal) => Promise<ValidationReport>
  cancel: (signal: AbortSignal) => Promise<void>
  retry: (signal: AbortSignal) => Promise<ValidationRun>
  remove: (signal: AbortSignal) => Promise<void>
}

export interface ValidationObservation {
  run: ValidationRun | null
  report: ValidationReport | null
  stages: readonly StageOutcome[]
  diagnostics: readonly Diagnostic[]
  sequence: number
  loading: boolean
  controlling: boolean
  cancelRequested: boolean
  retryUncertain: boolean
  error: string | null
  actionError: string | null
}

const TERMINAL = new Set(["done", "failed", "cancelled"])

/** Owns one session/run observer, bounded stage replay, serialized REST reconciliation and explicit controls without owning the durable worker. */
export class ValidationObserver {
  private state: Readonly<ValidationObservation> = {
    run: null,
    report: null,
    stages: [],
    diagnostics: [],
    sequence: -1,
    loading: true,
    controlling: false,
    cancelRequested: false,
    retryUncertain: false,
    error: null,
    actionError: null,
  }
  private readonly listeners = new Set<() => void>()
  private disposed = false
  private pending: Promise<void> | null = null
  private read: AbortController | null = null
  private control: AbortController | null = null
  private timer: ReturnType<typeof setTimeout> | null = null

  /** Binds immutable run identity and captured-session transport; no network or timer starts until refresh is explicitly called. */
  constructor(
    readonly id: string,
    private readonly transport: ValidationTransport
  ) {}

  /** Returns the stable immutable observation snapshot for React subscribers. */
  getSnapshot = (): Readonly<ValidationObservation> => this.state

  /** Subscribes to current observation state; unsubscribing alone never cancels the durable execution. */
  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => {
      this.listeners.delete(listener)
    }
  }

  /** Publishes one complete owned state transition, ignoring any completion after disposal. */
  private publish(next: ValidationObservation): void {
    if (this.disposed) return
    this.state = Object.freeze(next)
    this.listeners.forEach((listener) => listener())
  }

  /** Coalesces concurrent reads and polls only after a completed nonterminal read; errors require explicit observation retry. */
  refresh(): Promise<void> {
    if (this.disposed) return Promise.resolve()
    if (this.pending) return this.pending
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
    const controller = new AbortController()
    this.read = controller
    this.publish({
      ...this.state,
      loading: this.state.run === null,
      error: null,
    })
    const pending = this.reconcile(controller).finally(() => {
      if (this.pending !== pending) return
      this.pending = null
      this.read = null
      if (
        !this.disposed &&
        this.state.run &&
        !TERMINAL.has(this.state.run.status) &&
        !this.state.error
      )
        this.timer = setTimeout(() => {
          void this.refresh()
        }, 2000)
    })
    this.pending = pending
    return pending
  }

  /** Reads actual lifecycle first and verified publication only for a done job; malformed/stale identities cannot replace this run. */
  private async reconcile(controller: AbortController): Promise<void> {
    try {
      const run = decodeArchitecture(
        "ValidationRun",
        await this.transport.run(controller.signal)
      )
      if (this.disposed) return
      if (run.run_id !== this.id)
        throw new Error("Validation response belongs to another execution.")
      if (
        this.state.run &&
        identityDigest(this.state.run.config) !== identityDigest(run.config)
      )
        throw new Error("Validation execution changed its immutable request.")
      this.publish({ ...this.state, run, loading: false })
      if (run.status === "done" && !this.state.report) {
        const report = decodeArchitecture(
          "ValidationReport",
          await this.transport.report(controller.signal)
        )
        if (this.disposed) return
        requireValidationReport(run, report)
        this.publish({
          ...this.state,
          report,
          stages: report.stages,
          diagnostics: report.diagnostics,
        })
      }
    } catch (error) {
      if (!this.disposed)
        this.publish({
          ...this.state,
          loading: false,
          error:
            error instanceof Error
              ? error.message.slice(0, 2000)
              : "Validation observation failed.",
        })
    }
  }

  /** Applies only this run's new typed stage/terminal event and reconciles terminal lifecycle through REST instead of trusting a socket verdict. */
  event(value: unknown): void {
    if (this.disposed) return
    const event: ValidationEvent = decodeArchitecture("ValidationEvent", value)
    if (event.run_id !== this.id)
      throw new Error("Validation event belongs to another execution.")
    if (this.state.report) return
    if (event.sequence <= this.state.sequence) return
    const stages = event.stage
      ? [
          ...this.state.stages.filter(
            (item) => item.stage !== event.stage!.stage
          ),
          event.stage,
        ].slice(-16)
      : this.state.stages
    const diagnostics = event.diagnostic
      ? [...this.state.diagnostics, event.diagnostic].slice(-64)
      : this.state.diagnostics
    this.publish({
      ...this.state,
      sequence: event.sequence,
      stages,
      diagnostics,
    })
    if (event.type !== "validation_stage") void this.refresh()
  }

  /** Requests cooperative cancellation once; only authoritative lifecycle determines when execution has actually stopped. */
  async cancel(): Promise<void> {
    if (
      this.disposed ||
      this.control ||
      !this.state.run ||
      TERMINAL.has(this.state.run.status) ||
      this.state.cancelRequested
    )
      return
    const controller = new AbortController()
    this.control = controller
    this.publish({ ...this.state, controlling: true, actionError: null })
    try {
      await this.transport.cancel(controller.signal)
      this.publish({ ...this.state, cancelRequested: true })
    } catch (error) {
      this.publish({
        ...this.state,
        actionError:
          error instanceof Error
            ? error.message.slice(0, 2000)
            : "Cancellation request failed.",
      })
    } finally {
      if (this.control === controller) this.control = null
      this.publish({ ...this.state, controlling: false })
      if (!this.disposed) await this.refresh()
    }
  }

  /** Retries the original saved snapshot with a new identity; ambiguous outcomes disable duplicate retry until the user checks the queue. */
  async retry(): Promise<ValidationRun | undefined> {
    if (
      this.disposed ||
      this.control ||
      !this.state.run ||
      !TERMINAL.has(this.state.run.status) ||
      this.state.retryUncertain
    )
      return
    const previous = this.state.run
    const controller = new AbortController()
    this.control = controller
    this.publish({ ...this.state, controlling: true, actionError: null })
    try {
      const run = decodeArchitecture(
        "ValidationRun",
        await this.transport.retry(controller.signal)
      )
      const { request_id: previousRequest, ...before } = previous.config
      const { request_id: nextRequest, ...after } = run.config
      if (
        run.run_id === this.id ||
        previousRequest === nextRequest ||
        identityDigest(before) !== identityDigest(after)
      )
        throw new Error(
          "Retried validation differs from its immutable saved snapshot."
        )
      return this.disposed ? undefined : run
    } catch (error) {
      const rejected =
        error instanceof ArchitectureApiError &&
        error.status >= 400 &&
        error.status < 500
      this.publish({
        ...this.state,
        retryUncertain: !rejected,
        actionError: rejected
          ? error.message
          : "Retry outcome is unknown. Check the queue before retrying again.",
      })
      return undefined
    } finally {
      if (this.control === controller) this.control = null
      this.publish({ ...this.state, controlling: false })
    }
  }

  /** Deletes only a terminal history row on explicit user action; an unsuccessful or disposed request cannot navigate the replacement session. */
  async remove(): Promise<boolean> {
    if (
      this.disposed ||
      this.control ||
      !this.state.run ||
      !TERMINAL.has(this.state.run.status)
    )
      return false
    const controller = new AbortController()
    this.control = controller
    this.publish({ ...this.state, controlling: true, actionError: null })
    try {
      await this.transport.remove(controller.signal)
      return !this.disposed
    } catch (error) {
      this.publish({
        ...this.state,
        actionError:
          error instanceof Error
            ? error.message.slice(0, 2000)
            : "Validation history could not be deleted.",
      })
      return false
    } finally {
      if (this.control === controller) this.control = null
      this.publish({ ...this.state, controlling: false })
    }
  }

  /** Stops only this observer's timer/transports and suppresses late data; never invokes the durable cancel operation. */
  dispose(): void {
    this.disposed = true
    if (this.timer !== null) clearTimeout(this.timer)
    this.timer = null
    this.read?.abort()
    this.control?.abort()
    this.listeners.clear()
  }
}
