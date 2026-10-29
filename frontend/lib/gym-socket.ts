import { WS_URL, getToken, handleSessionExpired } from "./api"

// The backend's exact error frame for a bad/expired token (api/src/ws.rs).
const AUTH_FAILED_MESSAGE = "authentication failed"

// Drives a whole gym run over one socket, and reports it as it happens.
//
//   client → { token }                              (first frame, auth)
//   server → { type: "ready" }
//   client → { temperature?, top_k?, top_p?, context_multiplier?, seed? }
//   server → { type: "episode_start", episode_id, index, task_type, prompt }
//   server → { type: "token", token, episode_id }   (decoded text, repeated)
//   server → { type: "tool_call", tool, argument, episode_id }
//   server → { type: "tool_result", tool, argument, body, episode_id }
//   server → { type: "round_done", text, usage, episode_id }
//   server → { type: "episode_done", episode_id, score, correct, answer, ground_truth }
//   server → { type: "run_done", score, correct, graded, episode_count }
//   server → { type: "error", error }               (at any point)
//
// The same vocabulary a chat turn streams, with two differences that matter.
// The LLM engine's `done` arrives as **`round_done`**: an episode takes a round per
// command, and a client counting `done` would see an episode end every time the
// model ran `ls`. And every frame carries the `episode_id` it belongs to, so a
// run of twenty tasks is one socket rather than twenty.
//
// The model and the tasks are the run's, fixed when it was created — this
// socket carries only the sampling, and closing it abandons the run.

export interface GymSampling {
  temperature: number
  topK: number
  topP: number
  /**
   * Stretches the window past what the model trained at, via NTK-aware RoPE.
   * It matters more here than in chat: a task plus its system block is ~70
   * tokens before the shell has answered anything, and half the window is
   * reserved for the reply — so a model trained under 256 cannot hold a task
   * at all unless this is above 1.
   */
  contextMultiplier: number
}

export interface GymEpisodeStart {
  episodeId: string
  index: number
  taskType: string
  prompt: string
}

export interface GymToolCall {
  episodeId: string
  tool: string
  argument: string
}

export interface GymToolResult extends GymToolCall {
  /** The result in the marker vocabulary — a `<|codeblock|>`, or a `<|warn|>`. */
  body: string
}

export interface GymVerdict {
  episodeId: string
  score: number | null
  correct: boolean
  answer: string | null
  groundTruth: string | null
}

export interface GymRunTotals {
  score: number | null
  correct: number
  graded: number
  episodeCount: number
}

export interface GymRunHandlers {
  onEpisodeStart: (start: GymEpisodeStart) => void
  onToken: (episodeId: string, token: string) => void
  onToolCall: (call: GymToolCall) => void
  onToolResult: (result: GymToolResult) => void
  /** One per generation round. An episode has as many as it ran commands. */
  onRoundDone: (episodeId: string, elapsedSecs: number | null) => void
  onEpisodeDone: (verdict: GymVerdict) => void
  onRunDone: (totals: GymRunTotals) => void
  onError: (message: string) => void
}

/** Opens the stream and returns a cancel function. */
export function streamGymRun(
  runId: string,
  sampling: GymSampling,
  handlers: GymRunHandlers
): () => void {
  const ws = new WebSocket(`${WS_URL}/ws/gyms/runs/${runId}`)
  const token = getToken()
  let settled = false

  const settle = (fn: () => void) => {
    if (settled) return
    settled = true
    fn()
    try {
      ws.close()
    } catch {
      // already closing
    }
  }

  ws.onopen = () => {
    ws.send(JSON.stringify({ token }))
  }

  ws.onmessage = (event) => {
    let msg: {
      type?: string
      episode_id?: string
      index?: number
      task_type?: string
      prompt?: string
      token?: string
      tool?: string
      argument?: string
      body?: string
      score?: number | null
      correct?: number | boolean
      graded?: number
      episode_count?: number
      answer?: string | null
      ground_truth?: string | null
      error?: string
      usage?: { elapsed_secs?: number }
    }
    try {
      msg = JSON.parse(event.data)
    } catch {
      return
    }

    const episodeId = msg.episode_id ?? ""

    switch (msg.type) {
      case "ready":
        // Each knob only when it is a real number, for the reason the chat
        // socket omits them: a NaN out of a corrupted setting would serialize
        // to `null` and be rejected, turning a bad setting into a failed run.
        ws.send(
          JSON.stringify({
            ...(Number.isFinite(sampling.temperature)
              ? { temperature: sampling.temperature }
              : {}),
            ...(Number.isFinite(sampling.topK)
              ? { top_k: Math.round(sampling.topK) }
              : {}),
            ...(Number.isFinite(sampling.topP) ? { top_p: sampling.topP } : {}),
            ...(Number.isFinite(sampling.contextMultiplier)
              ? { context_multiplier: sampling.contextMultiplier }
              : {}),
          })
        )
        break
      case "episode_start":
        handlers.onEpisodeStart({
          episodeId,
          index: msg.index ?? 0,
          taskType: msg.task_type ?? "",
          prompt: msg.prompt ?? "",
        })
        break
      case "token":
        handlers.onToken(episodeId, msg.token ?? "")
        break
      case "tool_call":
        handlers.onToolCall({
          episodeId,
          tool: msg.tool ?? "",
          argument: msg.argument ?? "",
        })
        break
      case "tool_result":
        handlers.onToolResult({
          episodeId,
          tool: msg.tool ?? "",
          argument: msg.argument ?? "",
          body: msg.body ?? "",
        })
        break
      case "round_done":
        handlers.onRoundDone(
          episodeId,
          typeof msg.usage?.elapsed_secs === "number" ? msg.usage.elapsed_secs : null
        )
        break
      case "episode_done":
        handlers.onEpisodeDone({
          episodeId,
          score: msg.score ?? null,
          correct: msg.correct === true,
          answer: msg.answer ?? null,
          groundTruth: msg.ground_truth ?? null,
        })
        break
      case "run_done":
        settle(() =>
          handlers.onRunDone({
            score: msg.score ?? null,
            correct: typeof msg.correct === "number" ? msg.correct : 0,
            graded: msg.graded ?? 0,
            episodeCount: msg.episode_count ?? 0,
          })
        )
        break
      case "error":
        if (msg.error === AUTH_FAILED_MESSAGE) {
          settle(() => handleSessionExpired(token))
          break
        }
        settle(() => handlers.onError(msg.error || "Something went wrong."))
        break
    }
  }

  ws.onerror = () => settle(() => handlers.onError("Connection error."))
  ws.onclose = () => settle(() => handlers.onError("Connection closed."))

  // Closing is the cancel. Episodes already graded are stored; the rest are
  // never run, and the run is marked abandoned on the way out.
  return () => settle(() => {})
}
