"use client"

import * as React from "react"

/**
 * What the recogniser has heard so far. The two halves are not the same kind of
 * text: `final` is settled and will not change, `interim` is the tail it is
 * still revising and is rewritten on nearly every event. Rendering them
 * differently is the whole reason they are kept apart.
 */
export interface SpeechTranscript {
  final: string
  interim: string
}

export interface SpeechRecognitionState {
  /** False on Firefox, and during SSR and the hydrating render. */
  supported: boolean
  listening: boolean
  transcript: SpeechTranscript
  /**
   * The recogniser is running entirely on this machine (Chrome 138+ with the
   * language pack already installed) — the only configuration in which the
   * audio does not leave the browser.
   */
  onDevice: boolean
  error: string | null
  start: () => void
  stop: () => void
}

/** A session ending faster than this heard nothing and is a failure, not a
 *  pause: the browser rejected the start rather than timing out on silence. */
const SESSION_TOO_SHORT_MS = 250

/** How many failures in a row before we stop restarting. Without a bound the
 *  `onend` → `start()` chain is an unbounded loop against a browser that has
 *  already said no. */
const MAX_CONSECUTIVE_FAILURES = 3

/** `error` codes worth telling someone about. `no-speech` and `aborted` are
 *  ordinary punctuation in a long dictation and are deliberately absent. */
const ERROR_MESSAGES: Record<string, string> = {
  "not-allowed": "Microphone access was blocked.",
  "service-not-allowed": "This browser won't allow dictation on this page.",
  "audio-capture": "No microphone was found.",
  network: "Dictation lost its connection.",
}

const RESTART_EXHAUSTED = "Dictation kept stopping. Try again in a moment."

interface SpeechAlternative {
  readonly transcript: string
}

interface SpeechResult {
  readonly isFinal: boolean
  readonly [index: number]: SpeechAlternative | undefined
}

interface SpeechResultList {
  readonly length: number
  readonly [index: number]: SpeechResult | undefined
}

interface SpeechResultEvent {
  readonly resultIndex: number
  readonly results: SpeechResultList
}

interface SpeechErrorEvent {
  readonly error: string
}

interface SpeechRecognitionInstance {
  continuous: boolean
  interimResults: boolean
  lang: string
  processLocally?: boolean
  start(): void
  stop(): void
  abort(): void
  onresult: ((event: SpeechResultEvent) => void) | null
  onerror: ((event: SpeechErrorEvent) => void) | null
  onend: (() => void) | null
}

interface SpeechRecognitionCtor {
  new (): SpeechRecognitionInstance
  /** Chrome 138+. Resolves "available" | "downloadable" | "unavailable". */
  available?: (options: {
    langs: string[]
    processLocally: boolean
  }) => Promise<string>
}

/**
 * Read off `window` rather than declared as a global: the vendor-prefixed name
 * is the only one Safari has, and every TypeScript release that adds its own
 * declaration for the standard one would collide with ours.
 */
function recognitionCtor(): SpeechRecognitionCtor | null {
  if (typeof window === "undefined") return null
  const w = window as unknown as {
    SpeechRecognition?: SpeechRecognitionCtor
    webkitSpeechRecognition?: SpeechRecognitionCtor
  }
  return w.SpeechRecognition ?? w.webkitSpeechRecognition ?? null
}

/** `SpeechRecognition` either exists when the script loads or never does. */
const subscribeToNothing = () => () => {}

function dictationLanguage(): string {
  return typeof navigator === "undefined" ? "en-US" : navigator.language
}

/** Results carry no leading or trailing space of their own, so concatenating
 *  them directly joins the last word of one chunk to the first of the next. */
function appendChunk(existing: string, chunk: string): string {
  const next = chunk.trim()
  if (!next) return existing
  if (!existing) return next
  return `${existing} ${next}`
}

/**
 * Live dictation through the browser's own recogniser — free, no key, no
 * bundle. Chrome and Edge stream the audio to a Google service unless
 * `onDevice` is true; Safari uses its own. See
 * `docs/decisions/voice-input-uses-the-browser.md`.
 */
export function useSpeechRecognition(): SpeechRecognitionState {
  // A capability of the page rather than a thing that changes: subscribed to
  // rather than set in an effect, which is also what keeps the hydrating render
  // from disagreeing with the server about whether the button exists.
  const supported = React.useSyncExternalStore(
    subscribeToNothing,
    () => recognitionCtor() !== null,
    () => false
  )
  const [listening, setListening] = React.useState(false)
  const [onDevice, setOnDevice] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [transcript, setTranscript] = React.useState<SpeechTranscript>({
    final: "",
    interim: "",
  })

  const recognitionRef = React.useRef<SpeechRecognitionInstance | null>(null)
  // What the person wants, as opposed to whether a session happens to be
  // running: the browser ends sessions on its own and `onend` needs to know
  // whether that ending was asked for.
  const wantListeningRef = React.useRef(false)
  const finalRef = React.useRef("")
  const sessionStartedAtRef = React.useRef(0)
  const failuresRef = React.useRef(0)
  const onDeviceRef = React.useRef(false)

  React.useEffect(() => {
    const Ctor = recognitionCtor()
    if (!Ctor) return

    // Only an already-installed pack is used. "downloadable" would mean
    // starting a several-hundred-megabyte fetch because someone tapped a
    // microphone, which is not a thing to do behind their back.
    let cancelled = false
    void Ctor.available?.({
      langs: [dictationLanguage()],
      processLocally: true,
    })
      .then((status) => {
        if (cancelled || status !== "available") return
        onDeviceRef.current = true
        setOnDevice(true)
      })
      .catch(() => {})
    return () => {
      cancelled = true
    }
  }, [])

  const teardown = React.useCallback(() => {
    const recognition = recognitionRef.current
    recognitionRef.current = null
    if (recognition) {
      recognition.onresult = null
      recognition.onerror = null
      recognition.onend = null
      recognition.abort()
    }
    setListening(false)
  }, [])

  const stop = React.useCallback(() => {
    wantListeningRef.current = false
    const recognition = recognitionRef.current
    if (!recognition) {
      setListening(false)
      return
    }
    // `stop` rather than `abort`: it delivers whatever it was mid-way through
    // recognising as a final result, so the last word said is not the one lost.
    recognition.stop()
  }, [])

  const start = React.useCallback(() => {
    if (recognitionRef.current) return
    const Ctor = recognitionCtor()
    if (!Ctor) return

    const recognition = new Ctor()
    recognition.continuous = true
    recognition.interimResults = true
    recognition.lang = dictationLanguage()
    if (onDeviceRef.current) recognition.processLocally = true

    recognition.onresult = (event) => {
      let interim = ""
      // Bounded by the result list; `resultIndex` is where this event's news
      // begins, so everything before it has already been folded into `final`.
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const result = event.results[i]
        const text = result?.[0]?.transcript ?? ""
        if (result?.isFinal)
          finalRef.current = appendChunk(finalRef.current, text)
        else interim = appendChunk(interim, text)
      }
      setTranscript({ final: finalRef.current, interim })
    }

    recognition.onerror = (event) => {
      const message = ERROR_MESSAGES[event.error]
      if (!message) return
      setError(message)
      wantListeningRef.current = false
    }

    recognition.onend = () => {
      const failed =
        Date.now() - sessionStartedAtRef.current < SESSION_TOO_SHORT_MS
      failuresRef.current = failed ? failuresRef.current + 1 : 0

      if (!wantListeningRef.current) {
        teardown()
        return
      }
      if (failuresRef.current >= MAX_CONSECUTIVE_FAILURES) {
        setError(RESTART_EXHAUSTED)
        wantListeningRef.current = false
        teardown()
        return
      }
      // A session ends by itself after a stretch of silence even with
      // `continuous` set, so dictating through a pause means starting it again.
      sessionStartedAtRef.current = Date.now()
      try {
        recognition.start()
      } catch {
        teardown()
      }
    }

    finalRef.current = ""
    failuresRef.current = 0
    sessionStartedAtRef.current = Date.now()
    wantListeningRef.current = true
    recognitionRef.current = recognition
    setTranscript({ final: "", interim: "" })
    setError(null)

    try {
      recognition.start()
      setListening(true)
    } catch {
      teardown()
    }
  }, [teardown])

  React.useEffect(() => teardown, [teardown])

  return { supported, listening, transcript, onDevice, error, start, stop }
}
