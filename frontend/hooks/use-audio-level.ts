"use client"

import * as React from "react"

/** Low, mid and high, each 0–1. Low carries the pitch of a voice, mid the part
 *  that makes speech intelligible, high the sibilants. */
export type AudioBands = readonly [number, number, number]

/** 128 bins, so each covers roughly 190 Hz at a 48 kHz sample rate. Small
 *  enough that a frame costs nothing; the bands below are wide anyway. */
const FFT_SIZE = 256

/** The analyser's own smoothing, on top of the per-band smoothing below. */
const ANALYSER_SMOOTHING = 0.6

/** Bin ranges, half-open, in the order of `AudioBands`. Bin 0 is DC and is
 *  skipped. Above bin 40 (~7.5 kHz) there is no voice left to show. */
const BAND_BINS: readonly (readonly [number, number])[] = [
  [1, 5],
  [5, 16],
  [16, 40],
]

/** Per-band inertia. Deliberately unequal: three lines sharing one constant
 *  move as one thick line, and the point of having three is that they don't. */
const BAND_INERTIA: readonly number[] = [0.72, 0.55, 0.4]

/** Conversational speech only ever fills a fraction of the byte range, so the
 *  bands are scaled up to reach the top of the meter at a normal voice. */
const BAND_GAIN = 2.4

function readBands(
  analyser: AnalyserNode,
  bins: Uint8Array<ArrayBuffer>,
  smoothed: number[]
): void {
  analyser.getByteFrequencyData(bins)
  for (let band = 0; band < BAND_BINS.length; band++) {
    const [from, to] = BAND_BINS[band]
    let sum = 0
    for (let bin = from; bin < to; bin++) sum += bins[bin]
    const level = Math.min(1, (sum / (to - from) / 255) * BAND_GAIN)
    const inertia = BAND_INERTIA[band]
    smoothed[band] = smoothed[band] * inertia + level * (1 - inertia)
  }
}

/**
 * Microphone loudness, split into three bands, delivered once per frame.
 *
 * A second capture alongside the one the recogniser holds — `SpeechRecognition`
 * exposes no audio at all, so there is no way to draw what someone is saying
 * except to listen to it again. `onBands` is called from a `requestAnimationFrame`
 * loop and should write to the DOM directly rather than set state; sixty
 * renders a second to move three lines is not a trade worth making.
 *
 * Returns the reason the microphone could not be opened, or null.
 */
export function useAudioLevel(
  active: boolean,
  onBands: (bands: AudioBands) => void
): string | null {
  const [error, setError] = React.useState<string | null>(null)
  const onBandsRef = React.useRef(onBands)
  React.useEffect(() => {
    onBandsRef.current = onBands
  })

  React.useEffect(() => {
    if (!active) return

    let frame = 0
    let stream: MediaStream | null = null
    let context: AudioContext | null = null
    let cancelled = false
    const smoothed = [0, 0, 0]

    const release = () => {
      cancelAnimationFrame(frame)
      for (const track of stream?.getTracks() ?? []) track.stop()
      void context?.close().catch(() => {})
    }

    navigator.mediaDevices
      .getUserMedia({
        audio: { echoCancellation: true, noiseSuppression: true },
      })
      .then((opened) => {
        stream = opened
        if (cancelled) {
          release()
          return
        }
        context = new AudioContext()
        const analyser = context.createAnalyser()
        analyser.fftSize = FFT_SIZE
        analyser.smoothingTimeConstant = ANALYSER_SMOOTHING
        context.createMediaStreamSource(opened).connect(analyser)

        const bins = new Uint8Array(new ArrayBuffer(analyser.frequencyBinCount))
        const tick = () => {
          readBands(analyser, bins, smoothed)
          onBandsRef.current([smoothed[0], smoothed[1], smoothed[2]])
          frame = requestAnimationFrame(tick)
        }
        frame = requestAnimationFrame(tick)
      })
      .catch(() => {
        if (!cancelled) setError("Couldn't open the microphone.")
      })

    return () => {
      cancelled = true
      release()
    }
  }, [active])

  return error
}
