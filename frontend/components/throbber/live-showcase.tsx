"use client"

import * as React from "react"
import { ArtStream } from "@/lib/live-throbber/stream"
import { VISUAL_TARGETS_PER_SECOND } from "@/lib/live-throbber/sampler"
import { useSettings } from "@/components/settings/settings-provider"
import { ThinkingLabel } from "@/components/chat/thinking-label"
import { TokenFlow } from "@/components/chat/token-flow"
import { TurnClock } from "@/components/chat/turn-clock"
import { Button } from "@/components/ui/button"
import { Switch } from "@/components/ui/switch"
import { Panel } from "@/components/ui/panel"
import { Throbber, type ThrobberMode } from "@/components/ui/throbber"
import { LiveArt, type ArtStatus } from "./live-art"

// These ids are synthetic: the fixture exercises the wire contract without a model.
const TOKENS = [
  0, 218, 947, 32, 416, 9, 1206, 51, 28, 703, 19, 467, 82, 300, 11, 17,
]
const TEXT = [
  "The",
  " shape",
  " of",
  " a",
  " thought",
  " changes",
  " as",
  " each",
  " token",
  " arrives",
  ".",
  " Soft",
  " contours",
  " become",
  " new",
  " colours",
]
const MODES: ThrobberMode[] = [
  "idle",
  "loading",
  "thinking",
  "generating",
  "tool",
  "training",
  "benchmarking",
  "tokenizing",
]
const FIXTURE_ID = "slate:live-throbber:fixture:v1"
const STREAM_SPEEDS = [
  { label: "Slow", rate: 1 },
  { label: "Steady", rate: 24 },
  { label: "Fast", rate: 240 },
]
const TICK_MS = 90

export function LiveShowcase() {
  const [stream] = React.useState(() => new ArtStream(FIXTURE_ID))
  const [mode, setMode] = React.useState<ThrobberMode>("generating")
  const [paused, setPaused] = React.useState(false)
  const [useIds, setUseIds] = React.useState(true)
  const [speed, setSpeed] = React.useState(24)
  const [count, setCount] = React.useState(0)
  const [status, setStatus] = React.useState<ArtStatus>("starting")
  const { settings, setSetting } = useSettings()

  const advance = React.useCallback(
    (amount: number) => {
      for (let i = 0; i < amount; i++) {
        const index = stream.snapshot.count % TOKENS.length
        stream.push(TEXT[index], useIds ? TOKENS[index] : undefined)
      }
      setCount(stream.snapshot.count)
    },
    [stream, useIds]
  )

  React.useEffect(() => {
    if (paused || (mode !== "generating" && mode !== "thinking")) return
    let remainder = 0
    const timer = setInterval(() => {
      remainder += (speed * TICK_MS) / 1000
      const amount = Math.floor(remainder)
      remainder -= amount
      if (amount) advance(amount)
    }, TICK_MS)
    return () => clearInterval(timer)
  }, [paused, mode, advance, speed])

  const replay = () => {
    stream.reset(FIXTURE_ID)
    setCount(0)
  }

  return (
    <Panel
      variant="open"
      title="Live token art"
      aside={
        status === "wgpu"
          ? "wgpu · WebAssembly"
          : status === "starting"
            ? "Starting renderer…"
            : "Classic fallback · WebGPU unavailable"
      }
    >
      <div className="grid items-center gap-8 border-y border-border/60 py-8 sm:grid-cols-2">
        <div className="flex justify-center" data-live-preview>
          <LiveArt
            mode={mode}
            stream={stream}
            size={192}
            onStatus={setStatus}
          />
        </div>
        <div className="flex min-w-0 flex-col gap-5">
          <div className="flex items-center justify-between gap-4">
            <label
              htmlFor="live-throbber-setting"
              className="text-sm font-medium"
            >
              Use live throbber
            </label>
            <Switch
              id="live-throbber-setting"
              checked={settings.useLiveThrobber}
              onCheckedChange={(value) => setSetting("useLiveThrobber", value)}
            />
          </div>
          <div className="flex items-center gap-3 text-sm" data-live-status-row>
            <Throbber live stream={stream} mode={mode} />
            <TurnClock />
            <span aria-hidden className="-mx-1 text-border">
              |
            </span>
            <TokenFlow direction="out" count={count} />
            {mode === "thinking" && <ThinkingLabel />}
          </div>
          <p className="font-mono text-xs text-muted-foreground" data-art-seed>
            {useIds ? "Synthetic token IDs" : "Text fallback"} ·{" "}
            {stream.snapshot.seed.toString(16).padStart(8, "0")}
          </p>
          <div className="flex flex-wrap gap-1.5">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => setPaused(!paused)}
            >
              {paused ? "Resume stream" : "Pause stream"}
            </Button>
            <Button size="sm" variant="ghost" onClick={replay}>
              Replay
            </Button>
            <Button size="sm" variant="ghost" onClick={() => advance(256)}>
              Burst 256
            </Button>
          </div>
          <div
            className="flex flex-wrap gap-1.5"
            role="group"
            aria-label="Seed source"
          >
            <Button
              size="sm"
              variant={useIds ? "secondary" : "ghost"}
              aria-pressed={useIds}
              onClick={() => setUseIds(true)}
            >
              Token IDs
            </Button>
            <Button
              size="sm"
              variant={!useIds ? "secondary" : "ghost"}
              aria-pressed={!useIds}
              onClick={() => setUseIds(false)}
            >
              Text fallback
            </Button>
          </div>
          <div
            className="flex flex-wrap items-center gap-1.5"
            role="group"
            aria-label="Stream speed"
          >
            {STREAM_SPEEDS.map(({ label, rate }) => (
              <Button
                key={rate}
                size="sm"
                variant={speed === rate ? "secondary" : "ghost"}
                aria-pressed={speed === rate}
                onClick={() => setSpeed(rate)}
              >
                {label}
              </Button>
            ))}
            <span className="ml-2 text-xs text-muted-foreground">
              {speed} tokens/s
            </span>
          </div>
          <p className="text-xs text-muted-foreground">
            Visual targets ≤ {VISUAL_TARGETS_PER_SECOND}/s · Depth 4 · 31 nodes
          </p>
        </div>
      </div>
      <div
        className="mt-4 flex flex-wrap gap-1.5"
        role="group"
        aria-label="Live throbber state"
      >
        {MODES.map((option) => (
          <Button
            key={option}
            size="sm"
            variant={mode === option ? "secondary" : "ghost"}
            aria-pressed={mode === option}
            className="capitalize"
            onClick={() => setMode(option)}
          >
            {option}
          </Button>
        ))}
      </div>
      <div className="mt-6 flex items-center gap-8 border-b border-border/60 pb-4 text-xs text-muted-foreground">
        <div className="flex items-center gap-3">
          <Throbber live={false} mode={mode} />
          Classic
        </div>
        <div className="flex items-center gap-3">
          <Throbber live mode={mode} stream={stream} />
          Live · actual size
        </div>
      </div>
    </Panel>
  )
}
