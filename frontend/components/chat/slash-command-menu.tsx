"use client"

import * as React from "react"
import { Check, Moon, Sun, SunMoon } from "lucide-react"

import {
  fuzzyMatch,
  getCommandSuggestions,
  type CommandContext,
  type SlashCommandDef,
  type SuggestionResult,
} from "@/lib/commands"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

export interface SlashCommandMenuProps {
  /** Whether the composer currently wants a menu shown at all. Kept separate
   *  from having something to render, so closing can play an exit animation
   *  on the last content it had rather than vanishing instantly. */
  open: boolean
  input: string
  selectedIndex: number
  onSelectCommand: (command: SlashCommandDef) => void
  onSelectOption: (value: string, executeImmediately?: boolean) => void
  ctx: CommandContext
}

// How long the slide/fade takes, in ms. Kept in one place since the exit
// timer and the CSS duration below have to agree.
const TRANSITION_MS = 150

/**
 * Clean, subtle highlight for matched fuzzy characters without noisy underlines.
 */
function HighlightMatched({
  text,
  query,
  className,
}: {
  text: string
  query: string
  className?: string
}) {
  if (!query) return <span className={className}>{text}</span>

  const { match, ranges } = fuzzyMatch(query, text)
  if (!match || ranges.length === 0) {
    return <span className={className}>{text}</span>
  }

  const nodes: React.ReactNode[] = []
  let lastIndex = 0

  for (let i = 0; i < ranges.length; i++) {
    const [start, end] = ranges[i]
    if (start > lastIndex) {
      nodes.push(
        <span key={`nomatch-${i}`}>{text.slice(lastIndex, start)}</span>
      )
    }
    nodes.push(
      <span key={`match-${i}`} className="font-semibold text-sidebar-primary">
        {text.slice(start, end)}
      </span>
    )
    lastIndex = end
  }

  if (lastIndex < text.length) {
    nodes.push(<span key="tail">{text.slice(lastIndex)}</span>)
  }

  return <span className={className}>{nodes}</span>
}

// The panel slides up from behind the composer and fades in; closing plays
// the same motion in reverse. `visible` toggles between the two poses —
// transform/opacity only, so it stays cheap — and the surrounding mount
// lifecycle (below) is what actually delays unmount long enough to see it.
function panelClass(visible: boolean) {
  return cn(
    "absolute bottom-full left-0 right-0 z-50 mb-2 overflow-hidden rounded-lg border border-sidebar-border bg-sidebar text-sidebar-foreground shadow-md",
    "origin-bottom transition-[opacity,transform] duration-150 ease-out",
    visible ? "opacity-100 translate-y-0" : "opacity-0 translate-y-1.5"
  )
}

const HINT_CLASS =
  "font-mono text-[11px] tracking-normal lowercase text-sidebar-foreground/50"

// Row styling shared by every option list: a hairline accent rule on the left
// that grows in on hover/active, rather than a filled highlight.
function rowClass(isActive: boolean) {
  return cn(
    "relative flex w-full items-center justify-between gap-3 rounded-md px-2.5 py-1.5 text-left text-xs transition-colors",
    "before:absolute before:left-1 before:top-1/2 before:h-0 before:w-0.5 before:-translate-y-1/2 before:rounded-full before:bg-sidebar-primary before:transition-all before:content-['']",
    isActive
      ? "bg-sidebar-accent text-sidebar-accent-foreground before:h-3.5"
      : "text-sidebar-foreground hover:bg-sidebar-accent/60 hover:before:h-2"
  )
}

export function SlashCommandMenu({
  open,
  input,
  selectedIndex,
  onSelectCommand,
  onSelectOption,
  ctx,
}: SlashCommandMenuProps) {
  const liveResult = React.useMemo(
    () => getCommandSuggestions(input, ctx),
    [input, ctx]
  )
  // There's something to actually show once the composer wants a menu *and*
  // the input resolves to real content — a bare "/" is `open` with no result.
  const shouldShow = open && liveResult !== null

  // Freeze the last real result so a close (composer clears, command runs,
  // "/" gets backspaced away) fades the content that was there rather than
  // going blank a frame before the panel itself finishes animating out.
  const lastResult = React.useRef<SuggestionResult | null>(null)
  if (liveResult) lastResult.current = liveResult
  const result = open ? liveResult : lastResult.current

  const [mounted, setMounted] = React.useState(shouldShow)
  const [visible, setVisible] = React.useState(false)

  React.useEffect(() => {
    if (shouldShow) {
      setMounted(true)
      // Mount in the hidden pose first, then flip to visible so the browser
      // has something to transition from instead of snapping straight to
      // open. A single rAF isn't enough here: it fires *before* the browser
      // paints, so if we flip to visible inside it, the hidden pose never
      // actually gets painted and there's nothing to transition from — the
      // two state changes land in the same frame. Nesting a second rAF
      // forces a real paint of the hidden pose in between.
      let inner = 0
      const outer = requestAnimationFrame(() => {
        inner = requestAnimationFrame(() => setVisible(true))
      })
      return () => {
        cancelAnimationFrame(outer)
        cancelAnimationFrame(inner)
      }
    }
    setVisible(false)
    const timer = setTimeout(() => setMounted(false), TRANSITION_MS)
    return () => clearTimeout(timer)
  }, [shouldShow])

  const listRef = React.useRef<HTMLDivElement>(null)

  // Scroll active item into view
  React.useEffect(() => {
    if (!listRef.current) return
    const activeEl = listRef.current.querySelector("[data-active='true']")
    if (activeEl) {
      activeEl.scrollIntoView({ block: "nearest" })
    }
  }, [selectedIndex, result])

  if (!mounted || !result) return null

  // -------------------------------------------------------------------------
  // Stage 1: Searching for commands — a bare "/" shows nothing (commands.ts
  // returns null for an empty query), so by the time this renders the user
  // has already typed something and every row here is a real match. No
  // header, no args hint, no description: just the matched command per line,
  // the same information a fuzzy finder anywhere else gives you.
  // -------------------------------------------------------------------------
  if (result.mode === "command") {
    const { suggestions, query } = result

    if (suggestions.length === 0) {
      return (
        <div className={panelClass(visible)}>
          <div className="px-3.5 py-2.5 text-xs text-sidebar-foreground/60">
            No command matching <span className="font-mono text-sidebar-foreground">“/{query}”</span>
          </div>
        </div>
      )
    }

    return (
      <div className={panelClass(visible)}>
        <div ref={listRef} className="p-1 space-y-0.5">
          {suggestions.map(({ command, currentDisplay }, idx) => {
            const isActive = idx === selectedIndex
            const Icon = command.icon

            return (
              <button
                key={command.name}
                type="button"
                data-active={isActive}
                onClick={() => onSelectCommand(command)}
                className={rowClass(isActive)}
              >
                <span className="flex items-center gap-2">
                  <Icon className="size-3.5 shrink-0 text-sidebar-foreground/50" />
                  <span className="font-mono font-medium text-sidebar-primary">
                    /<HighlightMatched text={command.name} query={query} />
                  </span>
                </span>

                {currentDisplay && (
                  <span className="shrink-0 font-mono text-[11px] text-sidebar-foreground/50 tabular-nums">
                    {currentDisplay}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </div>
    )
  }

  // -------------------------------------------------------------------------
  // Stage 2: Sub-options & Argument Helpers
  // -------------------------------------------------------------------------
  const { command, argQuery, data } = result

  return (
    <div className={panelClass(visible)}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-sidebar-border px-3 py-1.5 text-[10px] font-medium tracking-[0.14em] text-sidebar-foreground/50 uppercase">
        <div className="flex items-center gap-1.5">
          <span className="font-mono font-medium text-sidebar-primary">/{command.name}</span>
          <span>·</span>
          <span>{command.label}</span>
        </div>
        <span className={HINT_CLASS}>↵ execute · esc dismiss</span>
      </div>

      <div ref={listRef} className="max-h-64 overflow-y-auto p-1">
        {/* Model Options */}
        {data.type === "model" && (
          <div className="space-y-0.5">
            {data.options.length === 0 ? (
              <div className="px-3 py-2 text-xs text-sidebar-foreground/60">
                No model matching <span className="font-mono text-sidebar-foreground">“{argQuery}”</span>
              </div>
            ) : (
              data.options.map(({ model, isCurrent }, idx) => {
                const isActive = idx === selectedIndex
                return (
                  <button
                    key={model.id}
                    type="button"
                    data-active={isActive}
                    onClick={() => onSelectOption(model.id, true)}
                    className={rowClass(isActive)}
                  >
                    <span className="flex items-center gap-2">
                      <Check
                        className={cn(
                          "size-3.5 text-sidebar-primary transition-opacity",
                          isCurrent ? "opacity-100" : "opacity-0"
                        )}
                      />
                      <span className={cn(isCurrent ? "font-medium text-sidebar-primary" : "text-sidebar-foreground")}>
                        <HighlightMatched text={model.name} query={argQuery} />
                      </span>
                      {model.family && (
                        <span className="text-sidebar-foreground/50">({model.family})</span>
                      )}
                    </span>
                    {model.tier && (
                      <span className="ml-auto font-mono text-[10px] font-medium tracking-[0.12em] text-sidebar-foreground/50 uppercase">
                        {model.tier}
                      </span>
                    )}
                  </button>
                )
              })
            )}
          </div>
        )}

        {/* Budget Options */}
        {data.type === "budget" && (
          <div className="space-y-0.5">
            {data.options.map((opt, idx) => {
              const isActive = idx === selectedIndex
              return (
                <button
                  key={opt.value}
                  type="button"
                  data-active={isActive}
                  onClick={() => onSelectOption(opt.value, true)}
                  className={rowClass(isActive)}
                >
                  <span className="flex items-center gap-2">
                    <Check
                      className={cn(
                        "size-3.5 text-sidebar-primary transition-opacity",
                        opt.isCurrent ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <span className={cn("font-medium", opt.isCurrent && "text-sidebar-primary")}>
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto text-sidebar-foreground/50 text-[11px]">{opt.hint}</span>
                </button>
              )
            })}
          </div>
        )}

        {/* Toggle Options */}
        {data.type === "toggle" && (
          <div className="space-y-0.5">
            {data.options.map((opt, idx) => {
              const isActive = idx === selectedIndex
              return (
                <button
                  key={opt.value}
                  type="button"
                  data-active={isActive}
                  onClick={() => onSelectOption(opt.value, true)}
                  className={rowClass(isActive)}
                >
                  <span className="flex items-center gap-2">
                    <Check
                      className={cn(
                        "size-3.5 text-sidebar-primary transition-opacity",
                        opt.isCurrent ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <span className={cn(opt.isCurrent && "font-medium text-sidebar-primary")}>
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto font-mono text-sidebar-foreground/50 text-[11px]">
                    {opt.value}
                  </span>
                </button>
              )
            })}
          </div>
        )}

        {/* Theme Options */}
        {data.type === "theme" && (
          <div className="space-y-0.5">
            {data.options.map((opt, idx) => {
              const isActive = idx === selectedIndex
              const Icon =
                opt.value === "light"
                  ? Sun
                  : opt.value === "dark"
                    ? Moon
                    : SunMoon

              return (
                <button
                  key={opt.value}
                  type="button"
                  data-active={isActive}
                  onClick={() => onSelectOption(opt.value, true)}
                  className={rowClass(isActive)}
                >
                  <span className="flex items-center gap-2">
                    <Check
                      className={cn(
                        "size-3.5 text-sidebar-primary transition-opacity",
                        opt.isCurrent ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <Icon className="size-3.5 text-sidebar-foreground/50" />
                    <span className={cn(opt.isCurrent && "font-medium text-sidebar-primary")}>
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto text-sidebar-foreground/50 text-[11px]">
                    {opt.description}
                  </span>
                </button>
              )
            })}
          </div>
        )}

        {/* Number Range & Presets */}
        {data.type === "number" && (
          <div className="flex flex-col gap-2 p-2.5">
            <div className="flex items-baseline justify-between text-xs text-sidebar-foreground/60">
              <span>
                Range:{" "}
                <span className="font-mono text-sidebar-foreground">
                  {data.info.min.toFixed(data.info.precision)} –{" "}
                  {data.info.max.toFixed(data.info.precision)}
                </span>
              </span>
              <span>
                Current:{" "}
                <span className="font-mono font-medium text-sidebar-primary tabular-nums">
                  {data.info.current.toFixed(data.info.precision)}
                </span>
              </span>
            </div>

            {data.info.presets.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                <span className="text-[10px] font-medium tracking-[0.12em] text-sidebar-foreground/50 uppercase">
                  Presets:
                </span>
                <div className="inline-flex gap-0.5 rounded border border-sidebar-border p-0.5">
                  {data.info.presets.map((preset, idx) => {
                    const isPresetActive =
                      data.info.current === preset.value || idx === selectedIndex
                    return (
                      <Button
                        key={preset.label}
                        type="button"
                        size="xs"
                        variant={isPresetActive ? "secondary" : "ghost"}
                        onClick={() =>
                          onSelectOption(
                            preset.value.toFixed(data.info.precision),
                            true
                          )
                        }
                        className={cn(
                          "font-mono text-xs h-6 px-2",
                          isPresetActive && "text-sidebar-primary font-medium"
                        )}
                      >
                        {preset.label}
                      </Button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* System Prompt Helper */}
        {data.type === "text" && (
          <div className="flex flex-col gap-2 p-2.5">
            <div className="flex items-baseline justify-between text-xs">
              <span className="text-sidebar-foreground/60">Current steering prompt:</span>
              {data.info.current && (
                <button
                  type="button"
                  onClick={() => onSelectOption("clear", true)}
                  className="text-xs text-destructive hover:underline"
                >
                  Clear
                </button>
              )}
            </div>

            <div className="rounded border border-sidebar-border bg-sidebar-accent/30 p-2 text-xs text-sidebar-foreground/90 font-mono">
              {data.info.current
                ? `“${data.info.current}”`
                : "No system prompt configured."}
            </div>

            {data.info.templates.length > 0 && (
              <div className="space-y-0.5 pt-0.5">
                <span className="text-[10px] font-medium tracking-[0.12em] text-sidebar-foreground/50 uppercase">
                  Templates:
                </span>
                <div className="space-y-0.5">
                  {data.info.templates.map((tpl, idx) => {
                    const isActive = idx === selectedIndex
                    return (
                      <button
                        key={tpl.label}
                        type="button"
                        data-active={isActive}
                        onClick={() => onSelectOption(tpl.value, true)}
                        className={cn(
                          "flex w-full items-baseline justify-between gap-2 rounded px-2 py-1 text-left text-xs transition-colors",
                          isActive
                            ? "bg-sidebar-accent text-sidebar-accent-foreground font-medium"
                            : "text-sidebar-foreground hover:bg-sidebar-accent/50"
                        )}
                      >
                        <span className="font-medium text-sidebar-primary shrink-0">{tpl.label}</span>
                        <span className="truncate text-sidebar-foreground/50 text-[11px]">
                          {tpl.value}
                        </span>
                      </button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {/* Action summary */}
        {data.type === "action" && (
          <div className="flex items-center justify-between p-2.5 text-xs text-sidebar-foreground/60">
            <span>{data.description}</span>
            <Button
              type="button"
              size="xs"
              variant="secondary"
              onClick={() => onSelectOption("", true)}
              className="text-sidebar-primary font-medium"
            >
              Run
            </Button>
          </div>
        )}
      </div>
    </div>
  )
}
