"use client"

import * as React from "react"
import { Check, Moon, Sun, SunMoon } from "lucide-react"

import {
  fuzzyMatch,
  getCommandSuggestions,
  type CommandContext,
  type SlashCommandDef,
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
      <span key={`match-${i}`} className="font-semibold text-primary">
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

// The menu is the upper half of the composer's card, not a panel floating above
// it: one outline around both, with a rule where they meet.
//
// That is why nothing here draws a border, a radius, a shadow or a surface. The
// card owns all four, and this only contributes the rule along its own bottom
// edge. It also settles the blur: joined, there is a single `backdrop-filter` on
// the card, and a single backdrop root. As a separate floating panel this needed
// its own blur, and a nested one samples only what its ancestor's root already
// painted — which, for a panel sitting outside the card's box, was nothing at
// all, so it rendered as plain transparency with the page crisply readable
// through it.
//
// Opening animates `grid-template-rows` from 0fr to 1fr with the content in an
// `overflow-hidden` track, which is the one way to transition to a height nobody
// has measured. The card is bottom-anchored (sticky in chat-thread.tsx), so the
// growth pushes its top edge up the page rather than shoving the editor down
// under the reader's cursor.
function Panel({
  visible,
  children,
}: {
  visible: boolean
  children: React.ReactNode
}) {
  return (
    <div
      className={cn(
        "grid transition-[grid-template-rows,opacity] duration-150 ease-out",
        visible ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0"
      )}
    >
      <div className="overflow-hidden">
        <div className="border-b border-border">{children}</div>
      </div>
    </div>
  )
}

const HINT_CLASS =
  "font-mono text-[11px] tracking-normal lowercase text-muted-foreground"

// Row styling shared by every option list: a hairline accent rule on the left
// that grows in on hover/active, rather than a filled highlight.
function rowClass(isActive: boolean) {
  return cn(
    "relative flex w-full items-center justify-between gap-3 rounded-md px-2.5 py-1.5 text-left text-xs transition-colors",
    "before:absolute before:top-1/2 before:left-1 before:h-0 before:w-0.5 before:-translate-y-1/2 before:rounded-full before:bg-primary before:transition-all before:content-['']",
    isActive
      ? "bg-accent text-accent-foreground before:h-3.5"
      : "text-foreground hover:bg-accent/60 hover:before:h-2"
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
  // Adjusted during render rather than in an Effect — the React-recommended
  // way to derive state from a prop, and it keeps this in sync a render
  // earlier than an Effect would.
  const [frozenResult, setFrozenResult] = React.useState(liveResult)
  if (liveResult && liveResult !== frozenResult) setFrozenResult(liveResult)
  const result = liveResult ?? frozenResult

  const [mounted, setMounted] = React.useState(shouldShow)
  const [visible, setVisible] = React.useState(false)
  const [prevShouldShow, setPrevShouldShow] = React.useState(shouldShow)

  // Also a render-time adjustment: mounting (and dropping back to the hidden
  // pose to transition from) has to be visible in the very render that
  // follows `shouldShow` flipping, not a render later via an Effect, or the
  // browser would have nothing painted to transition from.
  if (shouldShow !== prevShouldShow) {
    setPrevShouldShow(shouldShow)
    setVisible(false)
    if (shouldShow) setMounted(true)
  }

  // Once mounted in the hidden pose, flip to visible so the browser has
  // something to transition from instead of snapping straight to open. A
  // single rAF isn't enough: it fires *before* the browser paints, so
  // flipping inside it would land in the same frame as the hidden pose above
  // and there'd be nothing to transition from. Nesting a second rAF forces a
  // real paint in between.
  React.useEffect(() => {
    if (!shouldShow || visible) return
    let inner = 0
    const outer = requestAnimationFrame(() => {
      inner = requestAnimationFrame(() => setVisible(true))
    })
    return () => {
      cancelAnimationFrame(outer)
      cancelAnimationFrame(inner)
    }
  }, [shouldShow, visible])

  // Unmount once the exit transition has had time to finish.
  React.useEffect(() => {
    if (shouldShow || !mounted) return
    const timer = setTimeout(() => setMounted(false), TRANSITION_MS)
    return () => clearTimeout(timer)
  }, [shouldShow, mounted])

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
  // Every list here paints bottom-up (`flex-col-reverse`), so index 0 — the best
  // match, and the one Enter and Tab act on — is the row closest to the line you
  // are typing on. The menu grows upward out of the composer, so ranking upward
  // is the direction that puts the answer where the eye already is; a best match
  // at the far top of a six-row list is the furthest thing from the cursor.
  //
  // Reversed in paint rather than in the data: `selectedIndex` still means "nth
  // best" for the composer's Tab/Enter handling, and only the arrow keys have to
  // know about the flip (see chat-composer.tsx).
  // -------------------------------------------------------------------------

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
        <Panel visible={visible}>
          <div className="px-3.5 py-2.5 text-xs text-muted-foreground">
            No command matching{" "}
            <span className="font-mono text-foreground">“/{query}”</span>
          </div>
        </Panel>
      )
    }

    return (
      <Panel visible={visible}>
        <div ref={listRef} className="flex flex-col-reverse gap-0.5 p-1">
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
                  <Icon className="size-3.5 shrink-0 text-muted-foreground" />
                  <span className="font-mono font-medium text-primary">
                    /<HighlightMatched text={command.name} query={query} />
                  </span>
                </span>

                {currentDisplay && (
                  <span className="shrink-0 font-mono text-[11px] text-muted-foreground tabular-nums">
                    {currentDisplay}
                  </span>
                )}
              </button>
            )
          })}
        </div>
      </Panel>
    )
  }

  // -------------------------------------------------------------------------
  // Stage 2: Sub-options & Argument Helpers
  // -------------------------------------------------------------------------
  const { command, argQuery, data } = result

  return (
    <Panel visible={visible}>
      {/* Header */}
      <div className="flex items-center justify-between border-b border-border px-3.5 py-1.5 text-[10px] font-medium tracking-[0.14em] text-muted-foreground uppercase">
        <div className="flex items-center gap-1.5">
          <span className="font-mono font-medium text-primary">
            /{command.name}
          </span>
          <span>·</span>
          <span>{command.label}</span>
        </div>
        <span className={HINT_CLASS}>↵ execute · esc dismiss</span>
      </div>

      <div ref={listRef} className="max-h-64 overflow-y-auto p-1">
        {/* Model Options */}
        {data.type === "model" && (
          <div className="flex flex-col-reverse gap-0.5">
            {data.options.length === 0 ? (
              <div className="px-3.5 py-2.5 text-xs text-muted-foreground">
                No model matching{" "}
                <span className="font-mono text-foreground">“{argQuery}”</span>
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
                          "size-3.5 text-primary transition-opacity",
                          isCurrent ? "opacity-100" : "opacity-0"
                        )}
                      />
                      <span
                        className={cn(
                          isCurrent
                            ? "font-medium text-primary"
                            : "text-foreground"
                        )}
                      >
                        <HighlightMatched text={model.name} query={argQuery} />
                      </span>
                      {model.family && (
                        <span className="text-muted-foreground">
                          ({model.family})
                        </span>
                      )}
                    </span>
                    {model.tier && (
                      <span className="ml-auto font-mono text-[10px] font-medium tracking-[0.12em] text-muted-foreground uppercase">
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
          <div className="flex flex-col-reverse gap-0.5">
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
                        "size-3.5 text-primary transition-opacity",
                        opt.isCurrent ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <span
                      className={cn(
                        "font-medium",
                        opt.isCurrent && "text-primary"
                      )}
                    >
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto text-[11px] text-muted-foreground">
                    {opt.hint}
                  </span>
                </button>
              )
            })}
          </div>
        )}

        {/* Toggle Options */}
        {data.type === "toggle" && (
          <div className="flex flex-col-reverse gap-0.5">
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
                        "size-3.5 text-primary transition-opacity",
                        opt.isCurrent ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <span
                      className={cn(
                        opt.isCurrent && "font-medium text-primary"
                      )}
                    >
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto font-mono text-[11px] text-muted-foreground">
                    {opt.value}
                  </span>
                </button>
              )
            })}
          </div>
        )}

        {/* Theme Options */}
        {data.type === "theme" && (
          <div className="flex flex-col-reverse gap-0.5">
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
                        "size-3.5 text-primary transition-opacity",
                        opt.isCurrent ? "opacity-100" : "opacity-0"
                      )}
                    />
                    <Icon className="size-3.5 text-muted-foreground" />
                    <span
                      className={cn(
                        opt.isCurrent && "font-medium text-primary"
                      )}
                    >
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto text-[11px] text-muted-foreground">
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
            <div className="flex items-baseline justify-between text-xs text-muted-foreground">
              <span>
                Range:{" "}
                <span className="font-mono text-foreground">
                  {data.info.min.toFixed(data.info.precision)} –{" "}
                  {data.info.max.toFixed(data.info.precision)}
                </span>
              </span>
              <span>
                Current:{" "}
                <span className="font-mono font-medium text-primary tabular-nums">
                  {data.info.current.toFixed(data.info.precision)}
                </span>
              </span>
            </div>

            {data.info.presets.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                <span className="text-[10px] font-medium tracking-[0.12em] text-muted-foreground uppercase">
                  Presets:
                </span>
                <div className="inline-flex gap-0.5 rounded-md border border-border p-0.5">
                  {data.info.presets.map((preset, idx) => {
                    const isPresetActive =
                      data.info.current === preset.value ||
                      idx === selectedIndex
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
                          "h-6 px-2 font-mono text-xs",
                          isPresetActive && "font-medium text-primary"
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
              <span className="text-muted-foreground">
                Current steering prompt:
              </span>
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

            <div className="rounded-md border border-border bg-accent/30 p-2 font-mono text-xs text-foreground">
              {data.info.current
                ? `“${data.info.current}”`
                : "No system prompt configured."}
            </div>

            {data.info.templates.length > 0 && (
              <div className="space-y-0.5 pt-0.5">
                <span className="text-[10px] font-medium tracking-[0.12em] text-muted-foreground uppercase">
                  Templates:
                </span>
                <div className="flex flex-col-reverse gap-0.5">
                  {data.info.templates.map((tpl, idx) => {
                    const isActive = idx === selectedIndex
                    return (
                      <button
                        key={tpl.label}
                        type="button"
                        data-active={isActive}
                        onClick={() => onSelectOption(tpl.value, true)}
                        className={cn(
                          "flex w-full items-baseline justify-between gap-2 rounded-md px-2.5 py-1.5 text-left text-xs transition-colors",
                          isActive
                            ? "bg-accent font-medium text-accent-foreground"
                            : "text-foreground hover:bg-accent/50"
                        )}
                      >
                        <span className="shrink-0 font-medium text-primary">
                          {tpl.label}
                        </span>
                        <span className="truncate text-[11px] text-muted-foreground">
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
          <div className="flex items-center justify-between p-2.5 text-xs text-muted-foreground">
            <span>{data.description}</span>
            <Button
              type="button"
              size="xs"
              variant="secondary"
              onClick={() => onSelectOption("", true)}
              className="font-medium text-primary"
            >
              Run
            </Button>
          </div>
        )}
      </div>
    </Panel>
  )
}
