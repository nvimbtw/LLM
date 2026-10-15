"use client"

import * as React from "react"

import {
  fuzzyMatch,
  getCommandSuggestions,
  type CommandContext,
  type ArgumentSuggestions,
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

const TRANSITION_MS = 150

interface HighlightMatchedProps {
  text: string
  query: string
  className?: string
}

/**
 * Emphasizes matching query ranges in command or option text with quiet weight changes.
 */
function HighlightMatched({ text, query, className }: HighlightMatchedProps) {
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
      <span key={`match-${i}`} className="font-medium text-foreground">
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

interface CommandDrawerProps {
  visible: boolean
  children: React.ReactNode
}

/** Reveals a glass command drawer from behind its composer's upper edge. The clipped join and lower stacking layer keep its motion out of the editor without changing editor height. */
function CommandDrawer({ visible, children }: CommandDrawerProps) {
  return (
    <div
      data-slot="slash-command-drawer"
      className="pointer-events-none absolute inset-x-3 bottom-full z-0 overflow-hidden rounded-t-(--radius-control)"
    >
      <div
        role="region"
        aria-label="Slash commands"
        aria-hidden={!visible}
        inert={!visible}
        data-slot="slash-command-menu"
        className={cn(
          "max-h-[50dvh] overflow-y-auto overscroll-contain rounded-t-(--radius-control) border border-b-0 border-border/60 bg-card/75 text-foreground backdrop-blur-2xl backdrop-saturate-150 transition-[opacity,transform] duration-150 ease-out motion-reduce:transition-none",
          visible
            ? "pointer-events-auto translate-y-0 opacity-100"
            : "pointer-events-none translate-y-3 opacity-0"
        )}
      >
        {children}
      </div>
    </div>
  )
}

/** Formats a numeric command value at its declared precision without redundant zeros; context multipliers carry their × unit. */
function numericDisplay(
  commandName: string,
  value: number,
  precision: number
): string {
  const number = String(Number(value.toFixed(precision)))
  return commandName === "context-multiplier" ? `${number}×` : number
}

/** Reads the current setting for a selected command's quiet heading; actions and plan editing have no setting readout. */
function argumentCurrentDisplay(
  commandName: string,
  data: ArgumentSuggestions
): string | null {
  switch (data.type) {
    case "number":
      return numericDisplay(commandName, data.info.current, data.info.precision)
    case "model":
      return data.options.find((option) => option.isCurrent)?.model.name ?? null
    case "budget":
    case "theme":
      return data.options.find((option) => option.isCurrent)?.label ?? null
    case "toggle":
      return data.options.find((option) => option.isCurrent)?.value ?? null
    default:
      return null
  }
}

/** Returns a transparent command/option row with a thin selection marker and visible keyboard focus. */
function rowClass(isActive: boolean): string {
  return cn(
    "relative flex h-auto min-h-9 w-full items-center justify-between gap-3 rounded-none px-2 py-2 text-left text-sm font-normal transition-colors",
    "before:absolute before:top-1/2 before:left-0 before:h-3 before:w-px before:-translate-y-1/2 before:bg-foreground before:opacity-0 before:content-['']",
    isActive
      ? "font-medium text-foreground before:opacity-70"
      : "text-muted-foreground hover:text-foreground"
  )
}

/** Shows bottom-up fuzzy matches and argument controls in the composer's top drawer. Retains the last result during exit, owns its animation timers, and preserves keyboard selection near the editor. */
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
  const shouldShow = open && liveResult !== null
  const [frozenResult, setFrozenResult] = React.useState(liveResult)
  if (liveResult && liveResult !== frozenResult) setFrozenResult(liveResult)
  const result = liveResult ?? frozenResult

  const [mounted, setMounted] = React.useState(shouldShow)
  const [visible, setVisible] = React.useState(false)
  const [prevShouldShow, setPrevShouldShow] = React.useState(shouldShow)
  if (shouldShow !== prevShouldShow) {
    setPrevShouldShow(shouldShow)
    setVisible(false)
    if (shouldShow) setMounted(true)
  }
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
  React.useEffect(() => {
    if (shouldShow || !mounted) return
    const timer = setTimeout(() => setMounted(false), TRANSITION_MS)
    return () => clearTimeout(timer)
  }, [shouldShow, mounted])

  const listRef = React.useRef<HTMLDivElement>(null)
  React.useEffect(() => {
    if (!listRef.current) return
    const activeEl = listRef.current.querySelector("[data-active='true']")
    if (activeEl) {
      activeEl.scrollIntoView({ block: "nearest" })
    }
  }, [selectedIndex, result])

  if (!mounted || !result) return null
  if (result.mode === "command") {
    const { suggestions, query } = result

    if (suggestions.length === 0) {
      return (
        <CommandDrawer visible={visible}>
          <div className="px-3.5 py-2.5 text-xs text-muted-foreground">
            No command matching{" "}
            <span className="font-mono text-foreground">“/{query}”</span>
          </div>
        </CommandDrawer>
      )
    }

    return (
      <CommandDrawer visible={visible}>
        <div ref={listRef} className="flex flex-col-reverse gap-0.5 p-2">
          {suggestions.map(({ command, currentDisplay }, idx) => {
            const isActive = idx === selectedIndex

            return (
              <Button
                variant="text"
                key={command.name}
                type="button"
                data-active={isActive}
                onClick={() => onSelectCommand(command)}
                className={rowClass(isActive)}
              >
                <HighlightMatched text={command.name} query={query} />

                {currentDisplay && (
                  <span className="shrink-0 text-2xs text-muted-foreground tabular-nums">
                    {currentDisplay}
                  </span>
                )}
              </Button>
            )
          })}
        </div>
      </CommandDrawer>
    )
  }
  const { command, argQuery, data } = result
  const currentDisplay = argumentCurrentDisplay(command.name, data)

  return (
    <CommandDrawer visible={visible}>
      <div className="flex items-baseline gap-3 px-4 pt-3 pb-1 text-xs text-muted-foreground">
        <span>{command.name}</span>
        {currentDisplay && (
          <span className="truncate text-foreground tabular-nums">
            <span className="sr-only">Current: </span>
            {currentDisplay}
          </span>
        )}
      </div>

      <div ref={listRef} className="p-2">
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
                  <Button
                    variant="text"
                    key={model.id}
                    type="button"
                    data-active={isActive}
                    onClick={() => onSelectOption(model.id, true)}
                    className={rowClass(isActive)}
                  >
                    <span className="flex items-center gap-2">
                      <span
                        className={cn(
                          isCurrent
                            ? "font-medium text-foreground"
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
                      <span className="ml-auto tag font-medium text-muted-foreground">
                        {model.tier}
                      </span>
                    )}
                  </Button>
                )
              })
            )}
          </div>
        )}

        {data.type === "plan" && (
          <div className="flex flex-col-reverse gap-0.5">
            {data.options.map((option, index) => (
              <Button
                key={option.value}
                variant="text"
                data-active={index === selectedIndex}
                className={rowClass(index === selectedIndex)}
                onClick={() => onSelectOption(option.value, option.execute)}
              >
                <span>{option.label}</span>
                <span className="truncate text-2xs text-muted-foreground">
                  {option.description}
                </span>
              </Button>
            ))}
            {!data.options.length && (
              <p className="p-2.5 text-xs text-muted-foreground">
                Use add &lt;task&gt;, edit, or save.
              </p>
            )}
          </div>
        )}

        {data.type === "budget" && (
          <div className="flex flex-col-reverse gap-0.5">
            {data.options.map((opt, idx) => {
              const isActive = idx === selectedIndex
              return (
                <Button
                  variant="text"
                  key={opt.value}
                  type="button"
                  data-active={isActive}
                  onClick={() => onSelectOption(opt.value, true)}
                  className={rowClass(isActive)}
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={cn(
                        "font-medium",
                        opt.isCurrent && "text-foreground"
                      )}
                    >
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto text-2xs text-muted-foreground">
                    {opt.hint}
                  </span>
                </Button>
              )
            })}
          </div>
        )}

        {data.type === "toggle" && (
          <div className="flex flex-col-reverse gap-0.5">
            {data.options.map((opt, idx) => {
              const isActive = idx === selectedIndex
              return (
                <Button
                  variant="text"
                  key={opt.value}
                  type="button"
                  data-active={isActive}
                  onClick={() => onSelectOption(opt.value, true)}
                  className={rowClass(isActive)}
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={cn(
                        opt.isCurrent && "font-medium text-foreground"
                      )}
                    >
                      {opt.label}
                    </span>
                  </span>
                </Button>
              )
            })}
          </div>
        )}

        {data.type === "theme" && (
          <div className="flex flex-col-reverse gap-0.5">
            {data.options.map((opt, idx) => {
              const isActive = idx === selectedIndex

              return (
                <Button
                  variant="text"
                  key={opt.value}
                  type="button"
                  data-active={isActive}
                  onClick={() => onSelectOption(opt.value, true)}
                  className={rowClass(isActive)}
                >
                  <span className="flex items-center gap-2">
                    <span
                      className={cn(
                        opt.isCurrent && "font-medium text-foreground"
                      )}
                    >
                      {opt.label}
                    </span>
                  </span>
                  <span className="ml-auto text-2xs text-muted-foreground">
                    {opt.description}
                  </span>
                </Button>
              )
            })}
          </div>
        )}

        {data.type === "number" && (
          <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-2 px-2 pb-2">
            <div className="flex flex-wrap gap-1">
              {data.info.presets.map((preset, idx) => (
                <Button
                  key={preset.label}
                  type="button"
                  size="xs"
                  variant="text"
                  title={preset.label}
                  aria-label={preset.label}
                  aria-pressed={data.info.current === preset.value}
                  data-active={idx === selectedIndex}
                  onClick={() =>
                    onSelectOption(
                      preset.value.toFixed(data.info.precision),
                      true
                    )
                  }
                  className={cn(
                    "h-6 rounded-none px-2 text-xs",
                    idx === selectedIndex &&
                      "font-medium text-foreground underline underline-offset-4"
                  )}
                >
                  {numericDisplay(
                    command.name,
                    preset.value,
                    data.info.precision
                  )}
                </Button>
              ))}
            </div>
            <span className="ml-auto text-2xs text-muted-foreground tabular-nums">
              <span className="sr-only">Allowed range: </span>
              {numericDisplay(command.name, data.info.min, data.info.precision)}
              –
              {numericDisplay(command.name, data.info.max, data.info.precision)}
            </span>
          </div>
        )}

        {data.type === "text" && (
          <div className="flex flex-col gap-2 p-2.5">
            <div className="flex items-start justify-between gap-3 text-xs">
              <div className="border-l border-border pl-3 leading-relaxed text-foreground">
                {data.info.current
                  ? `“${data.info.current}”`
                  : "No system prompt configured."}
              </div>
              {data.info.current && (
                <Button
                  variant="text"
                  type="button"
                  onClick={() => onSelectOption("clear", true)}
                  className="text-xs text-destructive hover:underline"
                >
                  Clear
                </Button>
              )}
            </div>

            {data.info.templates.length > 0 && (
              <div className="space-y-0.5 pt-0.5">
                <div className="flex flex-col-reverse gap-0.5">
                  {data.info.templates.map((tpl, idx) => {
                    const isActive = idx === selectedIndex
                    return (
                      <Button
                        variant="text"
                        key={tpl.label}
                        type="button"
                        data-active={isActive}
                        onClick={() => onSelectOption(tpl.value, true)}
                        className={rowClass(isActive)}
                      >
                        <span className="shrink-0 font-medium text-foreground">
                          {tpl.label}
                        </span>
                        <span className="truncate text-2xs text-muted-foreground">
                          {tpl.value}
                        </span>
                      </Button>
                    )
                  })}
                </div>
              </div>
            )}
          </div>
        )}

        {data.type === "action" && (
          <div className="flex items-center justify-between p-2.5 text-xs text-muted-foreground">
            <span>{data.description}</span>
            <Button
              type="button"
              size="xs"
              variant="text"
              onClick={() => onSelectOption("", true)}
              className="font-medium text-foreground"
            >
              Run
            </Button>
          </div>
        )}
      </div>
    </CommandDrawer>
  )
}
