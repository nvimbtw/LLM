"use client"

import * as React from "react"

import { cn } from "@/lib/utils"

export interface TextareaProps extends React.ComponentProps<"textarea"> {
  variant?: "default" | "inline"
}

/** Expands an inline prose field to its content when native field sizing is unavailable, preserving all multiline text at narrow widths. */
function sizeProseField(element: HTMLTextAreaElement | null) {
  if (!element || CSS.supports("field-sizing", "content")) return
  element.style.height = "auto"
  element.style.height = `${element.scrollHeight + element.offsetHeight - element.clientHeight}px`
}

/** Renders accessible multiline input; inline prose retains its caret and resizing fallback without focus decoration. */
function Textarea({
  className,
  variant = "default",
  ref,
  onChange,
  ...props
}: TextareaProps) {
  const element = React.useRef<HTMLTextAreaElement | null>(null)
  React.useLayoutEffect(() => {
    if (variant !== "inline") return
    sizeProseField(element.current)
    const observer = new ResizeObserver(() => sizeProseField(element.current))
    if (element.current) observer.observe(element.current)
    return () => observer.disconnect()
  }, [variant, props.value])
  return (
    <textarea
      ref={(node) => {
        element.current = node
        if (typeof ref === "function") return ref(node)
        else if (ref) ref.current = node
      }}
      data-slot="textarea"
      className={cn(
        "flex field-sizing-content min-h-16 w-full rounded-(--radius-control) border border-input bg-transparent px-2.5 py-2 text-base transition-colors outline-none placeholder:text-muted-foreground focus-visible:border-ring focus-visible:ring-3 focus-visible:ring-ring/50 disabled:cursor-not-allowed disabled:bg-input/50 disabled:opacity-50 aria-invalid:border-destructive aria-invalid:ring-3 aria-invalid:ring-destructive/20 md:text-sm dark:bg-input/30 dark:disabled:bg-input/80 dark:aria-invalid:border-destructive/50 dark:aria-invalid:ring-destructive/40",
        variant === "inline" &&
          "min-h-5 resize-none overflow-hidden rounded-none border-0 bg-transparent px-0 py-0 text-sm leading-relaxed shadow-none focus-visible:ring-0 focus-visible:outline-none disabled:bg-transparent dark:bg-transparent dark:disabled:bg-transparent",
        className
      )}
      onChange={(event) => {
        if (variant === "inline") sizeProseField(event.currentTarget)
        onChange?.(event)
      }}
      {...props}
    />
  )
}

export { Textarea }
