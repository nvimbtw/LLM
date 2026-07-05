"use client"

import * as React from "react"
import { Check, ChevronDown } from "lucide-react"

import {
  FluxThrobber,
  type ThrobberMode,
} from "@/components/chat/flux-throbber"
import {
  Card,
  CardAction,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { cn } from "@/lib/utils"

// A collapsible phase of the training flow. The header is clickable and the
// section auto-expands when it becomes `active` and collapses once it finishes,
// so completed phases shrink to a one-line summary instead of the page growing
// without bound. While active it shows the throbber in the matching mode.
export function Section({
  title,
  description,
  active = false,
  done = false,
  throbberMode,
  action,
  children,
}: {
  title: string
  description?: React.ReactNode
  active?: boolean
  done?: boolean
  throbberMode?: ThrobberMode
  action?: React.ReactNode
  children: React.ReactNode
}) {
  const [open, setOpen] = React.useState(active)

  // Follow the phase: expand when this section becomes active, collapse once it
  // finishes. The user can still toggle manually in between. This is React's
  // "adjust state during render" pattern rather than an effect, so the section
  // never flashes open/closed on the frame the phase changes.
  const [prevActive, setPrevActive] = React.useState(active)
  if (active !== prevActive) {
    setPrevActive(active)
    setOpen(active)
  }

  return (
    <Card className="gap-0 py-0">
      <CardHeader
        role="button"
        tabIndex={0}
        onClick={() => setOpen((o) => !o)}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault()
            setOpen((o) => !o)
          }
        }}
        className="cursor-pointer items-center gap-3 py-4 select-none"
      >
        <div className="flex min-w-0 items-center gap-3">
          {active && throbberMode ? (
            <FluxThrobber mode={throbberMode} className="h-8 w-14 shrink-0" />
          ) : done ? (
            <span className="flex size-8 shrink-0 items-center justify-center rounded-md bg-muted/60 text-chart-2">
              <Check className="size-4" />
            </span>
          ) : null}
          <div className="grid min-w-0 gap-1">
            <CardTitle>{title}</CardTitle>
            {description && (
              <CardDescription className="truncate">
                {description}
              </CardDescription>
            )}
          </div>
        </div>
        <CardAction className="flex items-center gap-2 self-center">
          {action && <span onClick={(e) => e.stopPropagation()}>{action}</span>}
          <ChevronDown
            className={cn(
              "size-4 text-muted-foreground transition-transform",
              open && "rotate-180"
            )}
          />
        </CardAction>
      </CardHeader>
      {open && <CardContent className="pt-1 pb-5">{children}</CardContent>}
    </Card>
  )
}
