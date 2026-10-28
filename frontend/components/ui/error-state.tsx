"use client"

import * as React from "react"
import { Button } from "./button"
import {
  notificationText,
  toast,
  useNotificationFeedbackAvailable,
} from "./toast"
import { cn } from "@/lib/utils"

export interface FormErrorProps {
  children: React.ReactNode
  className?: string
}
export interface LoadErrorProps {
  title?: React.ReactNode
  hint?: React.ReactNode
  onRetry?: () => void
  className?: string
}

/** Routes an error episode once through notifications; clearing it resets deduplication while StrictMode reruns keep the same episode identity. */
function useErrorFeedback(title: React.ReactNode, hint?: React.ReactNode) {
  const available = useNotificationFeedbackAvailable()
  const id = React.useId()
  const text = notificationText(title)
  const detail = notificationText(hint)
  const episode = React.useRef({
    text: "",
    detail: "",
    revision: 0,
    emitted: -1,
  })
  React.useEffect(() => {
    if (episode.current.text !== text || episode.current.detail !== detail)
      episode.current = {
        text,
        detail,
        revision: episode.current.revision + 1,
        emitted: -1,
      }
    if (
      available &&
      (text || detail) &&
      episode.current.emitted !== episode.current.revision
    ) {
      episode.current.emitted = episode.current.revision
      toast({
        title: text,
        description: detail,
        type: "error",
        dedupKey: `error:${id}:${episode.current.revision}`,
      })
    }
  }, [available, id, text, detail])
  return available
}

/** Reports a rejected form through notification feedback while preserving a screen-reader focus target; standalone fixtures retain visible fallback text. */
export function FormError({ children, className }: FormErrorProps) {
  const available = useErrorFeedback(children)
  return (
    <span
      className={
        available ? "sr-only" : cn("text-sm text-destructive", className)
      }
      role={available ? undefined : "alert"}
    >
      {children}
    </span>
  )
}

/** Reports a failed read through notifications and keeps its useful retry action at the failed content location. */
export function LoadError({
  title = "That didn't load.",
  hint,
  onRetry,
  className,
}: LoadErrorProps) {
  const available = useErrorFeedback(title, hint)
  return (
    <div className={cn("py-5", className)}>
      {!available && (
        <p role="alert" className="text-sm">
          {title}
          {hint && (
            <span className="mt-1 block text-xs text-muted-foreground">
              {hint}
            </span>
          )}
        </p>
      )}
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  )
}
