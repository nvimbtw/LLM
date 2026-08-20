"use client"

import * as React from "react"
import { Toast as ToastPrimitive } from "@base-ui/react/toast"
import { XIcon } from "lucide-react"

import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"

// A single shared manager so `toast(...)` can be called from anywhere — event
// handlers, slash commands, non-React modules — not only from inside a hook.
const manager = ToastPrimitive.createToastManager()

export type ToastType = "default" | "success" | "error"

export interface ToastOptions {
  title?: React.ReactNode
  description?: React.ReactNode
  type?: ToastType
  /** Auto-dismiss delay in ms. Defaults to the provider's `timeout`. */
  timeout?: number
}

/** Fire a toast from anywhere. Returns the toast id (usable with `dismissToast`). */
export function toast(options: ToastOptions): string {
  return manager.add(options)
}
toast.success = (title: React.ReactNode, description?: React.ReactNode) =>
  toast({ title, description, type: "success" })
toast.error = (title: React.ReactNode, description?: React.ReactNode) =>
  toast({ title, description, type: "error" })

export function dismissToast(id?: string) {
  manager.close(id)
}

// A tinted left rail is the only per-type flourish — quiet, editorial, no loud
// filled banners. Neutral by default, success/error borrow the chart/destructive
// tokens so it reads as one system in both themes.
const RAIL_BY_TYPE: Record<ToastType, string> = {
  default: "before:bg-border",
  success: "before:bg-chart-2",
  error: "before:bg-destructive",
}

function ToastList() {
  const { toasts } = ToastPrimitive.useToastManager()
  return toasts.map((t) => {
    const type = (t.type as ToastType) ?? "default"
    return (
      <ToastPrimitive.Root
        key={t.id}
        toast={t}
        swipeDirection="right"
        className={cn(
          // Frosted on the composer's terms — see chat-composer.tsx. A toast is
          // the other thing in the app that lands on top of live content it must
          // not erase: it arrives over a thread or a running queue, and a solid
          // card cuts a hole in whatever it covers for four seconds. The blur is
          // what keeps what passes behind it from being readable, and the
          // saturate puts back the colour that blurring alone averages out.
          // `bg-clip-padding` matters more here than it looks: without it the
          // translucent fill runs under the border and the edge goes muddy.
          "relative flex items-start gap-3 overflow-hidden rounded-(--radius-sheet) border border-border bg-popover/75 bg-clip-padding py-3 pr-3 pl-4 text-popover-foreground shadow-overlay backdrop-blur-2xl backdrop-saturate-150",
          "before:absolute before:inset-y-0 before:left-0 before:w-1 before:content-['']",
          "transition-all duration-200 ease-out data-ending-style:translate-x-4 data-ending-style:opacity-0 data-starting-style:translate-x-4 data-starting-style:opacity-0",
          RAIL_BY_TYPE[type]
        )}
      >
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          {t.title != null && (
            <ToastPrimitive.Title className="text-sm font-medium text-foreground" />
          )}
          {t.description != null && (
            <ToastPrimitive.Description className="text-xs leading-relaxed text-muted-foreground" />
          )}
        </div>
        <ToastPrimitive.Close
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              className="-mt-1 -mr-1 size-6 shrink-0"
            />
          }
          aria-label="Dismiss"
        >
          <XIcon className="size-3.5" />
        </ToastPrimitive.Close>
      </ToastPrimitive.Root>
    )
  })
}

/**
 * Wraps the app so `toast(...)` renders. Mounts the top-right viewport itself, so
 * a single `<ToastProvider>` in the root layout is all that's needed.
 */
export function ToastProvider({ children }: { children: React.ReactNode }) {
  return (
    <ToastPrimitive.Provider toastManager={manager} timeout={4000}>
      {children}
      <ToastPrimitive.Portal>
        <ToastPrimitive.Viewport className="fixed top-0 right-0 z-[100] m-4 flex w-80 max-w-[calc(100vw-2rem)] flex-col gap-2 outline-none">
          <ToastList />
        </ToastPrimitive.Viewport>
      </ToastPrimitive.Portal>
    </ToastPrimitive.Provider>
  )
}
