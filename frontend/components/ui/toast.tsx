"use client"

import * as React from "react"
import { Toast as ToastPrimitive } from "@base-ui/react/toast"
import { XIcon } from "lucide-react"
import { cn } from "@/lib/utils"
import { getToken } from "@/lib/api"
import { useSessionToken } from "@/hooks/use-auth"
import {
  loadNotificationAccount,
  type NotificationSeverity,
} from "@/lib/notifications"
import {
  NotificationStore,
  type NotificationInput,
  type LiveNotification,
} from "@/lib/notification-store"
import { Button } from "./button"
import { NotificationDialog } from "@/components/notifications/notification-dialog"

const manager = ToastPrimitive.createToastManager()
const store = new NotificationStore({
  token: getToken,
  read: (key) => localStorage.getItem(key),
  write: (key, value) => localStorage.setItem(key, value),
  now: () => Date.now(),
  popup: (item) => manager.add(popupOptions(item)),
  close: (id) => manager.close(id),
})
/** Maps one store-owned popup to Base UI delivery, leaving lifetime scheduling with the store. */
function popupOptions(item: LiveNotification) {
  return {
    id: item.id,
    title: item.titleNode ?? item.title,
    description: item.descriptionNode ?? item.description,
    type: item.severity,
    timeout: 0,
    priority: item.severity === "error" ? ("high" as const) : ("low" as const),
    onClose: () => store.popupClosed(item.id),
  }
}
const EMPTY_ITEMS: readonly LiveNotification[] = []
export type ToastType = NotificationSeverity
export type ToastOptions = NotificationInput

/** Flattens supported React content for safe durable text while leaving its live toast rendering intact. */
export function notificationText(node: React.ReactNode): string {
  if (typeof node === "string" || typeof node === "number") return String(node)
  if (Array.isArray(node)) return node.map(notificationText).join("")
  if (React.isValidElement<{ children?: React.ReactNode }>(node))
    return notificationText(node.props.children)
  return ""
}

/** Emits one notification from any caller; severity and lifetime are independent. */
export function toast(options: ToastOptions): string {
  return store.add(
    options,
    notificationText(options.title),
    notificationText(options.description)
  )
}
/** Emits routine success feedback with transient delivery by default. */
toast.success = (title: React.ReactNode, description?: React.ReactNode) =>
  toast({ title, description, type: "success" })
/** Emits an error with a sticky popup and retained history by default. */
toast.error = (title: React.ReactNode, description?: React.ReactNode) =>
  toast({ title, description, type: "error" })

/** Dismisses the notification from popup and retained history, or clears both when no id is supplied. */
export function dismissToast(id?: string) {
  store.dismiss(id)
}

interface NotificationContextValue {
  items: readonly LiveNotification[]
  error: string | null
  ready: boolean
  open: boolean
  setOpen: (open: boolean) => void
  retry: () => void
}
const NotificationContext =
  React.createContext<NotificationContextValue | null>(null)

/** Exposes the current account's history and modal controls, masking replacement-session records immediately. */
export function useNotifications() {
  const context = React.useContext(NotificationContext)
  if (!context) throw new Error("Notifications require ToastProvider.")
  return context
}

/** Reports whether an error component can route explanations through the mounted notification system. */
export function useNotificationFeedbackAvailable() {
  return React.useContext(NotificationContext) !== null
}

/** Renders Base UI accessible popups only for entries still owned by the current session. */
function ToastList({ items }: { items: readonly LiveNotification[] }) {
  const { toasts, add } = ToastPrimitive.useToastManager()
  React.useEffect(() => {
    items
      .filter(
        (item) =>
          item.popup && !toasts.some((existing) => existing.id === item.id)
      )
      .forEach((item) => add(popupOptions(item)))
  }, [items, toasts, add])
  const ids = new Set(items.filter((item) => item.popup).map((item) => item.id))
  return toasts
    .filter((item) => ids.has(item.id) || item.transitionStatus === "ending")
    .map((item) => (
      <ToastPrimitive.Root
        key={item.id}
        toast={item}
        aria-hidden={item.transitionStatus === "ending"}
        swipeDirection="right"
        className={cn(
          "relative flex items-start gap-3 overflow-hidden rounded-(--radius-sheet) border border-border bg-card/75 bg-clip-padding py-3 pr-3 pl-4 text-foreground shadow-overlay backdrop-blur-2xl backdrop-saturate-150 transition-all duration-200 ease-out data-ending-style:pointer-events-none data-ending-style:translate-x-4 data-ending-style:opacity-0 data-starting-style:translate-x-4 data-starting-style:opacity-0 motion-reduce:transition-none",
          item.type === "error" && "border-destructive/40",
          item.type === "warning" && "border-chart-1/40"
        )}
      >
        <div className="flex min-w-0 flex-1 flex-col gap-0.5">
          {item.title != null && (
            <ToastPrimitive.Title className="text-sm font-medium" />
          )}
          {item.description != null && (
            <ToastPrimitive.Description className="text-xs leading-relaxed text-muted-foreground" />
          )}
        </div>
        <ToastPrimitive.Close
          render={
            <Button
              variant="ghost"
              size="icon-sm"
              className="-mt-1 -mr-1 size-6 shrink-0"
              onClick={() => dismissToast(item.id)}
            />
          }
          aria-label="Dismiss notification"
          aria-hidden={false}
        >
          <XIcon className="size-3.5" />
        </ToastPrimitive.Close>
      </ToastPrimitive.Root>
    ))
}

/** Owns verified identity, history restoration, accessible toast delivery and the notifications modal for the application. */
export function ToastProvider({ children }: { children: React.ReactNode }) {
  const token = useSessionToken()
  const state = React.useSyncExternalStore(
    store.subscribe,
    store.getSnapshot,
    store.getSnapshot
  )
  const [open, setOpen] = React.useState(false)
  const [retryKey, retryIdentity] = React.useReducer(
    (value: number) => value + 1,
    0
  )
  const items = state.token === token ? state.items : EMPTY_ITEMS
  const error = state.token === token ? state.error : null
  React.useEffect(() => {
    store.setSession(token)
    if (!token) return
    const controller = new AbortController()
    let active = true
    void loadNotificationAccount(token, controller.signal)
      .then((accountId) => {
        if (active && getToken() === token) store.restore(token, accountId)
      })
      .catch((cause: unknown) => {
        if (active && getToken() === token) store.fail(cause)
      })
    return () => {
      active = false
      controller.abort()
    }
  }, [token, retryKey])
  React.useEffect(() => {
    store.resume()
    return () => store.dispose()
  }, [])
  React.useEffect(() => {
    /** Opens the notification center with the unassigned semantic application shortcut. */
    const key = (event: KeyboardEvent) => {
      if (
        event.ctrlKey &&
        event.altKey &&
        !event.metaKey &&
        !event.shiftKey &&
        event.key?.toLowerCase() === "n"
      ) {
        event.preventDefault()
        setOpen(true)
      }
    }
    window.addEventListener("keydown", key)
    return () => window.removeEventListener("keydown", key)
  }, [])
  React.useEffect(() => {
    if (error)
      toast({
        title: "Notification history unavailable",
        description: error,
        type: "error",
        lifetime: "transient",
        timeout: 6000,
        dedupKey: "notification-storage-error",
      })
  }, [error])
  /** Retries a blocked identity/read or a failed write without discarding live history. */
  const retry = React.useCallback(() => {
    if (state.ready) store.retryWrite()
    else retryIdentity()
  }, [state.ready])
  const context = {
    items,
    error,
    ready: state.token === token && state.ready,
    open,
    setOpen,
    retry,
  }
  return (
    <NotificationContext.Provider value={context}>
      {children}
      <NotificationDialog />
      <ToastPrimitive.Provider
        key={token ?? "guest"}
        toastManager={manager}
        timeout={0}
        limit={100}
      >
        <ToastPrimitive.Portal>
          <ToastPrimitive.Viewport className="fixed top-0 right-0 z-[100] m-4 flex max-h-[calc(100dvh-2rem)] w-80 max-w-[calc(100vw-2rem)] flex-col gap-2 overflow-y-auto outline-none">
            <ToastList items={items} />
          </ToastPrimitive.Viewport>
        </ToastPrimitive.Portal>
      </ToastPrimitive.Provider>
    </NotificationContext.Provider>
  )
}
