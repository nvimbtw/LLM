"use client"

import * as React from "react"
import { XIcon } from "lucide-react"
import { useNotifications, dismissToast } from "@/components/ui/toast"
import {
  AccountModal,
  AccountModalRail,
  ACCOUNT_MODAL_PANEL,
  ACCOUNT_MODAL_TAB,
} from "@/components/ui/account-modal"
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs"
import { Button } from "@/components/ui/button"

/** Shows retained undismissed notifications in the same frame and section rail as account settings. */
export function NotificationDialog() {
  const { items, open, setOpen, error, retry, ready } = useNotifications()
  const [section, setSection] = React.useState("all")
  const retained = items.filter((item) => item.lifetime !== "transient")
  const visible =
    section === "errors"
      ? retained.filter(
          (item) => item.severity === "error" || item.severity === "warning"
        )
      : retained
  return (
    <AccountModal
      open={open}
      onOpenChange={setOpen}
      title="Notifications"
      description="Review and dismiss your retained notifications."
    >
      <Tabs
        orientation="vertical"
        value={section}
        onValueChange={(value) => setSection(String(value))}
        className="min-h-0 flex-1 gap-0"
      >
        <AccountModalRail title="Notifications" shortcut="Ctrl+Alt+N">
          <TabsList
            aria-label="Notification sections"
            activateOnFocus
            variant="line"
            className="w-full items-stretch justify-start gap-1 p-0"
          >
            <TabsTrigger value="all" className={ACCOUNT_MODAL_TAB}>
              All{" "}
              <span className="ml-auto text-muted-foreground">
                {retained.length}
              </span>
            </TabsTrigger>
            <TabsTrigger value="errors" className={ACCOUNT_MODAL_TAB}>
              Needs attention
            </TabsTrigger>
          </TabsList>
        </AccountModalRail>
        {["all", "errors"].map((value) => (
          <TabsContent
            key={value}
            value={value}
            className={ACCOUNT_MODAL_PANEL}
          >
            <div className="mb-6 flex items-center justify-between gap-3 pr-4">
              <h2 className="font-display text-2xl">
                {value === "all" ? "Notifications" : "Needs attention"}
              </h2>
              {retained.length > 0 && (
                <Button
                  variant="ghost"
                  size="xs"
                  onClick={() => dismissToast()}
                >
                  Dismiss all
                </Button>
              )}
            </div>
            {error && (
              <div className="mb-5 flex items-center gap-3 text-xs">
                <span>{error}</span>
                <Button variant="outline" size="xs" onClick={retry}>
                  Try again
                </Button>
              </div>
            )}
            {!error && !ready && (
              <p className="text-sm text-muted-foreground">
                Loading notifications…
              </p>
            )}
            {ready && !visible.length && (
              <p className="text-sm text-muted-foreground">No notifications</p>
            )}
            <ol className="space-y-6">
              {[...visible].reverse().map((item) => (
                <li key={item.id} className="flex gap-3">
                  <div className="min-w-0 flex-1">
                    <p
                      className={
                        item.severity === "error"
                          ? "text-sm font-medium break-words text-destructive"
                          : "text-sm font-medium break-words"
                      }
                    >
                      {item.title}
                    </p>
                    {item.description && (
                      <p className="mt-1 text-xs leading-relaxed break-words text-muted-foreground">
                        {item.description}
                      </p>
                    )}
                    <time
                      dateTime={new Date(item.createdAt).toISOString()}
                      className="mt-2 block text-2xs text-muted-foreground"
                    >
                      {new Date(item.createdAt).toLocaleString()}
                    </time>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon-xs"
                    aria-label={`Dismiss ${item.title || "notification"}`}
                    onClick={() => dismissToast(item.id)}
                  >
                    <XIcon />
                  </Button>
                </li>
              ))}
            </ol>
          </TabsContent>
        ))}
      </Tabs>
    </AccountModal>
  )
}
