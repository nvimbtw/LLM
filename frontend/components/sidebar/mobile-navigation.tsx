"use client"

import { SidebarTrigger } from "@/components/ui/sidebar"

/** Keeps the signed-in mobile sidebar discoverable with a compact visible opener; desktop pages retain their existing frame. */
export function MobileNavigation() {
  return (
    <div className="flex items-center px-4 pt-3 md:hidden">
      <SidebarTrigger aria-label="Open navigation" className="size-10" />
      <span className="ml-1 text-xs text-muted-foreground">Navigation</span>
    </div>
  )
}
