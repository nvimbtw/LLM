"use client"

import type { ComponentProps } from "react"
import { SidebarMenuButton } from "@/components/ui/sidebar"
import { cn } from "@/lib/utils"

export type SidebarEntryButtonProps = ComponentProps<typeof SidebarMenuButton>

/** Renders project and conversation links with shared alignment and hover/active markers, preserving menu-button behavior. */
export function SidebarEntryButton({
  isActive = false,
  className,
  ...props
}: SidebarEntryButtonProps) {
  return (
    <SidebarMenuButton
      {...props}
      isActive={isActive}
      className={cn(
        "relative pl-3.5",
        "before:absolute before:top-1/2 before:left-1 before:h-0 before:w-px before:-translate-y-1/2",
        "before:rounded-full before:bg-sidebar-foreground/30 before:transition-all before:content-['']",
        "hover:before:h-4",
        isActive &&
          "before:h-4 before:w-0.5 before:bg-primary hover:before:bg-primary",
        className
      )}
    />
  )
}
