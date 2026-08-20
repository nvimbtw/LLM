"use client"

import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"

import { cn } from "@/lib/utils"

/**
 * Light/dark, for screens that have no sidebar to put it in — the auth pages.
 * Inside the app the same switch lives in the ⌘K settings dialog, and `d`
 * toggles it from anywhere (`theme-provider.tsx`).
 *
 * The two icons are both rendered and one is hidden by the `dark:` variant
 * rather than picked in JS. The theme is a class on `<html>`, so CSS already
 * knows which one to show, and the usual `mounted` state exists only because
 * the JS version cannot know before hydration — this has nothing to get wrong.
 */
export function ThemeToggle({ className }: { className?: string }) {
  const { resolvedTheme, setTheme } = useTheme()

  return (
    <button
      type="button"
      aria-label="Toggle theme"
      onClick={() => setTheme(resolvedTheme === "dark" ? "light" : "dark")}
      className={cn(
        "inline-flex size-8 items-center justify-center rounded-(--radius-control-sm) text-muted-foreground transition-colors hover:bg-accent hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring focus-visible:outline-none",
        className
      )}
    >
      <Sun className="size-4 dark:hidden" />
      <Moon className="hidden size-4 dark:block" />
    </button>
  )
}
