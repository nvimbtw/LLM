import Link from "next/link"
import type { ComponentProps, HTMLAttributes } from "react"
import { cn } from "@/lib/utils"

export interface ResourceListProps extends HTMLAttributes<HTMLUListElement> {
  density?: "open" | "compact"
}

/** Groups domain-owned resource rows with open catalogue or compact operational spacing. */
export function ResourceList({
  density = "open",
  className,
  ...props
}: ResourceListProps) {
  return (
    <ul
      data-density={density}
      className={cn("min-w-0", className)}
      {...props}
    />
  )
}

export interface ResourceRowProps extends HTMLAttributes<HTMLLIElement> {
  density?: "open" | "compact"
}

/** Gives a resource one unruled, responsive row; callers retain selection and action ownership. */
export function ResourceRow({
  density = "open",
  className,
  ...props
}: ResourceRowProps) {
  return (
    <li
      className={cn(
        "group relative min-w-0",
        density === "compact" ? "py-3" : "py-5",
        className
      )}
      {...props}
    />
  )
}

/** Opens the resource's main content while leaving sibling actions independently accessible. */
export function ResourceLink({
  className,
  ...props
}: ComponentProps<typeof Link>) {
  return (
    <Link
      className={cn(
        "block min-w-0 flex-1 outline-none hover:text-primary focus-visible:ring-2 focus-visible:ring-ring",
        className
      )}
      {...props}
    />
  )
}

/** Presents a wrapping resource name in the shared catalogue typography. */
export function ResourceTitle({
  className,
  ...props
}: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h2
      className={cn(
        "font-display text-xl leading-snug font-normal break-words",
        className
      )}
      {...props}
    />
  )
}

/** Presents provenance and current state without decorative separators. */
export function ResourceMetadata({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground",
        className
      )}
      {...props}
    />
  )
}

/** Groups measured resource facts, preserving domain-specific labels and unavailable values. */
export function ResourceFacts({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex flex-wrap gap-x-6 gap-y-2 text-sm text-muted-foreground tabular-nums",
        className
      )}
      {...props}
    />
  )
}

/** Keeps quiet row actions discoverable for pointer, touch and keyboard interaction. */
export function ResourceActions({
  className,
  ...props
}: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "flex shrink-0 items-center gap-1 text-muted-foreground",
        className
      )}
      {...props}
    />
  )
}
