"use client"

import { Tabs as TabsPrimitive } from "@base-ui/react/tabs"
import { cva, type VariantProps } from "class-variance-authority"

import { cn } from "@/lib/utils"

/** Coordinates accessible panel selection along the supplied horizontal or vertical axis. */
function Tabs({
  className,
  orientation = "horizontal",
  ...props
}: TabsPrimitive.Root.Props) {
  return (
    <TabsPrimitive.Root
      data-slot="tabs"
      data-orientation={orientation}
      orientation={orientation}
      className={cn(
        "group/tabs flex gap-4 data-horizontal:flex-col",
        className
      )}
      {...props}
    />
  )
}

const tabsListVariants = cva(
  "group/tabs-list inline-flex w-fit items-center gap-6 bg-transparent text-muted-foreground group-data-vertical/tabs:flex-col group-data-vertical/tabs:items-stretch group-data-vertical/tabs:gap-1",
  {
    variants: { variant: { default: "", line: "" } },
    defaultVariants: { variant: "default" },
  }
)

/** Arranges quiet text tabs; both legacy variant names share one horizontal design. */
function TabsList({
  className,
  variant = "default",
  ...props
}: TabsPrimitive.List.Props & VariantProps<typeof tabsListVariants>) {
  return (
    <TabsPrimitive.List
      data-slot="tabs-list"
      data-variant={variant}
      className={cn(tabsListVariants({ variant }), className)}
      {...props}
    />
  )
}

/** Selects a panel using type and primary colour; vertical modal navigation additionally retains its thin left marker. */
function TabsTrigger({ className, ...props }: TabsPrimitive.Tab.Props) {
  return (
    <TabsPrimitive.Tab
      data-slot="tabs-trigger"
      className={cn(
        "relative inline-flex min-h-9 shrink-0 items-center justify-center gap-1.5 rounded-none border-0 bg-transparent px-0 py-1 text-sm font-normal whitespace-nowrap text-muted-foreground transition-colors hover:text-foreground focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring disabled:pointer-events-none disabled:opacity-50 aria-disabled:pointer-events-none aria-disabled:opacity-50 motion-reduce:transition-none data-active:font-medium data-active:text-primary [&_svg]:pointer-events-none [&_svg]:shrink-0 [&_svg:not([class*='size-'])]:size-4",
        "group-data-vertical/tabs:w-full group-data-vertical/tabs:justify-start group-data-vertical/tabs:after:absolute group-data-vertical/tabs:after:inset-y-2 group-data-vertical/tabs:after:left-0 group-data-vertical/tabs:after:w-px group-data-vertical/tabs:after:bg-primary group-data-vertical/tabs:after:opacity-0 group-data-vertical/tabs:after:content-[''] group-data-vertical/tabs:data-active:text-foreground group-data-vertical/tabs:data-active:after:opacity-100",
        className
      )}
      {...props}
    />
  )
}

/** Presents the selected tab's associated content without adding a second surface. */
function TabsContent({ className, ...props }: TabsPrimitive.Panel.Props) {
  return (
    <TabsPrimitive.Panel
      data-slot="tabs-content"
      className={cn("flex-1 text-sm outline-none", className)}
      {...props}
    />
  )
}

export { Tabs, TabsList, TabsTrigger, TabsContent, tabsListVariants }
