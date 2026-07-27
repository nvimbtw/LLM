import { cn } from "@/lib/utils"

/**
 * The two page widths, and the only two.
 *
 * Every page used to carry its own `max-w-*`, which is how the app ended up a
 * narrow column with two deserts beside it on a wide monitor — the emptiness was
 * the aspect ratio, not the minimalism.
 *
 * - `reading` is for prose and forms. A composer's fields do not get better at
 *   1600px; a line of body copy gets worse. Chat, auth and every `/jobs/new/*`
 *   form live here.
 * - `board` is for anything data-bearing — the queue, cycles, benchmark, models,
 *   the corpus. These want columns, and columns are the editorial idiom the app
 *   was only ever using one of. Capped rather than edge-to-edge so a 4K monitor
 *   does not stretch a table into a horizon.
 */
export type PageWidth = "reading" | "board"

const WIDTH: Record<PageWidth, string> = {
  reading: "max-w-3xl",
  board: "max-w-[110rem]",
}

export interface PageShellProps {
  width?: PageWidth
  className?: string
  children: React.ReactNode
}

export function PageShell({
  width = "board",
  className,
  children,
}: PageShellProps) {
  return (
    <div
      className={cn(
        "mx-auto w-full flex-1 px-5 py-10 sm:px-8 lg:px-12",
        WIDTH[width],
        className
      )}
    >
      {children}
    </div>
  )
}
