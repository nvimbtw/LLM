"use client"

import * as React from "react"
import { ChevronRight } from "lucide-react"

import { cn } from "@/lib/utils"

// The one place progressive-disclosure detail lives: the tokenizer/build log and
// the saved checkpoints, framed as an article's appendix at the foot of the main
// column. Always present (even when empty), collapsed by default. It opens
// *downward* and its body scrolls internally at a fixed max-height, so expanding
// it never reflows the telemetry above — the "reserve the slot, fill it in" rule.
export function Appendix({
  log,
  checkpoints,
}: {
  log: string[]
  checkpoints: string[]
}) {
  const [open, setOpen] = React.useState(false)
  const logEndRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (open) logEndRef.current?.scrollIntoView({ block: "nearest" })
  }, [log, open])

  return (
    <section className="border-t pt-4">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center gap-1.5 text-sm text-muted-foreground select-none hover:text-foreground"
        aria-expanded={open}
      >
        <ChevronRight
          className={cn("size-4 transition-transform", open && "rotate-90")}
        />
        <span className="font-medium">Appendix</span>
        <span className="text-xs">
          — log ({log.length} lines) · checkpoints ({checkpoints.length})
        </span>
      </button>

      {open && (
        <div className="mt-4 space-y-4">
          <div className="space-y-1.5">
            <div className="text-xs font-medium text-muted-foreground">Log</div>
            <div className="max-h-64 overflow-y-auto rounded-(--radius-control-sm) border bg-muted/40 p-3 font-mono text-xs leading-relaxed">
              {log.length === 0 ? (
                <span className="text-muted-foreground">No output yet.</span>
              ) : (
                log.map((line, i) => <div key={i}>{line}</div>)
              )}
              <div ref={logEndRef} />
            </div>
          </div>

          <div className="space-y-1.5">
            <div className="text-xs font-medium text-muted-foreground">
              Checkpoints
            </div>
            {checkpoints.length === 0 ? (
              <span className="text-xs text-muted-foreground">
                No checkpoints yet.
              </span>
            ) : (
              <ul className="max-h-40 space-y-0.5 overflow-y-auto font-mono text-xs text-muted-foreground">
                {checkpoints.map((cp) => (
                  <li key={cp} className="truncate">
                    {cp}
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </section>
  )
}
