"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { useJobsContext } from "./jobs-provider"
import { AdminSectionHeader } from "@/components/admin/admin-section-header"
import { HistoryPager } from "./history-pager"
import { JobHistory } from "./job-history"

/** Shows five recent rows or the full current history page, retaining bounded paging and all lifecycle actions. */
export function HistoryView() {
  const {
    history,
    historyTotal,
    historyPage,
    historyPages,
    goToHistoryPage,
    remove,
    retry,
  } = useJobsContext()

  const [expanded, setExpanded] = React.useState(false)
  const shown = expanded ? history : history.slice(0, 5)

  return (
    <section>
      <AdminSectionHeader
        title="History"
        meta={`${historyTotal} ${historyTotal === 1 ? "job" : "jobs"}`}
        action={
          history.length > 5 && (
            <Button
              variant="text"
              size="sm"
              aria-expanded={expanded}
              onClick={() => setExpanded(!expanded)}
            >
              {expanded ? "Recent only" : "Full history"}
            </Button>
          )
        }
      />
      <JobHistory jobs={shown} onRetry={retry} onDelete={remove} />
      {!expanded && history.length > 5 && (
        <p className="py-2 text-xs text-muted-foreground">
          Showing 5 of {history.length} on this page
        </p>
      )}
      <HistoryPager
        page={historyPage}
        pages={historyPages}
        onGo={goToHistoryPage}
      />
    </section>
  )
}
