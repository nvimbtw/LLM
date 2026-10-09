"use client"

import { useJobsContext } from "./jobs-provider"
import { AdminSectionHeader } from "@/components/admin/admin-section-header"
import { HistoryPager } from "./history-pager"
import { JobHistory } from "./job-history"

/**
 * What has finished — the same ledger as `QueueView`, past tense.
 *
 * It renders last on whatever page carries it, and is a component of its own for
 * that reason rather than for any difference in what it is: a page of history is
 * hundreds of pixels of rows, and anything placed under it is unreachable without
 * scrolling past every run ever made.
 *
 * Full width, not a column beside the queue. That was tried, to use the margin,
 * but a row carries seven facts of its own — time, label, kind, ran, status,
 * menu — and squeezing them into a third of the page cost more than the empty
 * margin did.
 */
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

  return (
    <section>
      {/* The count describes the whole ledger, rather than the current page. */}
      <AdminSectionHeader
        title="History"
        meta={`${historyTotal} ${historyTotal === 1 ? "job" : "jobs"}`}
      />
      <JobHistory jobs={history} onRetry={retry} onDelete={remove} />
      <HistoryPager
        page={historyPage}
        pages={historyPages}
        onGo={goToHistoryPage}
      />
    </section>
  )
}
