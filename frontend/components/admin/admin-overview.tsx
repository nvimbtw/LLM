"use client"

import "./admin-header.css"

import { AdminDirectory } from "@/components/admin/admin-directory"
import { ServiceHealth } from "@/components/admin/service-health"
import { ServiceTree } from "@/components/admin/service-tree"
import { HistoryView } from "@/components/jobs/history-view"
import { QueueView } from "@/components/jobs/queue-view"
import { PageHeader } from "@/components/ui/page-header"
import { useServiceHealth } from "@/hooks/use-service-health"

/** Share one health subscription between the Admin header illustration and its service list. */
export function AdminOverview() {
  const services = useServiceHealth()
  return (
    <div className="mx-auto max-w-[104rem]">
      <PageHeader
        title="Admin"
        textureClassName="admin-header-texture"
        action={
          <div className="admin-service-tree w-[min(44rem,100cqw)]">
            <ServiceTree services={services} />
          </div>
        }
      />
      <div className="grid items-start gap-x-12 gap-y-10 lg:grid-cols-[minmax(0,1fr)_15rem]">
        <div className="min-w-0 space-y-10">
          <ServiceHealth services={services} />
          <QueueView />
          <HistoryView />
        </div>
        <AdminDirectory />
      </div>
    </div>
  )
}
