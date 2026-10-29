import type { ServiceState, ServiceStatus } from "@/lib/service-health"
import { cn } from "@/lib/utils"

export const SERVICES = [
  { id: "llm-engine", name: "LLM engine" },
  { id: "data-engine", name: "Data engine" },
  { id: "gyms-engine", name: "Gyms engine" },
  { id: "bench-engine", name: "Bench engine" },
  { id: "just-bash", name: "just-bash" },
  { id: "searxng", name: "SearXNG" },
] as const

type FigureState = ServiceState | "unavailable"

const LABELS: Record<FigureState, string> = {
  online: "Online",
  reconnecting: "Reconnecting",
  offline: "Offline",
  not_configured: "Not configured",
  unavailable: "—",
}

const INK: Record<FigureState, string> = {
  online: "text-status-success",
  reconnecting: "text-muted-foreground",
  offline: "text-destructive",
  not_configured: "text-muted-foreground",
  unavailable: "text-muted-foreground",
}

/** Show readable service observations in the same order as the header's tree. */
export function ServiceHealth({
  services,
}: {
  services: ServiceStatus[] | null
}) {
  const online = services?.filter(
    (service) => service.state === "online"
  ).length
  return (
    <section aria-labelledby="service-health-heading">
      <header className="mb-4 flex flex-wrap items-baseline gap-x-3 gap-y-1">
        <h2 id="service-health-heading" className="font-display text-2xl">
          Services
        </h2>
        <p aria-live="polite" className="text-xs text-muted-foreground">
          {services
            ? `${online} of ${services.length} online`
            : "Dashboard connection reconnecting"}
        </p>
      </header>
      <ul className="grid grid-cols-2 gap-x-6 gap-y-5 lg:grid-cols-3">
        {SERVICES.map((service) => {
          const observation = services?.find((item) => item.id === service.id)
          const state = observation?.state ?? "unavailable"
          return (
            <li key={service.id} className="min-w-0">
              <div className="min-w-0">
                <p className="text-sm font-medium text-foreground">
                  {service.name}
                </p>
                <p className={cn("mt-1 text-xs", INK[state])}>
                  {LABELS[state]}
                </p>
                {observation?.detail && (
                  <p className="mt-1 text-xs text-muted-foreground">
                    {observation.detail}
                  </p>
                )}
              </div>
            </li>
          )
        })}
      </ul>
    </section>
  )
}
