"use client"

import { useServiceHealth } from "@/hooks/use-service-health"
import { ServiceState } from "@/lib/service-health"

const NAMES: Record<string, string> = {
  "llm-engine": "LLM engine",
  "data-engine": "Data engine",
  "gyms-engine": "Gyms engine",
  "bench-engine": "Bench engine",
  "just-bash": "just-bash",
  searxng: "SearXNG",
}
const LABELS: Record<ServiceState, string> = {
  online: "Online",
  reconnecting: "Reconnecting",
  offline: "Offline",
  not_configured: "Not configured",
}

/** Show backend-observed service states and distinguish a disconnected dashboard subscription. */
export function ServiceHealth() {
  const services = useServiceHealth()
  return (
    <section aria-labelledby="service-health-heading">
      <h2 id="service-health-heading" className="font-serif text-xl">
        Services
      </h2>
      {services === null ? (
        <p className="mt-3 text-sm text-muted-foreground">
          Dashboard connection reconnecting
        </p>
      ) : (
        <ul className="mt-3 space-y-2 text-sm">
          {services.map((service) => (
            <li
              key={service.id}
              className="flex flex-wrap items-baseline justify-between gap-x-6 gap-y-1"
            >
              <span>{NAMES[service.id]}</span>
              <span
                className={
                  service.state === "online"
                    ? "text-status-success"
                    : service.state === "offline"
                      ? "text-destructive"
                      : "text-muted-foreground"
                }
              >
                {LABELS[service.state]}
              </span>
              {service.detail && (
                <span className="w-full text-xs text-muted-foreground">
                  {service.detail}
                </span>
              )}
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
