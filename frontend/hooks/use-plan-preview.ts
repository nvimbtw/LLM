"use client"

import * as React from "react"
import type { Plan, PlanTask } from "@/lib/plan"

export const PLAN_SCENARIOS = [
  { id: "working", label: "Working" },
  { id: "paused", label: "Paused" },
  { id: "complete", label: "Complete" },
  { id: "long", label: "Long plan" },
  { id: "empty", label: "No plan" },
] as const

export type PlanScenario = (typeof PLAN_SCENARIOS)[number]["id"]

const TASK_TITLES = [
  "Read the project conventions",
  "Trace the current tool pipeline",
  "Design the task list above the composer",
  "Connect the plan to the conversation",
  "Handle interruptions and revised tasks",
  "Check the small-screen layout",
  "Verify the finished experience",
]
const REPLAY_INTERVAL_MS = 2400

function fixture(scenario: PlanScenario, revision: number): Plan | null {
  if (scenario === "empty") return null
  const titles =
    scenario === "long"
      ? [
          ...TASK_TITLES,
          "Check keyboard access and focus",
          "Test the reduced-motion preference",
          "Review a task with a much longer title that needs to stay readable without stretching the composer beyond the conversation",
          "Review the light theme",
          "Review the dark theme",
          "Document the plan's behavior",
          "Prepare the next demonstration",
        ]
      : TASK_TITLES
  return {
    id: "composer-preview",
    title: "Give the model a visible plan",
    revision,
    tasks: titles.map(
      (title, index): PlanTask => ({
        id: `t${index + 1}`,
        title,
        status:
          scenario === "complete" || index < 2
            ? "completed"
            : index === 2
              ? "in_progress"
              : "pending",
      })
    ),
  }
}

function advance(plan: Plan | null): Plan | null {
  if (!plan) return null
  const current = plan.tasks.find((task) => task.status === "in_progress")
  const next = plan.tasks.find((task) => task.status === "pending")
  return {
    ...plan,
    revision: plan.revision + 1,
    tasks: plan.tasks.map((task) => ({
      ...task,
      status:
        task.id === current?.id
          ? "completed"
          : task.id === next?.id
            ? "in_progress"
            : task.status,
    })),
  }
}

export function usePlanPreview() {
  const [scenario, setScenario] = React.useState<PlanScenario>("working")
  const [plan, setPlan] = React.useState<Plan | null>(() =>
    fixture("working", 1)
  )
  const [playing, setPlaying] = React.useState(false)
  const [message, setMessage] = React.useState("")
  const [editing, setEditing] = React.useState(false)
  const current = plan?.tasks.find((task) => task.status === "in_progress")
  const running = scenario !== "paused" && Boolean(current)

  React.useEffect(() => {
    if (!playing || !current) return
    const timer = setInterval(() => setPlan(advance), REPLAY_INTERVAL_MS)
    return () => clearInterval(timer)
  }, [playing, current])

  function select(next: PlanScenario) {
    setPlaying(false)
    setScenario(next)
    setPlan((previous) => fixture(next, (previous?.revision ?? 0) + 1))
    setMessage("")
  }

  return {
    editing,
    save: setPlan,
    onEditingChange: (next: boolean) => {
      setEditing(next)
      if (next) setPlaying(false)
    },
    scenario,
    plan,
    running,
    current,
    message,
    playing: playing && Boolean(current),
    select,
    next: () => setPlan(advance),
    pause: () => {
      setScenario("paused")
      setPlaying(false)
    },
    replay: () => {
      select("working")
      setPlaying(true)
    },
    send: (text: string) => setMessage(text),
  }
}
