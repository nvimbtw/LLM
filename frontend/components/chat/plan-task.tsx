import type { CSSProperties } from "react"
import type { PlanTask as Task } from "@/lib/plan"
import { cn } from "@/lib/utils"

export interface PlanTaskProps {
  task: Task
  index: number
  retiring?: boolean
  running: boolean
}

const STATUS_LABELS = {
  pending: "Pending",
  in_progress: "In progress",
  completed: "Completed",
  cancelled: "Cancelled",
}

export function PlanTask({ task, index, retiring, running }: PlanTaskProps) {
  return (
    <li
      className={cn("plan-task", retiring && "plan-task-retiring")}
      data-status={task.status}
      data-running={running}
      style={{ "--plan-index": index } as CSSProperties}
      title={`${STATUS_LABELS[task.status]}: ${task.title}`}
    >
      <span className="plan-task-mark" aria-hidden="true">
        {task.status === "completed"
          ? "✓"
          : task.status === "cancelled"
            ? "−"
            : ""}
      </span>
      <span className="sr-only">{STATUS_LABELS[task.status]}: </span>
      <span className="plan-task-title">{task.title}</span>
      {task.status === "in_progress" && (
        <span className="plan-task-now tag">{running ? "now" : "paused"}</span>
      )}
    </li>
  )
}
