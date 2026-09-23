export type TaskStatus = "pending" | "in_progress" | "completed" | "cancelled"

export interface PlanTask {
  id: string
  title: string
  status: TaskStatus
}

export interface Plan {
  id: string
  title: string
  revision: number
  tasks: PlanTask[]
}

export const MAX_PLAN_TASKS = 100
export const MAX_TASK_TITLE = 240

export function newPlan(): Plan {
  return { id: crypto.randomUUID(), title: "Your plan", revision: 0, tasks: [] }
}

export function taskTitle(value: string): string {
  const title = value.trim()
  if (!title) throw new Error("Give the task a title.")
  if (title.length > MAX_TASK_TITLE)
    throw new Error(`Keep task titles under ${MAX_TASK_TITLE + 1} characters.`)
  return title
}

export function addTask(plan: Plan, title: string): Plan {
  if (plan.tasks.length >= MAX_PLAN_TASKS)
    throw new Error(`A plan can hold up to ${MAX_PLAN_TASKS} tasks.`)
  return {
    ...plan,
    tasks: [
      ...plan.tasks,
      { id: crypto.randomUUID(), title: taskTitle(title), status: "pending" },
    ],
  }
}

export function moveTask(plan: Plan, id: string, target: string): Plan {
  const from = plan.tasks.findIndex((task) => task.id === id)
  const to = plan.tasks.findIndex((task) => task.id === target)
  if (from < 0 || to < 0 || from === to) return plan
  if (taskGroup(plan.tasks[from]) !== taskGroup(plan.tasks[to])) return plan
  const tasks = [...plan.tasks]
  const [task] = tasks.splice(from, 1)
  tasks.splice(to, 0, task)
  return { ...plan, tasks }
}

export function parsePlan(raw: string | null): Plan | null {
  if (!raw) return null
  try {
    const value: unknown = JSON.parse(raw)
    if (!value || typeof value !== "object") return null
    const plan = value as Record<string, unknown>
    if (
      typeof plan.id !== "string" ||
      typeof plan.title !== "string" ||
      !Number.isInteger(plan.revision) ||
      !Array.isArray(plan.tasks) ||
      plan.tasks.length > MAX_PLAN_TASKS
    )
      return null
    const ids = new Set<string>()
    for (const value of plan.tasks) {
      if (!value || typeof value !== "object") return null
      const task = value as Record<string, unknown>
      if (
        typeof task.id !== "string" ||
        ids.has(task.id) ||
        typeof task.title !== "string" ||
        !task.title.trim() ||
        task.title.length > MAX_TASK_TITLE ||
        !["pending", "in_progress", "completed", "cancelled"].includes(
          String(task.status)
        )
      )
        return null
      ids.add(task.id)
    }
    return plan as unknown as Plan
  } catch {
    return null
  }
}

export function unfinished(task: PlanTask): boolean {
  return task.status === "pending" || task.status === "in_progress"
}

export function orderedTasks(tasks: PlanTask[]): PlanTask[] {
  return [...tasks].sort((a, b) => taskGroup(a) - taskGroup(b))
}

export function taskGroup(task: PlanTask): number {
  return task.status === "in_progress" ? 0 : unfinished(task) ? 1 : 2
}
