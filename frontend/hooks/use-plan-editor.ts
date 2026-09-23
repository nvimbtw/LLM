"use client"

import * as React from "react"
import { toast } from "@/components/ui/toast"
import {
  addTask,
  moveTask,
  newPlan,
  orderedTasks,
  taskTitle,
  type Plan,
} from "@/lib/plan"

export interface PlanEditing {
  draft: Plan
  error: string | null
  rename: (id: string, title: string) => void
  remove: (id: string) => void
  move: (id: string, target: string) => void
  add: (title: string) => boolean
  save: () => boolean
  cancel: () => void
}

export function usePlanEditor(
  plan: Plan | null,
  onSave: (plan: Plan) => void,
  onEditingChange?: (editing: boolean) => void
) {
  const [draft, setDraft] = React.useState<Plan | null>(null)
  const [base, setBase] = React.useState<Plan | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  function commit(next: Plan) {
    const saved = { ...next, revision: (plan?.revision ?? 0) + 1 }
    onSave(saved)
    setError(null)
  }

  function report(error: unknown) {
    const message =
      error instanceof Error
        ? error.message
        : "Unable to save the plan in this browser."
    setError(message)
    toast.error("Plan unchanged", message)
  }

  function add(title: string): boolean {
    try {
      const next = addTask(draft ?? plan ?? newPlan(), title)
      if (draft) setDraft({ ...next, tasks: orderedTasks(next.tasks) })
      else commit(next)
      setError(null)
      return true
    } catch (error) {
      report(error)
      return false
    }
  }

  function edit() {
    if (draft) return
    setBase(plan)
    const source = plan ?? newPlan()
    setDraft({ ...source, tasks: orderedTasks(source.tasks) })
    setError(null)
    onEditingChange?.(true)
  }

  function close() {
    setDraft(null)
    setError(null)
    onEditingChange?.(false)
  }

  function save(): boolean {
    if (!draft) {
      toast({
        title: "Nothing to save",
        description: "Use /plan edit to make changes first.",
      })
      return false
    }
    try {
      if (plan?.id !== base?.id || plan?.revision !== base?.revision)
        throw new Error(
          "The plan changed while you were editing. Cancel and reopen the editor to use the latest plan."
        )
      commit({
        ...draft,
        tasks: draft.tasks.map((task) => ({
          ...task,
          title: taskTitle(task.title),
        })),
      })
      close()
      toast({ title: "Plan saved", type: "success" })
      return true
    } catch (error) {
      report(error)
      return false
    }
  }

  const editing: PlanEditing | null = draft
    ? {
        draft,
        error,
        add,
        save,
        cancel: close,
        rename: (id, title) =>
          setDraft({
            ...draft,
            tasks: draft.tasks.map((task) =>
              task.id === id ? { ...task, title } : task
            ),
          }),
        remove: (id) =>
          setDraft({
            ...draft,
            tasks: draft.tasks.filter((task) => task.id !== id),
          }),
        move: (id, target) => setDraft(moveTask(draft, id, target)),
      }
    : null
  return { editing, add, edit, save }
}
