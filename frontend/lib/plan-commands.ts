import { toast } from "@/components/ui/toast"

export interface PlanCommands {
  add: (title: string) => boolean
  edit: () => void
  save: () => boolean
}

export interface PlanCommandOption {
  value: string
  label: string
  description: string
  execute: boolean
}

export function planOptions(query: string): PlanCommandOption[] {
  if (/^add\s+\S/i.test(query))
    return [
      {
        value: query,
        label: "Add task",
        description: query.slice(4),
        execute: true,
      },
    ]
  return [
    {
      value: "add",
      label: "Add a task",
      description: "/plan add <task>",
      execute: false,
    },
    {
      value: "edit",
      label: "Edit the plan",
      description: "Reorder, rename, add or remove tasks",
      execute: true,
    },
    {
      value: "save",
      label: "Save changes",
      description: "Apply your draft and close the editor",
      execute: true,
    },
  ].filter((option) => option.value.startsWith(query.toLowerCase()))
}

export function runPlanCommand(
  argument: string,
  commands?: PlanCommands
): void {
  if (!commands) {
    toast.error("Plan unavailable", "Open a composer to change its plan.")
    return
  }
  const match = /^(add|edit|save)(?:\s+([\s\S]*))?$/i.exec(argument)
  if (!match) {
    toast.error(
      "Plan commands",
      "Use /plan add <task>, /plan edit, or /plan save."
    )
    return
  }
  const [, action, title] = match
  if (action.toLowerCase() === "add") {
    if (commands.add(title ?? ""))
      toast({ title: "Task added", type: "success" })
  } else if (title) {
    toast.error(
      "Unexpected argument",
      `/plan ${action.toLowerCase()} takes no additional text.`
    )
  } else if (action.toLowerCase() === "edit") commands.edit()
  else commands.save()
}
