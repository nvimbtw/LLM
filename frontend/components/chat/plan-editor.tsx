"use client"

import * as React from "react"
import { GripVertical, Plus, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { FormError } from "@/components/ui/error-state"
import type { PlanEditing } from "@/hooks/use-plan-editor"
import { MAX_PLAN_TASKS, MAX_TASK_TITLE, taskGroup } from "@/lib/plan"
import { cn } from "@/lib/utils"

export interface PlanEditorProps {
  editing: PlanEditing
}

export function PlanEditor({ editing }: PlanEditorProps) {
  const [title, setTitle] = React.useState("")
  const [dragging, setDragging] = React.useState<string | null>(null)
  const [target, setTarget] = React.useState<string | null>(null)
  const [announcement, setAnnouncement] = React.useState("")
  const list = React.useRef<HTMLOListElement>(null)
  const hint = React.useId()
  const { draft } = editing

  function add() {
    if (editing.add(title)) setTitle("")
  }

  function finish() {
    if (dragging && target && dragging !== target) {
      editing.move(dragging, target)
      setAnnouncement("Task moved. Save to apply the new order.")
    }
    setDragging(null)
    setTarget(null)
  }

  return (
    <div className="plan-editor" aria-label="Edit plan">
      <p id={hint} className="plan-edit-hint">
        Drag to reorder · current stays first, finished stays last
        <span className="sr-only">
          {" "}
          Focus a handle and use arrow keys to reorder.
        </span>
      </p>
      <div className="plan-scroll">
        <ol ref={list} className="space-y-1 py-2">
          {draft.tasks.map((task, index) => (
            <li
              key={task.id}
              data-plan-task={task.id}
              className={cn(
                "plan-edit-row group/row flex items-center gap-1 rounded-(--radius-control-sm)",
                dragging === task.id && "opacity-50",
                target === task.id &&
                  dragging !== task.id &&
                  "bg-accent ring-1 ring-ring"
              )}
            >
              <Button
                variant="ghost"
                size="icon-sm"
                className="plan-drag-handle cursor-grab touch-none text-muted-foreground active:cursor-grabbing"
                aria-label={`Move task ${index + 1}`}
                aria-describedby={hint}
                disabled={task.status === "in_progress"}
                onPointerDown={(event) => {
                  if (event.button !== 0) return
                  event.currentTarget.setPointerCapture(event.pointerId)
                  setDragging(task.id)
                }}
                onPointerMove={(event) => {
                  if (!dragging) return
                  const row = document
                    .elementFromPoint(event.clientX, event.clientY)
                    ?.closest<HTMLElement>("[data-plan-task]")
                  const destination = draft.tasks.find(
                    (item) => item.id === row?.dataset.planTask
                  )
                  setTarget(
                    row &&
                      destination &&
                      taskGroup(destination) === taskGroup(task) &&
                      list.current?.contains(row)
                      ? (row.dataset.planTask ?? null)
                      : null
                  )
                }}
                onPointerUp={finish}
                onPointerCancel={() => {
                  setDragging(null)
                  setTarget(null)
                }}
                onKeyDown={(event) => {
                  const offset =
                    event.key === "ArrowUp"
                      ? -1
                      : event.key === "ArrowDown"
                        ? 1
                        : 0
                  if (!offset) return
                  event.preventDefault()
                  const destination = draft.tasks[index + offset]
                  if (
                    destination &&
                    taskGroup(destination) === taskGroup(task)
                  ) {
                    editing.move(task.id, destination.id)
                    setAnnouncement(
                      `Task moved to position ${index + offset + 1}.`
                    )
                  }
                }}
              >
                <GripVertical />
              </Button>
              <Input
                value={task.title}
                maxLength={MAX_TASK_TITLE}
                aria-label={`Task ${index + 1} title`}
                className="plan-title-input"
                onChange={(event) =>
                  editing.rename(task.id, event.target.value)
                }
              />
              <span className="plan-edit-status shrink-0 tag text-muted-foreground">
                {task.status === "in_progress"
                  ? "current"
                  : task.status === "completed"
                    ? "done"
                    : task.status === "cancelled"
                      ? "cancelled"
                      : ""}
              </span>
              <Button
                variant="ghost"
                size="icon-sm"
                aria-label={`Remove task ${index + 1}`}
                className="plan-remove"
                onClick={() => editing.remove(task.id)}
              >
                <X />
              </Button>
            </li>
          ))}
        </ol>
        {!draft.tasks.length && (
          <p className="py-3 text-xs text-muted-foreground">
            An empty page. Add the first task below.
          </p>
        )}
      </div>
      <form
        className="plan-add mt-3 flex gap-2"
        onSubmit={(event) => {
          event.preventDefault()
          add()
        }}
      >
        <Input
          value={title}
          onChange={(event) => setTitle(event.target.value)}
          placeholder="Add a task…"
          aria-label="New task title"
          maxLength={MAX_TASK_TITLE}
          className="plan-title-input"
        />
        <Button
          type="submit"
          size="icon"
          variant="ghost"
          aria-label="Add task"
          disabled={!title.trim() || draft.tasks.length >= MAX_PLAN_TASKS}
        >
          <Plus />
        </Button>
      </form>
      {editing.error && <FormError>{editing.error}</FormError>}
      <div className="plan-hint flex flex-wrap items-center justify-between gap-2">
        <span className="tag text-muted-foreground">
          Draft · {draft.tasks.length} tasks
        </span>
        <div className="flex gap-2">
          <Button
            size="sm"
            variant="ghost"
            className="text-muted-foreground"
            onClick={editing.cancel}
          >
            Cancel
          </Button>
          <Button
            size="sm"
            variant="ghost"
            className="text-primary"
            onClick={editing.save}
          >
            Save plan
          </Button>
        </div>
      </div>
      <p className="sr-only" role="status">
        {announcement}
      </p>
    </div>
  )
}
