"use client"

import * as React from "react"
import { Button } from "@/components/ui/button"
import { orderedTasks, unfinished, type Plan } from "@/lib/plan"
import { PlanTask } from "./plan-task"
import { PlanEditor } from "./plan-editor"
import type { PlanEditing } from "@/hooks/use-plan-editor"
import "./composer-plan.css"

const PREVIEW_TASKS = 3
const COMPLETE_BEAT_MS = 760
const HOVER_DELAY_MS = 180

export interface ComposerPlanProps {
  plan: Plan
  running?: boolean
  suppressed?: boolean
  editing?: PlanEditing | null
}

/** Presents ordered local tasks and revision-safe editing within the composer; disclosure never changes task status. */
export function ComposerPlan({
  plan,
  running = false,
  suppressed = false,
  editing,
}: ComposerPlanProps) {
  const [hovered, setHovered] = React.useState(false)
  const [pinned, setPinned] = React.useState(false)
  const [focused, setFocused] = React.useState(false)
  const [dismissed, setDismissed] = React.useState(false)
  const [previous, setPrevious] = React.useState(plan)
  const [retiring, setRetiring] = React.useState<string[]>([])
  const hoverTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null)
  const regionId = React.useId()

  if (previous !== plan) {
    setPrevious(plan)
    setRetiring(
      previous.id === plan.id
        ? plan.tasks
            .filter(
              (task) =>
                task.status === "completed" &&
                previous.tasks.some(
                  (old) => old.id === task.id && unfinished(old)
                )
            )
            .map((task) => task.id)
        : []
    )
  }

  React.useEffect(() => {
    if (!retiring.length) return
    const timer = setTimeout(() => setRetiring([]), COMPLETE_BEAT_MS)
    return () => clearTimeout(timer)
  }, [retiring])

  React.useEffect(
    () => () => {
      if (hoverTimer.current) clearTimeout(hoverTimer.current)
    },
    []
  )

  const expanded =
    !suppressed &&
    (Boolean(editing) || (!dismissed && (hovered || pinned || focused)))
  const tasks = orderedTasks(plan.tasks)
  const pending = tasks.filter(unfinished)
  const completed = tasks.filter((task) => task.status === "completed").length
  const complete = tasks.length > 0 && completed === tasks.length
  const settled = pending.length === 0
  const preview = [
    ...tasks.filter((task) => retiring.includes(task.id)),
    ...pending,
  ].slice(0, PREVIEW_TASKS)
  const label = editing
    ? "Editing plan"
    : complete
      ? "Plan complete"
      : settled
        ? "Plan finished"
        : "Plan"

  /** Releases hover disclosure and its pending timer when the pointer leaves. */
  function leave() {
    if (hoverTimer.current) clearTimeout(hoverTimer.current)
    setHovered(false)
    setDismissed(false)
  }

  return (
    <section
      className="composer-plan"
      aria-label={plan.title}
      data-expanded={expanded}
      data-suppressed={suppressed}
      data-editing={Boolean(editing)}
      onPointerEnter={(event) => {
        if (event.pointerType !== "mouse") return
        hoverTimer.current = setTimeout(() => setHovered(true), HOVER_DELAY_MS)
      }}
      onPointerLeave={leave}
      onBlur={(event) => {
        if (!event.currentTarget.contains(event.relatedTarget)) {
          setFocused(false)
          setDismissed(false)
        }
      }}
      onKeyDown={(event) => {
        if (event.key === "Escape") {
          event.stopPropagation()
          if (editing) return
          setPinned(false)
          setDismissed(true)
        }
      }}
    >
      <div className="plan-preview" aria-hidden={expanded} inert={expanded}>
        <div className="plan-heading tag">
          <span>{label}</span>
          <span className="plan-leader" />
          <span>
            {completed} / {tasks.length}
          </span>
        </div>
        <ol
          className="plan-rows"
          style={{
            height: suppressed
              ? 0
              : `calc(var(--plan-row-height) * ${preview.length})`,
          }}
        >
          {preview.map((task, index) => (
            <PlanTask
              key={task.id}
              task={task}
              index={index}
              retiring={retiring.includes(task.id)}
              running={running}
            />
          ))}
        </ol>
      </div>
      <div
        className="plan-disclosure"
        id={regionId}
        inert={!expanded}
        aria-hidden={!expanded}
      >
        <div className="plan-heading tag">
          <span>{label}</span>
          <span className="plan-leader" />
          <span>
            {completed} / {tasks.length}
          </span>
        </div>
        {editing ? (
          <PlanEditor editing={editing} />
        ) : (
          <>
            <p className="plan-caption">{plan.title}</p>
            <div className="plan-scroll">
              <ol className="plan-rows">
                {tasks.map((task, index) => (
                  <PlanTask
                    key={task.id}
                    task={task}
                    index={index}
                    running={running}
                  />
                ))}
              </ol>
            </div>
            <p className="plan-hint tag">
              {pinned ? "Pinned open · click to fold" : "Click to keep open"}
            </p>
          </>
        )}
      </div>
      {!editing && (
        <Button
          variant="ghost"
          className="plan-toggle"
          aria-label={`${expanded ? "Fold" : "Expand"} plan: ${plan.title}, ${completed} of ${tasks.length} completed`}
          aria-expanded={expanded}
          aria-controls={regionId}
          disabled={suppressed}
          onFocus={(event) => {
            if (event.currentTarget.matches(":focus-visible")) setFocused(true)
          }}
          onClick={() => {
            if (pinned) {
              setPinned(false)
              setDismissed(true)
            } else {
              setPinned(true)
              setDismissed(false)
            }
          }}
        />
      )}
      <span className="sr-only" role="status" aria-live="polite">
        {label}. {completed} of {tasks.length} completed.
        {pending[0] ? ` Next: ${pending[0].title}.` : ""}
      </span>
    </section>
  )
}
