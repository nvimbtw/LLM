"use client"

import { ChatComposer } from "@/components/chat/chat-composer"
import { Trace } from "@/components/chat/trace"
import { ToolLabel } from "@/components/chat/tool-label"
import { Button } from "@/components/ui/button"
import { PLAN_SCENARIOS, usePlanPreview } from "@/hooks/use-plan-preview"

const CALL = JSON.stringify({ action: "complete", id: "t2" })
const WORKING = `<|tool-call|>${JSON.stringify({ tool: "todo", argument: CALL })}<|/tool-call|>`
const TOOL_STATES = [
  { label: "Updating", body: null },
  {
    label: "Saved",
    body: "<|par|>Completed t2: Trace the current tool pipeline. Next: Design the task list above the composer.<|/par|>",
  },
  {
    label: "Needs attention",
    body: "<|warn|>Task t2 could not be updated. The plan has not changed.<|/warn|>",
  },
]

export function PlanShowcase() {
  const preview = usePlanPreview()

  return (
    <section
      id="composer-plan"
      className="mt-8 space-y-6 border-t border-border/60 pt-6"
    >
      <div className="space-y-3">
        <p className="eyebrow text-muted-foreground">
          The composer · a plan in motion
        </p>
        <h2 className="font-display text-xl text-foreground">
          A little room for what comes next.
        </h2>
        <p className="max-w-prose text-sm leading-relaxed text-muted-foreground">
          Three tasks in view. Hover to unfold the whole plan; click to keep it
          open. Complete a step to watch it settle into the finished work below.
        </p>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3">
        <div
          className="flex flex-wrap gap-1"
          role="group"
          aria-label="Plan preview state"
        >
          {PLAN_SCENARIOS.map((scenario) => (
            <Button
              key={scenario.id}
              size="sm"
              variant={preview.scenario === scenario.id ? "secondary" : "ghost"}
              aria-pressed={preview.scenario === scenario.id}
              onClick={() => preview.select(scenario.id)}
              disabled={preview.editing}
            >
              {scenario.label}
            </Button>
          ))}
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={preview.next}
            disabled={!preview.current || preview.playing || preview.editing}
          >
            Complete next
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={preview.playing ? preview.pause : preview.replay}
            disabled={preview.editing}
          >
            {preview.playing ? "Pause replay" : "Replay"}
          </Button>
        </div>
      </div>

      <div className="border-y border-border/60 pb-5">
        <div className="mx-auto flex min-h-96 max-w-2xl flex-col justify-end pt-12">
          <div className="mb-10 max-w-lg space-y-3 px-3">
            <p className="font-display text-xl text-foreground">
              The work stays in view.
            </p>
            <p className="text-sm leading-relaxed text-muted-foreground">
              I’ll start with the composer and its task list, then check how it
              feels as work finishes and the next step moves into place.
            </p>
            {preview.running && (
              <div className="pt-2 text-xs">
                <ToolLabel
                  call={{
                    tool: "todo",
                    argument: JSON.stringify({
                      action: "update",
                      id: preview.current?.id,
                      title: preview.current?.title,
                    }),
                  }}
                />
              </div>
            )}
            {preview.message && (
              <p className="text-sm text-muted-foreground" role="status">
                Preview message: {preview.message}
              </p>
            )}
          </div>
          <ChatComposer
            plan={preview.plan}
            onPlanChange={preview.save}
            onPlanEditingChange={preview.onEditingChange}
            sending={preview.running}
            onCancel={preview.pause}
            onSend={preview.send}
            placeholder="Try the composer, or type /…"
          />
        </div>
      </div>
      <p className="text-xs text-muted-foreground">
        Local preview · sending here stays on this page. No model is running.
        Try keyboard focus, a narrow window, and both themes.
      </p>
      <p className="text-xs text-muted-foreground">
        Try <code>/plan add Check the details</code>, <code>/plan edit</code>,
        and <code>/plan save</code>. Editing stays in a draft until you save;
        Cancel restores the previous plan.
      </p>

      <div className="space-y-6">
        <div className="space-y-2">
          <p className="eyebrow text-muted-foreground">The tool call</p>
          <p className="text-sm text-muted-foreground">
            The same call while it runs, after it saves, and when an update
            fails.
          </p>
        </div>
        {TOOL_STATES.map((state) => (
          <div
            key={state.label}
            className="grid gap-3 border-t border-border pt-5 sm:grid-cols-[8rem_1fr]"
          >
            <p className="tag text-muted-foreground">{state.label}</p>
            <Trace
              working={WORKING}
              answers={
                state.body === null
                  ? []
                  : [{ tool: "todo", argument: CALL, body: state.body }]
              }
              streaming={state.body === null}
            />
          </div>
        ))}
      </div>
    </section>
  )
}
