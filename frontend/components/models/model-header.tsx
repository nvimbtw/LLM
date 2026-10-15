"use client"

import Link from "next/link"
import type { ReactNode } from "react"
import { Button } from "@/components/ui/button"
import { FormError } from "@/components/ui/error-state"
import { Input } from "@/components/ui/input"
import { PageHeader } from "@/components/ui/page-header"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import type { ModelDetail, ModelEdits } from "@/hooks/use-model"
import { useModelEditor } from "@/hooks/use-model-editor"

export interface ModelHeaderProps {
  model: ModelDetail
  editable: boolean
  busy: boolean
  families: string[]
  onSave: (fields: ModelEdits) => Promise<void>
  actions: (saving: boolean) => ReactNode
}

const SIZE_LABELS = {
  unset: "Unassigned",
  fast: "Fast",
  balanced: "Balanced",
  flagship: "Flagship",
}
const INLINE_FIELD =
  "h-auto rounded-none border-0 bg-transparent px-0 py-0 shadow-none focus-visible:ring-0 focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-ring/50 dark:bg-transparent"

/** Renders model identity as editable page fields for admins and ordinary text for readers. Saves are explicit and failed drafts remain visible. */
export function ModelHeader({
  model,
  editable,
  busy,
  families,
  onSave,
  actions,
}: ModelHeaderProps) {
  const editor = useModelEditor(model, onSave)
  const created = model.createdAt
    ? new Date(model.createdAt).toLocaleDateString("en-US", {
        year: "numeric",
        month: "short",
        day: "numeric",
      })
    : null
  return (
    <form
      onSubmit={(event) => {
        event.preventDefault()
        if (editable && !busy) void editor.submit()
      }}
    >
      <PageHeader
        back={{ href: "/models", label: "Models" }}
        className="mb-6 border-b-0 pb-0"
        contentClassName="min-w-48 flex-1"
        title={
          editable ? (
            <Input
              aria-label="Model name"
              required
              maxLength={240}
              value={editor.fields.display_name}
              disabled={editor.saving || busy}
              onChange={(event) =>
                editor.change("display_name", event.target.value)
              }
              className={`${INLINE_FIELD} w-full font-display text-3xl font-semibold tracking-tight md:text-3xl`}
            />
          ) : (
            model.name
          )
        }
        description={
          <>
            {editable ? (
              <Textarea
                aria-label="Model description"
                rows={1}
                placeholder="Add a description…"
                value={editor.fields.description}
                disabled={editor.saving || busy}
                onChange={(event) =>
                  editor.change("description", event.target.value)
                }
                className={`${INLINE_FIELD} min-h-5 resize-none text-sm leading-relaxed text-muted-foreground`}
              />
            ) : (
              model.description && <p>{model.description}</p>
            )}
            <div className="flex flex-wrap items-center gap-x-5 gap-y-3 text-xs">
              {editable ? (
                <label className="flex items-baseline gap-2">
                  <span>Family</span>
                  <Input
                    aria-label="Model family"
                    list="model-families"
                    required
                    maxLength={240}
                    value={editor.fields.family}
                    disabled={editor.saving || busy}
                    onChange={(event) =>
                      editor.change("family", event.target.value)
                    }
                    className={`${INLINE_FIELD} w-28 text-xs text-foreground md:text-xs`}
                  />
                  <datalist id="model-families">
                    {families.map((family) => (
                      <option key={family} value={family} />
                    ))}
                  </datalist>
                </label>
              ) : (
                <span>{model.family}</span>
              )}
              {editable ? (
                <div className="flex items-center gap-2">
                  <span id="model-size-label">Size</span>
                  <Select
                    value={editor.fields.size ?? "unset"}
                    onValueChange={(value) => {
                      if (
                        value === "unset" ||
                        value === "fast" ||
                        value === "balanced" ||
                        value === "flagship"
                      )
                        editor.change("size", value === "unset" ? null : value)
                    }}
                    disabled={editor.saving || busy}
                  >
                    <SelectTrigger
                      aria-labelledby="model-size-label"
                      variant="inline"
                      size="sm"
                      className="w-auto gap-2 text-xs"
                    >
                      <SelectValue>
                        {(value: unknown) =>
                          value === "fast" ||
                          value === "balanced" ||
                          value === "flagship"
                            ? SIZE_LABELS[value]
                            : "Unassigned"
                        }
                      </SelectValue>
                    </SelectTrigger>
                    <SelectContent>
                      {Object.entries(SIZE_LABELS).map(([value, label]) => (
                        <SelectItem key={value} value={value}>
                          {label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              ) : (
                model.tier && (
                  <span className="capitalize italic">{model.tier}</span>
                )
              )}
              <span>{model.published ? "Published" : "Private"}</span>
              {created && <span>Created {created}</span>}
            </div>
            {model.checkpointIndex !== null && (
              <p className="text-xs">
                Checkpoint #{model.checkpointIndex}
                {model.checkpointStep !== null &&
                  ` · step ${model.checkpointStep.toLocaleString()}`}
                {model.parentId && (
                  <>
                    {" "}
                    ·{" "}
                    <Link
                      href={`/models/${model.parentId}`}
                      className="underline underline-offset-4"
                    >
                      Finished model
                    </Link>
                  </>
                )}
              </p>
            )}
          </>
        }
        action={
          editable && (
            <div className="flex flex-wrap items-center gap-2">
              {editor.hasChanges && (
                <>
                  <Button
                    type="submit"
                    size="sm"
                    variant="outline"
                    disabled={editor.saving || busy}
                  >
                    {editor.saving ? "Saving…" : "Save changes"}
                  </Button>
                  <Button
                    type="button"
                    size="sm"
                    variant="text"
                    disabled={editor.saving || busy}
                    onClick={editor.reset}
                  >
                    Cancel
                  </Button>
                </>
              )}
              {actions(editor.saving)}
            </div>
          )
        }
      />
      {editor.error && <FormError className="mb-6">{editor.error}</FormError>}
    </form>
  )
}
