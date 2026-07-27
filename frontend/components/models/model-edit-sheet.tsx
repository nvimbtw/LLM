"use client"

import * as React from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { toast } from "@/components/ui/toast"
import {
  Sheet,
  SheetClose,
  SheetContent,
  SheetDescription,
  SheetFooter,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"

export function ModelEditSheet({
  open,
  onOpenChange,
  initialName,
  initialDescription,
  initialFamily,
  families,
  onSave,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  initialName: string
  initialDescription: string
  initialFamily: string
  // Existing family names, offered as suggestions; typing a new one creates it.
  families: string[]
  onSave: (name: string, description: string, family: string) => Promise<void>
}) {
  const [name, setName] = React.useState(initialName)
  const [description, setDescription] = React.useState(initialDescription)
  const [family, setFamily] = React.useState(initialFamily)
  const [isSaving, setIsSaving] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)

  // Reset the form to the model's current values each time the sheet opens.
  React.useEffect(() => {
    if (open) {
      setName(initialName)
      setDescription(initialDescription)
      setFamily(initialFamily)
      setError(null)
    }
  }, [open, initialName, initialDescription, initialFamily])

  // The split, in one function: an empty field is a property of the form and
  // stays inline next to it, because you fix it here. A save that the server
  // refused is an event, and the sheet closes on success — so both halves of
  // that outcome are toasts, which outlive the sheet.
  async function handleSave() {
    const trimmedName = name.trim()
    const trimmedFamily = family.trim()
    if (!trimmedName) {
      setError("Name can't be empty.")
      return
    }
    if (!trimmedFamily) {
      setError("Family can't be empty.")
      return
    }
    setIsSaving(true)
    setError(null)
    try {
      await onSave(trimmedName, description.trim(), trimmedFamily)
      toast.success("Model updated.", trimmedName)
      onOpenChange(false)
    } catch (err) {
      toast.error(
        "Couldn't save changes",
        err instanceof Error ? err.message : "Something went wrong."
      )
    } finally {
      setIsSaving(false)
    }
  }

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="gap-0">
        <SheetHeader>
          <SheetTitle>Edit model</SheetTitle>
          <SheetDescription>
            Update the model&apos;s name and description. Architecture and
            training stats are fixed at train time.
          </SheetDescription>
        </SheetHeader>

        <div className="flex flex-col gap-4 px-4 py-2">
          <div className="flex flex-col gap-1.5">
            <label htmlFor="model-name" className="text-sm font-medium">
              Name
            </label>
            <Input
              id="model-name"
              value={name}
              onChange={(e) => setName(e.target.value)}
              disabled={isSaving}
            />
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="model-family" className="text-sm font-medium">
              Family
            </label>
            <Input
              id="model-family"
              value={family}
              onChange={(e) => setFamily(e.target.value)}
              disabled={isSaving}
              list="model-family-options"
              placeholder="Pick an existing family or type a new one"
              autoComplete="off"
            />
            <datalist id="model-family-options">
              {families.map((f) => (
                <option key={f} value={f} />
              ))}
            </datalist>
          </div>

          <div className="flex flex-col gap-1.5">
            <label htmlFor="model-description" className="text-sm font-medium">
              Description
            </label>
            <Textarea
              id="model-description"
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              rows={5}
              disabled={isSaving}
              placeholder="What is this model for?"
            />
          </div>

          {error && <p className="text-sm text-destructive">{error}</p>}
        </div>

        <SheetFooter className="flex-row justify-end">
          <SheetClose
            render={
              <Button variant="ghost" disabled={isSaving}>
                Cancel
              </Button>
            }
          />
          <Button onClick={handleSave} disabled={isSaving}>
            {isSaving ? "Saving…" : "Save changes"}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
