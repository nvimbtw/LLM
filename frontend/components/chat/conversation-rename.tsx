"use client"

import { useConversationRename } from "@/hooks/use-conversation-rename"
import { Input } from "@/components/ui/input"
import { Button } from "@/components/ui/button"
import { FormError } from "@/components/ui/error-state"

export interface ConversationRenameProps {
  title: string
  onSave: (title: string) => void | Promise<void>
  onDone: () => void
}

/** Renders the same explicit Save/Cancel title editor in the composer, archive and sidebar; blur preserves the attempt and Escape cancels. */
export function ConversationRename({
  title,
  onSave,
  onDone,
}: ConversationRenameProps) {
  const editor = useConversationRename(title, onSave, onDone)
  return (
    <div
      className="min-w-0 flex-1 space-y-2"
      onKeyDown={(event) => {
        if (event.key === "Escape" && !editor.pending) {
          event.preventDefault()
          event.stopPropagation()
          onDone()
        }
        if (event.key === "Enter" && !event.nativeEvent.isComposing) {
          event.preventDefault()
          event.stopPropagation()
          void editor.submit()
        }
      }}
    >
      <Input
        variant="inline"
        aria-label="Conversation title"
        autoFocus
        value={editor.value}
        disabled={editor.pending}
        onChange={(event) => editor.setValue(event.target.value)}
        className="text-sm"
      />
      <div className="flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={editor.pending}
          onClick={() => void editor.submit()}
        >
          {editor.pending ? "Saving…" : "Save changes"}
        </Button>
        <Button
          type="button"
          variant="text"
          size="sm"
          disabled={editor.pending}
          onClick={onDone}
        >
          Cancel
        </Button>
      </div>
      {editor.error && (
        <FormError className="text-xs">{editor.error}</FormError>
      )}
    </div>
  )
}
