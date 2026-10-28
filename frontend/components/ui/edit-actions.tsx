import { Button } from "./button"

export interface EditActionsProps {
  dirty: boolean
  busy: boolean
  onCancel: () => void
  disabled?: boolean
}

/** Renders consistent unsaved/saving feedback and explicit submit/cancel controls for page-native resource fields. */
export function EditActions({
  dirty,
  busy,
  onCancel,
  disabled,
}: EditActionsProps) {
  if (!dirty) return null
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span role="status" className="mr-1 text-xs text-muted-foreground">
        {busy ? "Saving…" : "Unsaved changes"}
      </span>
      <Button
        type="submit"
        size="sm"
        variant="outline"
        disabled={busy || disabled}
      >
        Save changes
      </Button>
      <Button
        type="button"
        size="sm"
        variant="text"
        disabled={busy}
        onClick={onCancel}
      >
        Cancel
      </Button>
    </div>
  )
}
