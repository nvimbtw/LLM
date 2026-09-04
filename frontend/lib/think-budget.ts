/**
 * How long the model should reason before answering.
 *
 * `auto` is the absence of a choice, not a fourth level: it sends no budget at
 * all, so the model answers the way it did before budgets existed. The three
 * named levels are the ones the SFT corpus is labelled with — the wire values
 * match the labels in `raw/sft/think-*.jsonl` exactly, because the string the
 * composer sends becomes the marker the model was trained under.
 *
 * Shared rather than owned by one selector component: the chat socket, the
 * chats provider, the composer's own choice, and both places a budget is
 * picked (the composer's consolidated picker, `/thinking`'s argument list in
 * `lib/commands.ts`) all need the same four values and the same copy.
 */
export type ThinkBudget = "auto" | "low" | "med" | "high"

export interface ThinkBudgetOption {
  value: ThinkBudget
  label: string
  hint: string
}

export const THINK_BUDGETS: ThinkBudgetOption[] = [
  { value: "auto", label: "Auto", hint: "Let the model decide" },
  { value: "low", label: "Low", hint: "Answer straight away" },
  { value: "med", label: "Medium", hint: "Weigh one alternative" },
  { value: "high", label: "High", hint: "Work it through properly" },
]
