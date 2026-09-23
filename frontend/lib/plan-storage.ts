import type { Plan } from "./plan"

export const PLAN_CHANGED = "plan-changed"
const PREFIX = "slate-plan:"

export function readPlan(chatId: string): string | null {
  try {
    return localStorage.getItem(PREFIX + chatId)
  } catch {
    return null
  }
}

export function writePlan(chatId: string, plan: Plan): void {
  localStorage.setItem(PREFIX + chatId, JSON.stringify(plan))
  window.dispatchEvent(new Event(PLAN_CHANGED))
}
