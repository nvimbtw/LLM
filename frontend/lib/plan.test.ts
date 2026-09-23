import { test } from "node:test"
import assert from "node:assert/strict"
import {
  addTask,
  moveTask,
  parsePlan,
  taskTitle,
  MAX_PLAN_TASKS,
  type Plan,
} from "./plan"

const original: Plan = {
  id: "test-plan",
  title: "Plan",
  revision: 3,
  tasks: [
    { id: "a", title: "Current", status: "in_progress" },
    { id: "b", title: "Second", status: "pending" },
    { id: "c", title: "Third", status: "pending" },
    { id: "d", title: "Finished", status: "completed" },
  ],
}

test("reordering preserves task identity, status, and the original snapshot", () => {
  const moved = moveTask(original, "c", "b")
  assert.deepEqual(
    moved.tasks.map((task) => task.id),
    ["a", "c", "b", "d"]
  )
  assert.deepEqual(
    original.tasks.map((task) => task.id),
    ["a", "b", "c", "d"]
  )
  assert.equal(moved.tasks[1], original.tasks[2])
  assert.equal(moveTask(original, "c", "missing"), original)
})

test("current and finished tasks keep their display groups", () => {
  assert.equal(moveTask(original, "a", "b"), original)
  assert.equal(moveTask(original, "b", "d"), original)
})

test("new tasks have stable unique ids and validated titles", () => {
  const first = addTask(original, "  Check the result  ")
  const second = addTask(first, "Check again")
  assert.equal(first.tasks[4].title, "Check the result")
  assert.equal(first.tasks[4].status, "pending")
  assert.notEqual(first.tasks[4].id, second.tasks[5].id)
  assert.throws(() => taskTitle("   "))
  assert.throws(() => taskTitle("x".repeat(241)))
  assert.throws(() =>
    addTask(
      { ...original, tasks: Array(MAX_PLAN_TASKS).fill(original.tasks[0]) },
      "Too many"
    )
  )
})

test("browser snapshots reject malformed tasks and duplicate ids", () => {
  assert.deepEqual(parsePlan(JSON.stringify(original)), original)
  assert.equal(parsePlan("{"), null)
  assert.equal(
    parsePlan(
      JSON.stringify({
        ...original,
        tasks: [original.tasks[0], original.tasks[0]],
      })
    ),
    null
  )
  assert.equal(
    parsePlan(
      JSON.stringify({
        ...original,
        tasks: [{ ...original.tasks[0], status: "invented" }],
      })
    ),
    null
  )
  assert.equal(
    parsePlan(
      JSON.stringify({
        ...original,
        tasks: [{ ...original.tasks[0], title: "" }],
      })
    ),
    null
  )
})
