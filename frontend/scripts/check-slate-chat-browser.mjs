import { chromium } from "playwright"
import assert from "node:assert/strict"

const origin = process.env.SLATE_URL ?? "http://localhost:3000"
const account = "30000000-0000-4000-8000-000000000001"
const chatId = "30000000-0000-4000-8000-000000000002"
const modelId = "30000000-0000-4000-8000-000000000003"
const key = `slate-notifications:v1:${account}`
const chat = {
  chat_id: chatId,
  title: "Fluid rail fixture",
  created_at: "2026-10-28T12:00:00Z",
  tokens: 20,
  project_id: null,
}
const rows = Array.from({ length: 120 }, (_, i) => [
  {
    message_id: `user-${i}`,
    role: "user",
    content: `Question ${i}: explain the composer.`,
    attachments: [],
  },
  {
    message_id: `reply-${i}`,
    role: "assistant",
    content: `<|par|>Reply ${i}: the composer keeps commands and plans inside its outline.`,
    attachments: [],
  },
]).flat()
const model = {
  model_id: modelId,
  name: "Published fixture",
  family: "Research",
  published: true,
  has_weights: true,
  parameters: null,
  context: 512,
  architecture: null,
  vocab_size: null,
  parent_id: null,
  description: null,
}
const errors = []
let usageReads = 0
let failUsage = true
const browser = await chromium.launch({
  headless: true,
  executablePath: process.env.CHROMIUM_PATH ?? "/home/x_x/.local/bin/chromium",
})
const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } })
page.setDefaultTimeout(60000)
page.setDefaultNavigationTimeout(60000)
page.on("pageerror", (error) => errors.push(error.message))
await page.route("http://localhost:8080/**", async (route) => {
  const path = new URL(route.request().url()).pathname
  const json = (value) => route.fulfill({ json: value })
  if (path === "/user/me")
    return json({ id_user: account, username: "agents", access_level: "admin" })
  if (path === "/user/me/settings") return json({ settings: {} })
  if (path === "/user/me/composer-choice")
    return json({ composer_choice: { model: modelId, budget: "auto" } })
  if (path === "/models")
    return json({
      models: [
        model,
        {
          ...model,
          model_id: "private-model",
          name: "Private fixture",
          published: false,
        },
      ],
    })
  if (path === "/chats") return json({ chats: [chat] })
  if (path === `/chats/${chatId}`)
    return json({ ...chat, messages: rows, total_tokens: 20 })
  if (path === "/runs")
    return json({
      active: [],
      history: [],
      history_total: 0,
      history_offset: 0,
      history_limit: 25,
      paused: false,
    })
  if (path === "/projects") return json({ projects: [] })
  if (path === "/tokenize/count") return json({ count: 5 })
  if (path === "/usage") {
    usageReads += 1
    return failUsage
      ? route.fulfill({ status: 503, body: "Usage temporarily unavailable" })
      : json({ series: [], by_model: [] })
  }
  return json({})
})
await page.addInitScript(
  ({ key }) => {
    localStorage.setItem("auth_token", "opaque-browser-fixture")
    localStorage.setItem("access_level", "admin")
    if (!localStorage.getItem(key))
      localStorage.setItem(
        key,
        JSON.stringify({
          version: 1,
          items: [
            {
              id: "saved-timed",
              title: "Saved notice",
              description: "Retained after expiry",
              severity: "default",
              lifetime: "timed",
              createdAt: 1000,
            },
            {
              id: "saved-error",
              title: "Saved error",
              description: "Needs attention",
              severity: "error",
              lifetime: "sticky",
              createdAt: 1001,
            },
          ],
        })
      )
  },
  { key }
)
try {
  await page.goto(`${origin}/chat/${chatId}`, { waitUntil: "domcontentloaded" })
  const editor = page.locator('[contenteditable="true"]')
  await editor.waitFor()
  await page.getByRole("button", { name: "Choose model", exact: true }).click()
  await page.getByRole("textbox", { name: "Search models" }).fill("Private")
  await page.getByText("No matching models", { exact: true }).waitFor()
  await page.getByRole("button", { name: "All models", exact: true }).click()
  await page
    .getByRole("menuitem")
    .filter({ hasText: "Private fixture" })
    .waitFor()
  await page.keyboard.press("Escape")
  await page
    .getByRole("button", { name: "Thinking budget", exact: true })
    .click()
  await page.getByRole("menuitem").filter({ hasText: "High" }).click()
  assert.match(
    await page
      .getByRole("button", { name: "Thinking budget", exact: true })
      .innerText(),
    /High/
  )
  await editor.fill("/plan add Browser check")
  await editor.press("Enter")
  await page
    .locator(".composer-plan")
    .getByText("Browser check", { exact: true })
    .first()
    .waitFor()
  await editor.fill("/plan edit")
  await editor.press("Enter")
  await page.getByRole("button", { name: "Save plan", exact: true }).waitFor()
  await editor.fill("/temperature")
  await page.locator('[data-slot="slash-command-menu"]').waitFor()
  assert.equal(
    await page
      .getByRole("button", { name: "Save plan", exact: true })
      .isVisible(),
    false
  )
  await editor.press("Escape")
  await page.getByRole("button", { name: "Save plan", exact: true }).waitFor()
  await page.getByRole("button", { name: "Cancel", exact: true }).click()
  const rail = page.getByRole("navigation", { name: "Conversation questions" })
  await rail.waitFor({ state: "visible" })
  assert.equal(await rail.locator("button").count(), 120)
  assert.ok(
    await rail.evaluate((node) => node.clientHeight < node.scrollHeight)
  )
  const active = rail.locator('[aria-current="location"]')
  await active.hover()
  await page.locator(".fluid-chat-preview").waitFor({ state: "visible" })
  assert.match(
    await page.locator(".fluid-chat-preview").innerText(),
    /Question.*Reply/s
  )
  await active.focus()
  await page.keyboard.press("ArrowUp")
  await page.keyboard.press("Enter")
  await page.keyboard.press("Escape")
  assert.equal(await page.locator(".fluid-chat-preview").isVisible(), false)
  await page.emulateMedia({ reducedMotion: "reduce" })
  assert.equal(
    await rail
      .locator("button span")
      .first()
      .evaluate((node) => getComputedStyle(node).transitionDuration),
    "0s"
  )
  await page
    .getByRole("button", { name: "Dismiss notification", exact: true })
    .first()
    .waitFor({ state: "visible" })
  await page.keyboard.press("Control+Alt+n")
  const dialog = page.getByRole("dialog")
  await dialog.getByText("Saved notice", { exact: true }).waitFor()
  await dialog.getByText("Saved error", { exact: true }).waitFor()
  await dialog
    .getByRole("button", { name: "Dismiss Saved notice", exact: true })
    .click()
  assert.equal(
    await page.evaluate(
      (key) =>
        JSON.parse(localStorage.getItem(key)).items.some(
          (item) => item.id === "saved-timed"
        ),
      key
    ),
    false
  )
  await dialog.getByRole("button", { name: "Close", exact: true }).click()
  await page.evaluate(() => {
    localStorage.setItem("auth_token", "replacement-bearer-same-account")
    window.dispatchEvent(new Event("session-changed"))
  })
  await page.keyboard.press("Control+Alt+n")
  await page
    .getByRole("dialog")
    .getByText("Saved error", { exact: true })
    .waitFor()
  assert.equal(
    await page
      .getByRole("dialog")
      .getByText("Saved notice", { exact: true })
      .count(),
    0
  )
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Dismiss notification", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Dismiss notification", exact: true })
    .waitFor({ state: "hidden" })
  await page.waitForTimeout(800)
  assert.equal(
    await page
      .getByRole("button", { name: "Dismiss notification", exact: true })
      .count(),
    0
  )
  assert.equal(
    await page.evaluate(
      (key) => JSON.parse(localStorage.getItem(key)).items.length,
      key
    ),
    0
  )
  await page.keyboard.press("Control+k")
  await page.getByRole("tab", { name: "Profile", exact: true }).click()
  await page.getByLabel("New password", { exact: true }).fill("fixture-a")
  await page
    .getByLabel("Confirm new password", { exact: true })
    .fill("fixture-b")
  await page
    .getByRole("button", { name: "Update password", exact: true })
    .click()
  assert.equal(
    await page
      .getByLabel("Confirm new password", { exact: true })
      .getAttribute("aria-invalid"),
    "true"
  )
  assert.equal(
    await page
      .getByLabel("Confirm new password", { exact: true })
      .evaluate((node) => node === document.activeElement),
    true
  )
  assert.equal(
    await page.locator("#password-confirmation-error .sr-only").count(),
    1
  )
  await page
    .getByRole("dialog")
    .getByRole("button", { name: "Close", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Dismiss notification", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Dismiss notification", exact: true })
    .waitFor({ state: "hidden" })
  await page.keyboard.press("Control+k")
  await page.getByRole("tab", { name: "Usage", exact: true }).click()
  await page.getByRole("button", { name: "Try again", exact: true }).waitFor()
  assert.equal(
    await page
      .getByRole("dialog")
      .getByText("Failed to load usage.", { exact: true })
      .count(),
    0
  )
  failUsage = false
  await page.getByRole("button", { name: "Try again", exact: true }).click()
  await page
    .getByRole("dialog")
    .getByText("Requests", { exact: true })
    .waitFor()
  assert.ok(usageReads >= 2)
  assert.deepEqual(errors, [])
  console.log(
    "PASS: footer search/thinking, plan slash restoration, bounded fluid rail pointer/keyboard/reduced motion, notification restore/dismiss/session identity"
  )
} finally {
  await browser.close()
}
