import { chromium } from "playwright"
import assert from "node:assert/strict"

const origin = process.env.NOTEBOOK_URL ?? "http://localhost:3000"
const account = "30000000-0000-4000-8000-000000000001"
const timestamp = "2026-10-27T12:00:00Z"
const model = {
  model_id: "30000000-0000-4000-8000-000000000003",
  name: "A deliberately long model name for wrapping catalogue rows without losing identity",
  description: "A measured catalogue fixture",
  family: "Research",
  size: null,
  context: 64,
  has_weights: true,
  parameters: { config: { mode: "pretrain" } },
  published: true,
  created_at: timestamp,
  architecture: null,
  vocab_size: null,
  parent_id: null,
  checkpoint_index: null,
  checkpoint_step: null,
  run_id: null,
}
let paused = false
const mutations = []
const reads = []
const errors = []
const passed = []

/** Supplies one durable execution envelope with distinct queue/history identities. */
function run(index, status = "done") {
  return {
    run_id: `40000000-0000-4000-8000-${String(index).padStart(12, "0")}`,
    kind: index % 2 ? "train" : "tokenize",
    status,
    label: `${status === "done" ? "Historical" : "Current"} job ${index}`,
    config: {},
    progress: status === "running" ? 0.35 : 0,
    stage: status === "running" ? "Optimizing" : null,
    ref_id: null,
    output: { type: "model", id: null },
    result: null,
    depends_on: null,
    error: null,
    created_at: timestamp,
    started_at: timestamp,
    finished_at: status === "done" ? "2026-10-27T12:01:30Z" : null,
  }
}

/** Intercepts every API request in this context; queue mutations update fixture state only. */
async function fixture(route) {
  const request = route.request()
  const url = new URL(request.url())
  const path = url.pathname
  const method = request.method()
  reads.push(`${method} ${path}${url.search}`)
  if (method !== "GET" && method !== "OPTIONS")
    mutations.push({ path, method, body: request.postData() })
  const json = (value) => route.fulfill({ json: value })
  if (path === "/user/me")
    return json({ id_user: account, username: "agents", access_level: "admin" })
  if (path === "/user/me/settings") return json({ settings: {} })
  if (path === "/user/me/composer-choice")
    return json({ composer_choice: { model: "", budget: "auto" } })
  if (path === "/projects")
    return json({ projects: [], total: 0, limit: 50, offset: 0 })
  if (path === "/chats") return json({ chats: [] })
  if (path === "/models") return json({ models: [model] })
  if (path.endsWith("/checkpoints")) return json({ checkpoints: [] })
  if (path === "/tokenizers") return json({ tokenizers: [] })
  if (path === "/datasets") return json({ datasets: [] })
  if (path === "/data/datasets") return json({ datasets: [] })
  if (path === "/data/runs") return json({ runs: [] })
  if (path === "/architectures" || path === "/blocks")
    return json({ items: [], next_cursor: null })
  if (path === "/gym") return json([])
  if (path === "/runs") {
    const offset = Number(url.searchParams.get("offset") ?? 0)
    return json({
      active: [
        run(80, "running"),
        run(81, "queued"),
        run(82, "queued"),
        { ...run(83, "waiting"), depends_on: run(81).run_id },
      ],
      history: Array.from({ length: 10 }, (_, index) =>
        run(offset + index + 1)
      ),
      history_offset: offset,
      history_limit: 25,
      history_total: 40,
      paused,
    })
  }
  if (path === "/runs/pause") {
    paused = url.searchParams.get("paused") === "true"
    return json({ paused })
  }
  if (
    (path.startsWith("/runs/") && method === "PATCH") ||
    path.endsWith("/cancel")
  )
    return json({})
  return route.fulfill({
    status: 404,
    json: { error: "Unneeded fixture endpoint" },
  })
}

/** Records checked user-visible behavior without logging account credentials. */
function record(name) {
  passed.push(name)
  console.log(`passed: ${name}`)
}

/** Rejects page-level horizontal overflow at the current viewport. */
async function fits(page) {
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth
    ),
    true
  )
}

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH ?? "/usr/bin/chromium",
  headless: true,
  args: ["--no-sandbox"],
})
const context = await browser.newContext({
  viewport: { width: 1440, height: 1000 },
  reducedMotion: "reduce",
})
await context.route("http://localhost:8080/**", fixture)
await context.routeWebSocket("ws://localhost:8080/**", (socket) =>
  socket.close()
)
await context.addInitScript(() => {
  localStorage.setItem("auth_token", "opaque-slate-design-fixture")
  localStorage.setItem("access_level", "admin")
  localStorage.setItem("theme", "light")
})
const page = await context.newPage()
page.setDefaultTimeout(60000)
page.setDefaultNavigationTimeout(60000)
page.on("pageerror", (error) => errors.push(error.message))
try {
  await page.goto(`${origin}/models`, { waitUntil: "domcontentloaded" })
  const search = page.getByRole("searchbox", {
    name: "Search models",
    exact: true,
  })
  await search.waitFor()
  await page.getByRole("heading", { name: model.name, exact: true }).waitFor()
  assert.equal(
    await search.evaluate(
      (element) => getComputedStyle(element).borderBottomWidth
    ),
    "1px"
  )
  assert.equal(
    await page.locator('main ul[data-density="open"] > li').count(),
    1
  )
  await search.fill("no-match")
  await page.getByText("Nothing matches that.", { exact: false }).waitFor()
  await search.fill("")
  await fits(page)
  record("shared model row, underlined search and filtering")
  await page.getByRole("button", { name: "Select", exact: true }).click()
  await page.getByRole("checkbox").check()
  await page.getByRole("button", { name: "Delete 1", exact: true }).click()
  const deletion = page.getByRole("alertdialog")
  await deletion.waitFor()
  assert.equal(mutations.filter((item) => item.method === "DELETE").length, 0)
  await deletion.getByRole("button", { name: "Cancel", exact: true }).click()
  await page.getByRole("button", { name: "Done", exact: true }).click()
  record("model selection keeps explicit bulk deletion confirmation")

  await page.goto(`${origin}/projects`, { waitUntil: "domcontentloaded" })
  await page
    .getByRole("heading", { name: "No projects yet", exact: true })
    .waitFor()
  assert.ok(
    (await page
      .locator('main [data-slot="section-texture"][data-tone="warm"]')
      .count()) > 0
  )
  record("textured collection empty state")

  await page.goto(`${origin}/admin`, { waitUntil: "domcontentloaded" })
  const workbench = page.getByRole("region", { name: "Workbench", exact: true })
  await workbench.waitFor()
  await page
    .getByRole("link", { name: "Historical job 1", exact: true })
    .waitFor()
  assert.equal(
    await page.getByRole("link", { name: /^Historical job/ }).count(),
    5
  )
  await page.getByRole("button", { name: "Full history", exact: true }).click()
  assert.equal(
    await page.getByRole("link", { name: /^Historical job/ }).count(),
    10
  )
  await page.getByRole("button", { name: "Recent only", exact: true }).click()
  await page.getByRole("button", { name: "Older jobs", exact: true }).click()
  await page.getByText("Page 2 of 2", { exact: true }).waitFor()
  await page
    .getByRole("link", { name: "Historical job 26", exact: true })
    .waitFor()
  assert.ok((await page.locator("main").last().innerText()).includes("1m 30s"))
  const firstDisclosure = workbench.locator("details").first()
  await firstDisclosure.locator("summary").click()
  assert.equal(await firstDisclosure.getAttribute("open"), null)
  await firstDisclosure.locator("summary").focus()
  await page.keyboard.press("Enter")
  assert.notEqual(await firstDisclosure.getAttribute("open"), null)
  assert.equal(
    await workbench
      .getByRole("link", { name: "Fine-tune", exact: true })
      .getAttribute("href"),
    "/train/new-train?mode=sft"
  )
  assert.equal(
    await workbench
      .getByRole("link", { name: "Language v3 Legacy", exact: true })
      .getAttribute("href"),
    "/bench?suite=language_v3"
  )
  const interfaceGroup = workbench
    .locator("details")
    .filter({ has: page.locator("summary", { hasText: /^Interface$/ }) })
  assert.equal(await interfaceGroup.getAttribute("open"), null)
  const queueBox = await page
    .getByRole("heading", { name: "Queue", exact: true })
    .boundingBox()
  assert.ok((await workbench.boundingBox()).x > queueBox.x)
  assert.equal(
    await workbench
      .getByRole("link", { name: "Blocks", exact: true })
      .getAttribute("href"),
    "/arch/blocks"
  )
  assert.equal(
    await workbench
      .getByRole("link", { name: "Processors", exact: true })
      .getAttribute("href"),
    "/processors"
  )
  assert.ok(
    (await page.locator("main").last().innerText()).includes(
      "after Current job 81"
    )
  )
  record(
    "Admin columns, compact/full/pageable history, metadata and keyboard Workbench disclosures"
  )
  await page
    .getByRole("button", { name: "Options for Current job 82", exact: true })
    .click()
  await page.getByRole("menuitem", { name: "Run next", exact: true }).click()
  assert.ok(
    mutations.some(
      (item) =>
        item.method === "PATCH" && item.path === `/runs/${run(82).run_id}`
    )
  )
  await page
    .locator('[data-slot="switch"][aria-label="Pause the queue"]')
    .click()
  await page
    .locator('[data-slot="switch"][aria-label="Resume the queue"]')
    .waitFor()
  await page
    .getByRole("link", { name: "Current job 80", exact: true })
    .first()
    .waitFor()
  record(
    "Admin columns, compact/full/pageable history, metadata, keyboard disclosures and dispatch pause"
  )

  await page.setViewportSize({ width: 390, height: 844 })
  await fits(page)
  assert.ok(
    (await workbench.boundingBox()).y >
      (
        await page
          .getByRole("heading", { name: "History", exact: true })
          .boundingBox()
      ).y
  )
  await page.screenshot({
    path: "/tmp/slate-design-admin-mobile.png",
    fullPage: true,
  })
  record("Admin responsive stacking without overflow")

  await page.goto(`${origin}/train/new-train?mode=sft`, {
    waitUntil: "domcontentloaded",
  })
  await page.getByRole("textbox", { name: "Model name", exact: true }).waitFor()
  assert.equal(
    await page
      .getByRole("button", { name: "Fine-tune", exact: true })
      .getAttribute("aria-pressed"),
    "true"
  )
  await page
    .getByRole("textbox", { name: "Model name", exact: true })
    .fill("Retained fine-tune draft")
  await page.waitForFunction(
    (account) =>
      Boolean(localStorage.getItem(`slate-training-draft:${account}`)),
    account
  )
  await page.goto(`${origin}/train/new-train?mode=pretrain`, {
    waitUntil: "domcontentloaded",
  })
  await page
    .getByRole("button", { name: "Switch to pre-training", exact: true })
    .waitFor()
  assert.equal(
    await page
      .getByRole("textbox", { name: "Model name", exact: true })
      .inputValue(),
    "Retained fine-tune draft"
  )
  assert.equal(
    await page
      .getByRole("button", { name: "Fine-tune", exact: true })
      .getAttribute("aria-pressed"),
    "true"
  )
  await page
    .getByRole("button", { name: "Switch to pre-training", exact: true })
    .click()
  assert.equal(
    await page
      .getByRole("button", { name: "Pre-train", exact: true })
      .getAttribute("aria-pressed"),
    "true"
  )
  await fits(page)
  record("separate training entries preserve drafts until explicit mode switch")
  assert.deepEqual(errors, [])
  console.log(JSON.stringify({ passed, mutations, pageErrors: errors }))
} catch (error) {
  console.error(
    JSON.stringify({
      url: page.url(),
      passed,
      errors,
      reads: reads.slice(-20),
      text: (await page.locator("main").last().innerText()).slice(-2500),
    })
  )
  throw error
} finally {
  await browser.close()
}
