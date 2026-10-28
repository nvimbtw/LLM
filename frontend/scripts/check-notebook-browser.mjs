import { chromium } from "playwright"
import assert from "node:assert/strict"
import { createHash } from "node:crypto"

const origin = process.env.NOTEBOOK_URL ?? "http://localhost:3000"
const ids = {
  account: "30000000-0000-4000-8000-000000000001",
  project: "30000000-0000-4000-8000-000000000002",
  model: "30000000-0000-4000-8000-000000000003",
  tokenizer: "30000000-0000-4000-8000-000000000004",
  architecture: "30000000-0000-4000-8000-000000000005",
  chat: "30000000-0000-4000-8000-000000000006",
  dataset: "30000000-0000-4000-8000-000000000007",
  attachment: "30000000-0000-4000-8000-000000000008",
}
const timestamp = "2026-10-27T12:00:00Z"
const project = {
  project_id: ids.project,
  name: "Notebook project",
  description: "Project description",
  notes: "Saved notes",
  thread_count: 0,
  created_at: timestamp,
  updated_at: timestamp,
}
const model = {
  model_id: ids.model,
  name: "Notebook model",
  description: "Model description",
  family: "Research",
  size: null,
  context: 64,
  has_weights: false,
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
const tokenizer = {
  tokenizer_id: ids.tokenizer,
  name: "Notebook vocabulary",
  description: "Vocabulary description",
  implicit: false,
  vocab_size: 1024,
  merges: 0,
  files: 1,
  bytes: 100,
  datasets: [],
  has_vocab: true,
  models: [],
}
const summary = {
  resource_id: ids.architecture,
  kind: "architecture",
  origin: "custom",
  name: "Notebook architecture",
  description: "Architecture description",
  archived: false,
  draft_version: 1,
  metadata_version: 1,
  accepted_revision: null,
}
const chat = {
  chat_id: ids.chat,
  title: "Notebook conversation",
  created_at: timestamp,
  tokens: 0,
  project_id: null,
}
const dataset = {
  id: ids.dataset,
  name: "Notebook corpus",
  description: null,
  kind: "pretrain",
  origin: "huggingface",
  status: "ready",
  source_repo: "fixture/corpus",
  source_config: null,
  source_split: null,
  parent_id: null,
  steps: null,
  document_count: 2,
  byte_size: 100,
  created_at: timestamp,
  updated_at: timestamp,
}
let yaml = "name: notebook\n"
let datasetPresent = true
let failSave = true
let failRename = true
let failDelete = true
const mutations = []
const errors = []
const passed = []

/** Computes the exact saved-source identity expected by the existing editor contract. */
function digest(text) {
  return createHash("sha256").update(text).digest("hex")
}

/** Serves deterministic existing API contracts; no fixture mutation reaches a running backend or service. */
async function fixture(route) {
  const request = route.request()
  const path = new URL(request.url()).pathname
  const method = request.method()
  const json = (value, status = 200) => route.fulfill({ json: value, status })
  if (method !== "GET" && method !== "OPTIONS") mutations.push({ path, method })
  if (path === "/user/me")
    return json({
      id_user: ids.account,
      username: "agents",
      access_level: "admin",
    })
  if (path === "/user/me/settings") return json({ settings: {} })
  if (path === "/user/me/composer-choice")
    return json({ composer_choice: { model: "", budget: "auto" } })
  if (path === "/runs")
    return json({
      active: [],
      history: [],
      history_offset: 0,
      history_limit: 25,
      history_total: 0,
      paused: false,
    })
  if (path === "/projects")
    return json({ projects: [project], total: 1, limit: 50, offset: 0 })
  if (path === `/projects/${ids.project}`) {
    if (method === "PATCH") {
      if (failSave)
        return route.fulfill({ status: 500, body: "Fixture save rejected" })
      Object.assign(project, request.postDataJSON())
      return json(project)
    }
    return json({
      ...project,
      threads: [],
      threads_total: 0,
      threads_limit: 50,
      threads_offset: 0,
    })
  }
  if (path === "/models") return json({ models: [model] })
  if (path === `/models/${ids.model}`) {
    if (method === "PATCH") {
      if (failSave)
        return route.fulfill({ status: 500, body: "Fixture save rejected" })
      const fields = request.postDataJSON()
      Object.assign(model, {
        name: fields.display_name ?? model.name,
        description: fields.description ?? model.description,
        family: fields.family ?? model.family,
        size: fields.size ?? null,
      })
    }
    return json(model)
  }
  if (path.endsWith("/checkpoints")) return json({ checkpoints: [] })
  if (path === "/tokenizers") return json({ tokenizers: [tokenizer] })
  if (path === `/tokenizers/${ids.tokenizer}`) {
    if (method === "PATCH") {
      if (failSave)
        return route.fulfill({ status: 500, body: "Fixture save rejected" })
      Object.assign(tokenizer, request.postDataJSON())
    }
    return json(tokenizer)
  }
  if (path === "/architectures" || path === "/blocks")
    return json({
      items: path === "/architectures" ? [summary] : [],
      next_cursor: null,
    })
  if (path === `/architectures/${ids.architecture}`) {
    if (method === "PATCH") {
      if (failSave)
        return route.fulfill({ status: 500, body: "Fixture save rejected" })
      const { expected_metadata_version, ...fields } = request.postDataJSON()
      assert.equal(expected_metadata_version, summary.metadata_version)
      Object.assign(summary, fields, {
        name: fields.name.trim(),
        metadata_version: summary.metadata_version + 1,
      })
    }
    return json({ summary, revision_count: 0, parameters: {} })
  }
  if (path === `/architectures/${ids.architecture}/draft`) {
    if (method === "PUT") {
      yaml = request.postDataJSON().yaml
      summary.draft_version += 1
      return json({
        resource_id: ids.architecture,
        draft_version: summary.draft_version,
        content_hash: digest(yaml),
      })
    }
    return json({
      resource_id: ids.architecture,
      draft_version: summary.draft_version,
      content_hash: digest(yaml),
      yaml,
    })
  }
  if (path.endsWith("/revisions")) return json({ items: [], next_cursor: null })
  if (path === "/chats") return json({ chats: [chat] })
  if (path === `/chats/${ids.chat}`) {
    if (method === "PATCH") {
      if (failRename)
        return route.fulfill({ status: 500, body: "Fixture rename rejected" })
      chat.title = request.postDataJSON().title
      return json(chat)
    }
    return json({ ...chat, messages: [], total_tokens: 0 })
  }
  if (path === "/data/datasets")
    return json({ datasets: datasetPresent ? [dataset] : [] })
  if (path === `/data/datasets/${ids.dataset}`) {
    if (method === "DELETE") {
      if (failDelete)
        return route.fulfill({ status: 500, body: "Fixture deletion rejected" })
      datasetPresent = false
      return json({})
    }
    return json(dataset)
  }
  if (path.endsWith("/documents")) return json({ documents: [], total: 0 })
  if (path === "/data/runs") return json({ runs: [] })
  if (path === "/gym")
    return json([
      {
        id: "general",
        name: "General",
        description: "Mix all sources",
        tools: [],
      },
      { id: "math", name: "Math", description: "Math practice", tools: [] },
    ])
  if (path === "/attachments")
    return json({
      attachment_id: ids.attachment,
      filename: "notes.txt",
      media_type: "text/plain",
      size: 5,
    })
  if (path === `/attachments/${ids.attachment}`) return json({})
  return route.fulfill({
    status: 404,
    json: { error: "Fixture endpoint not needed by this interaction" },
  })
}

/** Records a completed browser interaction without exposing private bearer data. */
function record(name) {
  passed.push(name)
  console.log(`passed: ${name}`)
}

const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
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
  localStorage.setItem("auth_token", "opaque-notebook-fixture-session")
  localStorage.setItem("access_level", "admin")
  localStorage.setItem("theme", "light")
  const supports = CSS.supports.bind(CSS)
  CSS.supports = (...args) =>
    args[0] === "field-sizing" ? false : supports(...args)
})
const page = await context.newPage()
page.setDefaultTimeout(15000)
page.on("pageerror", (error) => errors.push(error.message))

try {
  await page.goto(`${origin}/projects/${ids.project}`)
  const projectName = page.getByRole("textbox", {
    name: "Project name",
    exact: true,
  })
  await projectName.waitFor()
  await projectName.fill("Unfinished project")
  const notes = page.getByRole("textbox", {
    name: "Notes and instructions",
    exact: true,
  })
  await notes.fill(
    Array.from(
      { length: 20 },
      (_, index) => `Line ${index + 1}: keep this note`
    ).join("\n")
  )
  assert.equal(
    await notes.evaluate(
      (element) => element.clientHeight >= element.scrollHeight - 1
    ),
    true
  )
  await page.getByRole("link", { name: "Models", exact: true }).first().click()
  await page.waitForURL(`${origin}/models`)
  await page.goBack()
  await projectName.waitFor()
  assert.equal(await projectName.inputValue(), "Unfinished project")
  await page.reload()
  await projectName.waitFor()
  assert.equal(await projectName.inputValue(), "Unfinished project")
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  await page.locator("main").getByRole("alert").waitFor()
  assert.equal(await projectName.inputValue(), "Unfinished project")
  failSave = false
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  await page
    .getByRole("button", { name: "Save changes", exact: true })
    .waitFor({ state: "hidden" })
  await projectName.fill("Discard this")
  await page.getByRole("button", { name: "Cancel", exact: true }).click()
  assert.equal(await projectName.inputValue(), "Unfinished project")
  record(
    "project Back/reload recovery, failed save, retry, Cancel and multiline fallback"
  )

  await page.evaluate(() => {
    const original = Storage.prototype.setItem
    window.restoreNotebookStorage = () => {
      Storage.prototype.setItem = original
    }
    Storage.prototype.setItem = function (key, value) {
      if (this === sessionStorage)
        throw new Error("Fixture draft storage unavailable")
      return original.call(this, key, value)
    }
  })
  await projectName.fill("Unretained project edit")
  await page.locator("main").getByRole("alert").waitFor()
  let lossWarnings = 0
  page.once("dialog", async (dialog) => {
    lossWarnings += 1
    await dialog.dismiss()
  })
  await page.getByRole("link", { name: "Models", exact: true }).first().click()
  assert.equal(page.url(), `${origin}/projects/${ids.project}`)
  assert.equal(await projectName.inputValue(), "Unretained project edit")
  assert.equal(lossWarnings, 1)
  await page.getByRole("button", { name: "Cancel", exact: true }).click()
  await page.evaluate(() => {
    window.restoreNotebookStorage()
    delete window.restoreNotebookStorage
  })
  record(
    "failed draft retention warns before leaving and preserves declined edits"
  )

  for (const [route, label, original] of [
    [`/models/${ids.model}`, "Model name", model.name],
    [`/tokenizers/${ids.tokenizer}`, "Vocabulary name", tokenizer.name],
  ]) {
    await page.goto(`${origin}${route}`)
    const field = page.getByRole("textbox", { name: label, exact: true })
    await field.waitFor()
    await field.fill(`${original} draft`)
    await page.reload()
    await field.waitFor()
    assert.equal(await field.inputValue(), `${original} draft`)
    await page.getByRole("button", { name: "Cancel", exact: true }).click()
    assert.equal(await field.inputValue(), original)
    await field.focus()
    await page.keyboard.press("Tab")
    await page.keyboard.press("Shift+Tab")
    await page.waitForFunction(
      (label) => {
        const element = Array.from(document.querySelectorAll("input")).find(
          (input) => input.getAttribute("aria-label") === label
        )
        return (
          element === document.activeElement &&
          element?.matches(":focus-visible")
        )
      },
      label,
      { timeout: 5000 }
    )
    assert.deepEqual(
      await field.evaluate((element) => {
        const style = getComputedStyle(element)
        return {
          outline: style.outlineStyle,
          top: style.borderTopWidth,
          right: style.borderRightWidth,
          left: style.borderLeftWidth,
          bottom: style.borderBottomWidth,
        }
      }),
      {
        outline: "none",
        top: "0px",
        right: "0px",
        left: "0px",
        bottom: "0px",
      }
    )
  }
  record("model/vocabulary recovery, Cancel and keyboard focus")

  await page.goto(`${origin}/arch/${ids.architecture}`)
  const architectureName = page.getByRole("textbox", {
    name: "Architecture name",
    exact: true,
  })
  await architectureName.waitFor()
  assert.equal(
    await page.getByText("Resource details", { exact: true }).count(),
    0
  )
  await architectureName.fill("Architecture metadata draft")
  const source = page.getByRole("textbox", {
    name: "Architecture YAML draft",
    exact: true,
  })
  await source.waitFor({ timeout: 60000 })
  await source.focus()
  await page.keyboard.press("Control+End")
  await page.keyboard.type("\n# local source edit")
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  await page
    .getByRole("button", { name: "Save changes", exact: true })
    .waitFor({ state: "hidden" })
  assert.match(
    await page.locator(".monaco-editor .view-lines").innerText(),
    /local\s+source\s+edit/
  )
  await page.reload()
  await architectureName.waitFor()
  assert.equal(
    await architectureName.inputValue(),
    "Architecture metadata draft"
  )
  await source.waitFor({ timeout: 60000 })
  assert.match(
    await page.locator(".monaco-editor .view-lines").innerText(),
    /local\s+source\s+edit/
  )
  await architectureName.fill("Retain metadata while reloading source")
  await page
    .getByRole("button", { name: "Reload server draft", exact: true })
    .click()
  await page.getByRole("button", { name: "Reload source", exact: true }).click()
  await architectureName.waitFor()
  assert.equal(
    await architectureName.inputValue(),
    "Retain metadata while reloading source"
  )
  await source.waitFor({ timeout: 60000 })
  assert.doesNotMatch(
    await page.locator(".monaco-editor .view-lines").innerText(),
    /local\s+source\s+edit/
  )
  record(
    "architecture header, independent metadata/source saves, source recovery and explicit reload"
  )

  await page.goto(`${origin}/conversations`)
  await page
    .locator("main")
    .getByRole("button", { name: "Chat options", exact: true })
    .click()
  await page.getByRole("menuitem", { name: "Rename", exact: true }).click()
  let title = page.getByRole("textbox", {
    name: "Conversation title",
    exact: true,
  })
  await title.fill("Attempted conversation title")
  await title.press("Tab")
  assert.equal(await title.inputValue(), "Attempted conversation title")
  await title.press("Enter")
  await page.locator("main").getByRole("alert").waitFor()
  assert.equal(await title.inputValue(), "Attempted conversation title")
  failRename = false
  await page.getByRole("button", { name: "Save changes", exact: true }).click()
  await title.waitFor({ state: "hidden" })
  await page
    .locator('[data-slot="sidebar-content"]')
    .getByRole("button", { name: "Chat options", exact: true })
    .click()
  await page.getByRole("menuitem", { name: "Rename", exact: true }).click()
  title = page.getByRole("textbox", { name: "Conversation title", exact: true })
  await title.fill("Cancelled sidebar rename")
  await title.press("Escape")
  assert.equal(await title.count(), 0)
  await page.goto(`${origin}/chat/${ids.chat}`)
  await page
    .getByRole("button", { name: "Rename conversation", exact: true })
    .click()
  title = page.getByRole("textbox", { name: "Conversation title", exact: true })
  await title.fill("Composer title")
  await title.press("Enter")
  await title.waitFor({ state: "hidden" })
  assert.equal(chat.title, "Composer title")
  record(
    "all three rename entry points, blur/Enter/Escape and failed-write recovery"
  )

  await page.goto(`${origin}/data/${ids.dataset}`)
  await page
    .getByRole("heading", { name: dataset.name, exact: true, level: 1 })
    .waitFor()
  await page.goto(`${origin}/data`)
  assert.equal(
    await page.getByRole("tab", { name: "Processors", exact: true }).count(),
    0
  )
  await page.getByRole("link", { name: "Processors", exact: true }).waitFor()
  await page
    .getByRole("button", { name: `Delete ${dataset.name}`, exact: true })
    .click()
  const dialog = page.getByRole("alertdialog")
  assert.equal(mutations.filter((entry) => entry.method === "DELETE").length, 0)
  await dialog
    .getByRole("button", { name: "Delete dataset", exact: true })
    .click()
  await dialog.getByRole("alert").waitFor()
  assert.equal(await dialog.isVisible(), true)
  failDelete = false
  await dialog
    .getByRole("button", { name: "Delete dataset", exact: true })
    .click()
  await dialog.waitFor({ state: "hidden" })
  record(
    "dataset identity, route link semantics and confirmed deletion with retry"
  )

  await page.goto(`${origin}/gyms`)
  await page
    .getByRole("link", { name: "Practise General", exact: true })
    .waitFor()
  assert.equal(
    await page
      .getByRole("link", { name: "Evaluate and view history", exact: true })
      .count(),
    2
  )
  const general = page.getByRole("link", {
    name: "Practise General",
    exact: true,
  })
  const math = page.getByRole("link", { name: "Practise Math", exact: true })
  assert.equal(
    Math.round((await general.boundingBox()).x),
    Math.round((await math.boundingBox()).x)
  )
  record("single-column gym hierarchy and explicit evaluation link")

  await page.setViewportSize({ width: 390, height: 844 })
  await page
    .getByRole("button", { name: "Open navigation", exact: true })
    .click()
  await page
    .getByRole("dialog")
    .getByRole("link", { name: "Models", exact: true })
    .waitFor()
  await page.keyboard.press("Escape")
  await page.goto(`${origin}/projects/${ids.project}`)
  await projectName.waitFor()
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth
    ),
    false
  )
  await page.screenshot({
    path: "/tmp/slate-notebook-mobile.png",
    fullPage: true,
  })
  record("mobile navigation and narrow multiline layout")

  const touch = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
    colorScheme: "dark",
  })
  await touch.route("http://localhost:8080/**", fixture)
  await touch.routeWebSocket("ws://localhost:8080/**", (socket) =>
    socket.close()
  )
  await touch.addInitScript(() => {
    localStorage.setItem("auth_token", "opaque-touch-fixture-session")
    localStorage.setItem("access_level", "admin")
    localStorage.setItem("theme", "dark")
  })
  const touchPage = await touch.newPage()
  touchPage.on("pageerror", (error) => errors.push(error.message))
  await touchPage.goto(`${origin}/chat/${ids.chat}`)
  await touchPage.locator('input[type="file"]').setInputFiles({
    name: "notes.txt",
    mimeType: "text/plain",
    buffer: Buffer.from("notes"),
  })
  const removeAttachment = touchPage.getByRole("button", {
    name: "Remove notes.txt",
    exact: true,
  })
  await removeAttachment.waitFor()
  assert.equal(
    await removeAttachment.evaluate(
      (element) => getComputedStyle(element).opacity
    ),
    "1"
  )
  const removeBox = await removeAttachment.boundingBox()
  assert.ok(removeBox.width >= 32 && removeBox.height >= 32)
  await removeAttachment.tap()
  await removeAttachment.waitFor({ state: "hidden" })
  await touchPage.goto(`${origin}/conversations`)
  const options = touchPage
    .locator("main")
    .getByRole("button", { name: "Chat options", exact: true })
  await options.waitFor()
  assert.equal(
    await options.evaluate((element) => getComputedStyle(element).opacity),
    "1"
  )
  assert.ok((await options.boundingBox()).width >= 40)
  await options.tap()
  await touchPage.getByRole("menuitem", { name: "Rename", exact: true }).tap()
  await touchPage
    .getByRole("textbox", { name: "Conversation title", exact: true })
    .waitFor()
  record(
    "dark theme and coarse-pointer attachment/menu visibility and hit areas"
  )

  const reader = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
  })
  await reader.route("http://localhost:8080/**", fixture)
  await reader.addInitScript(() => {
    localStorage.setItem("auth_token", "opaque-reader-fixture-session")
    localStorage.setItem("access_level", "user")
  })
  const readerPage = await reader.newPage()
  await readerPage.goto(`${origin}/models/${ids.model}`)
  await readerPage
    .getByRole("heading", { name: model.name, exact: true, level: 1 })
    .waitFor()
  assert.equal(
    await readerPage
      .getByRole("textbox", { name: "Model name", exact: true })
      .count(),
    0
  )
  assert.equal(
    await readerPage
      .getByRole("button", { name: "Delete", exact: true })
      .count(),
    0
  )
  await readerPage.goto(`${origin}/tokenizers/${ids.tokenizer}`)
  await readerPage
    .getByRole("heading", { name: tokenizer.name, exact: true, level: 1 })
    .waitFor()
  assert.equal(
    await readerPage
      .getByRole("textbox", { name: "Vocabulary name", exact: true })
      .count(),
    0
  )
  assert.equal(
    await readerPage
      .getByRole("button", { name: `Delete ${tokenizer.name}`, exact: true })
      .count(),
    0
  )
  record("ordinary-reader model/vocabulary permissions")

  assert.deepEqual(errors, [])
  console.log(
    JSON.stringify({
      passed,
      mutationCount: mutations.length,
      pageErrors: errors,
    })
  )
} catch (error) {
  console.error(
    JSON.stringify({
      url: page.url(),
      passed,
      errors,
      text: (await page.locator("body").innerText()).slice(-6000),
    })
  )
  throw error
} finally {
  await browser.close()
}
