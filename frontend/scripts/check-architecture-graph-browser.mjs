import { chromium } from "playwright"
import assert from "node:assert/strict"
import { mkdir } from "node:fs/promises"
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  headless: true,
  args: ["--no-sandbox"],
})
const origin = process.env.AUTHORING_URL ?? "http://localhost:3011"
const screenshotDirectory =
  process.env.GRAPH_SCREENSHOT_DIR ?? "/tmp/llm-architecture-graph"
try {
  await mkdir(screenshotDirectory, { recursive: true })
  const page = await browser.newPage({
    viewport: { width: 1280, height: 1100 },
    reducedMotion: "reduce",
  })
  await page.goto(`${origin}/markup/graph`)
  const graph = page.getByRole("region", { name: "Model graph", exact: true })
  await page.getByRole("list", { name: "Graph nodes", exact: true }).waitFor()
  assert.match(
    await page.getByRole("heading", { level: 1 }).innerText(),
    /Synthetic/
  )
  const first = page.getByRole("button", {
    name: "Shared pass one · root/repeat/0",
    exact: true,
  })
  await first.focus()
  await page.keyboard.press("Enter")
  await page
    .getByRole("region", { name: "Selected Shared pass one", exact: true })
    .waitFor()
  assert.match(await graph.innerText(), /float32/)
  assert.match(await graph.innerText(), /Unavailable/)
  const canvas = page.getByRole("region", { name: /Graph canvas/ })
  const transform = canvas.locator("svg > g")
  const canvasBox = await canvas.boundingBox()
  const dragBefore = await transform.getAttribute("transform")
  await page.mouse.move(canvasBox.x + 20, canvasBox.y + 20)
  await page.mouse.down()
  await page.mouse.move(canvasBox.x + 60, canvasBox.y + 50)
  await page.mouse.up()
  assert.notEqual(await transform.getAttribute("transform"), dragBefore)
  await page.getByRole("button", { name: "Fit graph", exact: true }).click()
  const before = await transform.getAttribute("transform")
  await canvas.focus()
  await page.keyboard.press("ArrowRight")
  assert.notEqual(await transform.getAttribute("transform"), before)
  await page.getByRole("button", { name: "Zoom in", exact: true }).click()
  assert.match(await graph.innerText(), /125%/)
  await page.getByRole("button", { name: "Fit graph", exact: true }).click()
  assert.equal(await transform.getAttribute("transform"), before)
  await canvas.locator('[data-node-id="shared-b"] rect').click()
  await page
    .getByRole("region", { name: "Selected Shared pass two", exact: true })
    .waitFor()
  await page
    .getByRole("button", {
      name: "Independent layer · root/stack/1",
      exact: true,
    })
    .click()
  await page
    .getByRole("button", { name: "blocks/core.py:12", exact: true })
    .click()
  assert.match(
    await page.getByLabel("Selected source", { exact: true }).innerText(),
    /:12$/
  )
  await page
    .getByRole("button", { name: "Open Repeated core", exact: true })
    .click()
  await page
    .getByRole("button", { name: "Open Nested repeat", exact: true })
    .waitFor()
  await page
    .getByRole("button", { name: "Open Nested repeat", exact: true })
    .click()
  await page.getByText("root/repeat/nested", { exact: true }).waitFor()
  await page.getByRole("button", { name: "Root scope", exact: true }).click()
  await page
    .getByRole("button", { name: "Next graph page", exact: true })
    .click()
  await page
    .getByRole("list", { name: "Graph nodes", exact: true })
    .locator("li")
    .waitFor()
  assert.equal(
    await page
      .getByRole("list", { name: "Graph nodes", exact: true })
      .locator("li")
      .count(),
    1
  )
  await page.getByRole("combobox", { name: "Graph mode", exact: true }).click()
  await page.getByRole("option", { name: "Training", exact: true }).click()
  await page.getByRole("combobox", { name: "Graph view", exact: true }).click()
  await page
    .getByRole("option", { name: "Captured operations", exact: true })
    .click()
  for (const theme of ["light", "dark"]) {
    await page.evaluate((value) => localStorage.setItem("theme", value), theme)
    await page.reload()
    await page.getByRole("list", { name: "Graph nodes", exact: true }).waitFor()
    await page.screenshot({
      path: `${screenshotDirectory}/${theme}.png`,
      fullPage: true,
    })
    assert.equal(
      await page.evaluate(() =>
        document.documentElement.classList.contains("dark")
      ),
      theme === "dark"
    )
  }
  await page.setViewportSize({ width: 390, height: 844 })
  await page.screenshot({
    path: `${screenshotDirectory}/mobile.png`,
    fullPage: true,
  })
  assert.equal(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth
    ),
    true
  )
  console.log(
    JSON.stringify({
      source: "synthetic fixture",
      keyboard: "passed",
      scopes: "nested and paginated",
      inspector: "shape/dtype/source/absent metrics",
      themes: "light/dark",
      reducedMotion: "reduce",
      screenshots: screenshotDirectory,
      actualCaptureGate: "pending ISS-54",
    })
  )
} finally {
  await browser.close()
}
