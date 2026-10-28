import { chromium } from "playwright"
import assert from "node:assert/strict"
const browser = await chromium.launch({
  executablePath: process.env.CHROMIUM_PATH,
  headless: true,
  args: ["--no-sandbox"],
})
const page = await browser.newPage()
const external = []
const origin = process.env.AUTHORING_URL ?? "http://localhost:3011"
try {
  page.on("request", (request) => {
    if (!request.url().startsWith(origin) && !request.url().startsWith("data:"))
      external.push(request.url())
  })
  await page.goto(`${origin}/markup/authoring`)
  await page.getByRole("button", { name: "Check runtime decoder" }).click()
  assert.equal(
    await page.getByLabel("Runtime decoder result").innerText(),
    "passed"
  )
  const python = page.getByRole("textbox", {
    name: "Fixture Python source",
    exact: true,
  })
  await python.waitFor({ timeout: 60000 })
  await python.focus()
  await page.keyboard.press("Control+End")
  await page.keyboard.type("\nfixture_value = 3")
  await page
    .getByLabel("Python draft", { exact: true })
    .filter({ hasText: "fixture_value = 3" })
    .waitFor()
  const yaml = page.getByRole("textbox", {
    name: "Fixture YAML source",
    exact: true,
  })
  await yaml.focus()
  await page.keyboard.press("Control+End")
  await page.keyboard.type("\nname: fixture")
  await page
    .getByLabel("YAML draft", { exact: true })
    .filter({ hasText: "name: fixture" })
    .waitFor()
  const locked = page.getByRole("textbox", {
    name: "Fixture read-only source",
    exact: true,
  })
  await locked.focus()
  await page.keyboard.press("Control+End")
  await page.keyboard.type("MUTATED")
  assert.equal(
    await page
      .locator(".monaco-editor")
      .nth(2)
      .innerText()
      .then((text) => text.includes("MUTATED")),
    false
  )
  const processor = page.getByRole("textbox", {
    name: "Processor Python source",
    exact: true,
  })
  await processor.focus()
  await page.keyboard.press("Control+End")
  await page.keyboard.type("\nprocessor_edit = True")
  await page
    .getByLabel("Processor draft", { exact: true })
    .filter({ hasText: "processor_edit = True" })
    .waitFor()
  await page.getByLabel("width", { exact: true }).fill("48")
  await page.getByRole("switch", { name: "enabled" }).click()
  await page.getByRole("combobox", { name: "mode" }).click()
  await page.getByRole("option", { name: "eval", exact: true }).click()
  assert.deepEqual(
    JSON.parse(
      await page.getByLabel("Parameter draft", { exact: true }).innerText()
    ),
    { width: 48, enabled: true, mode: "eval" }
  )
  assert.equal(
    external.some((url) => /monaco|cdn/.test(url)),
    false
  )
  console.log(
    JSON.stringify({
      decoder: "passed",
      python: "edited",
      yaml: "edited",
      readOnly: "preserved",
      parameters: "typed and interactive",
      externalEditorRequests: external.filter((url) => /monaco|cdn/.test(url)),
    })
  )
} finally {
  await browser.close()
}
