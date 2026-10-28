import { test } from "node:test"
import assert from "node:assert/strict"
import { encodeArchitecture, decodeArchitecture } from "./decoder"
import { architectureRequest, ArchitectureApiError } from "./http"
import { renderToStaticMarkup } from "react-dom/server"
import { createElement } from "react"
import { ParameterField } from "../../components/authoring/parameter-fields"

test("omitted metadata patch fields remain omitted rather than clearing description", () => {
  assert.deepEqual(
    JSON.parse(
      encodeArchitecture("MetadataPatch", {
        expected_metadata_version: 1,
        name: "Updated",
      })
    ),
    { expected_metadata_version: 1, name: "Updated" }
  )
})
test("multibyte source observes UTF-8 byte limits independently of JSON string length", () => {
  assert.throws(
    () =>
      decodeArchitecture("BlockCreate", {
        name: "large",
        source: "🙂".repeat(65537),
      }),
    /byte budget/
  )
  assert.doesNotThrow(() =>
    decodeArchitecture("BlockCreate", {
      name: "bounded",
      source: "🙂".repeat(65536),
    })
  )
})
test("required unset boolean and enum controls retain an explicit unselected state", () => {
  for (const declaration of [
    { type: "boolean" as const, required: true },
    { type: "enum" as const, required: true, choices: [false, "false"] },
  ]) {
    const markup = renderToStaticMarkup(
      createElement(ParameterField, {
        name: "required",
        declaration,
        value: undefined,
        onChange: () => {},
      })
    )
    assert.match(markup, /Choose a value/)
    assert.doesNotMatch(markup, /aria-checked="false"/)
  }
})
test("Rust transport preserves captured bearer identity, abort signals and non-JSON HTTP errors", async () => {
  const originalFetch = globalThis.fetch
  const requests: RequestInit[] = []
  const signal = new AbortController().signal
  globalThis.fetch = (async (url, init) => {
    assert.equal(url, "http://localhost:8080/blocks")
    requests.push(init!)
    return new Response("proxy unavailable", { status: 502 })
  }) as typeof fetch
  try {
    await assert.rejects(
      architectureRequest("/blocks", "CatalogPage", {
        token: "captured-session",
        signal,
      }),
      (error: unknown) =>
        error instanceof ArchitectureApiError &&
        error.status === 502 &&
        error.detail === null
    )
    assert.equal(
      new Headers(requests[0].headers).get("Authorization"),
      "Bearer captured-session"
    )
    assert.equal(requests[0].signal, signal)
  } finally {
    globalThis.fetch = originalFetch
  }
})

test("nonfinite interpreted JSON numbers are rejected", () => {
  for (const value of [NaN, Infinity, -Infinity])
    assert.throws(() =>
      decodeArchitecture("ParameterDeclaration", {
        type: "number",
        default: value,
      })
    )
})

test("a delayed 401 from a replaced session cannot clear the new session", async () => {
  const originalFetch = globalThis.fetch
  const windowDescriptor = Object.getOwnPropertyDescriptor(globalThis, "window")
  const storageDescriptor = Object.getOwnPropertyDescriptor(
    globalThis,
    "localStorage"
  )
  let clears = 0
  Object.defineProperty(globalThis, "window", { configurable: true, value: {} })
  Object.defineProperty(globalThis, "localStorage", {
    configurable: true,
    value: {
      getItem: () => "replacement-session",
      removeItem: () => {
        clears++
      },
    },
  })
  globalThis.fetch = (async () =>
    new Response("expired", { status: 401 })) as typeof fetch
  try {
    await assert.rejects(
      architectureRequest("/blocks", "CatalogPage", { token: "old-session" }),
      /session has expired/
    )
    assert.equal(clears, 0)
  } finally {
    globalThis.fetch = originalFetch
    if (windowDescriptor)
      Object.defineProperty(globalThis, "window", windowDescriptor)
    else Reflect.deleteProperty(globalThis, "window")
    if (storageDescriptor)
      Object.defineProperty(globalThis, "localStorage", storageDescriptor)
    else Reflect.deleteProperty(globalThis, "localStorage")
  }
})

test("structured 409 diagnostics preserve phase and optimistic version for recovery", async () => {
  const originalFetch = globalThis.fetch
  globalThis.fetch = (async () =>
    Response.json(
      {
        error: {
          code: "stale_draft",
          message: "Reload the current draft.",
          phase: "save",
        },
        current_version: 4,
      },
      { status: 409 }
    )) as typeof fetch
  try {
    await assert.rejects(
      architectureRequest("/blocks", "CatalogPage", { token: "captured" }),
      (error: unknown) =>
        error instanceof ArchitectureApiError &&
        error.status === 409 &&
        error.detail?.error.phase === "save" &&
        error.detail.current_version === 4
    )
  } finally {
    globalThis.fetch = originalFetch
  }
})
